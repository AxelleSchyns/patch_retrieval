
import faiss
import torch
from dataset.dataset import *
import redis
import time
import gc

import numpy as np


# Class that represents the database
class Database:
    def __init__(self, filename, model, load=False, device='cuda:0'):
        self.num_features = model.num_features
        self.model = model
        self.device = device
        self.filename = filename

        res = faiss.StandardGpuResources()  # Allocation of streams and temporary memory 

        # Load an existing database
        if load:
            print("Loading FAISS index from file")
            try:
                self.index = faiss.read_index(self.filename)
            except RuntimeError as e:
                print(f"Failed to load FAISS index: {e}")
            self.r = redis.Redis(host='localhost', port=6379, db=0)
        else:
            # Create a new database
            print("Creating a new FAISS index")
            self.index = faiss.IndexFlatL2(self.num_features)
            self.index = faiss.IndexIDMap(self.index)
            self.r = redis.Redis(host='localhost', port=6379, db=0)
            self.r.flushdb()
            self.r.set('last_id', 0)  # Set a value in Redis with key = last_id

        if 'cuda' in device:
            print("Transferring index to GPU")
            try:
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("Index transferred to GPU")
            except Exception as e:
                print(f"Failed to transfer index to GPU: {e}")

    # x = vector of images 
    def add(self, x, names):
        last_id = int(self.r.get('last_id').decode('utf-8'))
        self.index.add_with_ids(x, np.arange(last_id, last_id + x.shape[0])) 
        # Previous command not working on cpu ! work when adding one by one

        for n, x_  in zip(names, x):
            self.r.set(str(last_id), n) # Set the name of the image at key = id
            self.r.set(n, str(last_id)) # Set the id of the image at key = name 
            last_id += 1

        self.r.set('last_id', last_id) # Update the last id to take into account the added images

    @torch.no_grad()
    def add_dataset(self, data_root):
        model = self.model
        model_name = self.model.model_name
        model_device = next(model.parameters()).device

        data = AddDataset(data_root, model_name)
        loader = torch.utils.data.DataLoader(
            data, batch_size=32, num_workers=8, pin_memory=True, timeout=30, persistent_workers=False )

        
        t_model, t_indexing, t_transfer = 0, 0, 0

        def log_mem(step=""):
            torch.cuda.empty_cache()
            print(f"[{step}] CUDA mem alloc: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            print(f"[{step}] CUDA mem reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

        for i, (images, filenames) in enumerate(loader):
            if i % 5000 == 0:
                print(f"Batch {i} / {len(loader)}")
                log_mem()

            # Ensure shape
            images = images.view(-1, 3, 224, 224).to(device=model_device, non_blocking=True)

            try:
                # TODO change with a method from models 
                t0 = time.time()

                out = model.encode(images)

                out = out.contiguous()

                torch.cuda.synchronize()
                t_model += time.time() - t0

                # Check for NaNs
                if torch.isnan(out).any():
                    print(f"🚨 NaNs in batch {i}")
                    continue

                # Transfer to CPU
                t = time.time()
                out_np = out.cpu().numpy()
                torch.cuda.synchronize()
                t_transfer += time.time() - t

                # Indexing / saving
                t = time.time()
                self.add(out_np, list(filenames))
                torch.cuda.synchronize()
                t_indexing += time.time() - t

            except RuntimeError as e:
                print(f"⚠️ Runtime error at batch {i}: {e}")
                log_mem(f"Error @ batch {i}")
                continue
            # Optional memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        print(f"✅ Time - model: {t_model:.2f}s | transfer: {t_transfer:.2f}s | indexing: {t_indexing:.2f}s")
        self.save()

    @torch.no_grad()
    def search(self, query_im, nrt_neigh=10):
        model = self.model
        model_device = next(model.parameters()).device

        image = query_im.to(device=model_device, non_blocking=True).reshape(-1, 3, 224, 224)

        t0_model = time.time()

        out = model.encode(image)

        torch.cuda.synchronize()
        t_model = time.time() - t0_model

        # Transfer to CPU
        torch.cuda.synchronize()
        t0_transfer = time.time()
        out = out.cpu().numpy()
        t_transfer = time.time() - t0_transfer

        # Search
        torch.cuda.synchronize()
        t0_search = time.time()
        distance, labels = self.index.search(out, nrt_neigh)
        labels = [l for l in list(labels[0]) if l != -1]
        values = [self.r.get(str(l)).decode("utf-8") for l in labels]
        t_search = time.time() - t0_search

        return values, distance[0], t_model, t_search, t_transfer

    def remove(self, name):
        key = self.r.get(name).decode('utf-8')
        label = int(key)

        idsel = faiss.IDSelectorRange(label, label+1)

        if self.device == 'gpu':
            self.index = faiss.index_gpu_to_cpu(self.index)
        self.index.remove_ids(idsel)
        self.save()
        self.r.delete(key)
        self.r.delete(name)
        if self.device == 'gpu':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def train_index(self):
        batch_size = 128
        x = []
        keys = []
        all_keys = self.r.keys("*")
        for k in all_keys:
            k = k.decode("utf-8")
            # Only keep the indexes as keys, not the names nor last_id 
            if k.find('/') == -1 and k.find('_')==-1:
                end_ind = k.find('l') # Remove the unecessary part of the indeex
                index = k[:end_ind]
                keys.append(index)
                index = int(index)
                vec = self.index.index.reconstruct(index)
                x.append(vec)
        if len(x) >= 10:
            num_clusters = int(np.sqrt(self.index.ntotal))

            self.quantizer = faiss.IndexFlatL2(self.model.num_features)
            self.index = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                                    num_clusters)

            if self.device == 'gpu':
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

            x = np.asarray(x, dtype=np.float32)
            self.index.train(x)
            self.index.nprobe = num_clusters // 10

            num_batches = self.index.ntotal // batch_size

            for i in range(num_batches+1):
                if i == num_batches:
                    x_ = x[i * batch_size:, :]
                    key = keys[i * batch_size:]
                else:
                    x_ = x[i * batch_size: (i + 1) * batch_size, :]
                    key = keys[i * batch_size: (i+1) * batch_size]
                self.index.add_with_ids(x_, np.array(key, dtype=np.int64))

    def save(self):
        if 'cuda' in self.device: # TODO change
            faiss.write_index(faiss.index_gpu_to_cpu(self.index), self.filename)
        else:
            faiss.write_index(self.index, self.filename)

