import json
import numpy as np
import faiss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import utils
import redis
import time
import seaborn as sns
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        '--db_name',
        default='db'
    )

    parser.add_argument(
        '--namefig',
        default='tsne'
    )
    args = parser.parse_args()

    index = faiss.read_index(args.db_name)
    r = redis.Redis(host='localhost', port='6379', db=0)
    # Retrieve the vectors from the index
    vectors = index.index.reconstruct_n(0, index.ntotal)

    labels = list(range(index.ntotal)) 
    names = []
    
    for l in labels:
        n = r.get(str(l)).decode('utf-8')
        names.append(utils.get_class(n))    
    
    classes = list(set(names))
    classes.sort()
    conversion = {x: i for i, x in enumerate(classes)}
    int_names = np.array([conversion[n] for n in names])
    # Perform t-SNE on the vectors
    tsne = TSNE(n_components=2, perplexity = 30, method = 'barnes_hut')
    
    t = time.time()
    embeddings = tsne.fit_transform(vectors)
    t_fit = time.time() - t

    
    # Visualize the embeddings
    plt.scatter(embeddings[:,0], embeddings[:,1], c=int_names,cmap='viridis', s=1.5, linewidths=1.5, edgecolors='none')
    plt.colorbar()
    plt.savefig(args.namefig + '.png')
