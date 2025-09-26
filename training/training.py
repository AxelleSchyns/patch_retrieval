import time

from training.loss import *
from dataset.dataset import *
from utils.training_utils import *

def get_optim(model, data, loss, lr, decay, beta_lr, lr_proxies):
        if loss == 'margin':
            loss_function = MarginLoss(model.device, n_classes=len(data.classes))

            to_optim = [{'params':model.parameters(),'lr':lr,'weight_decay':decay},
                        {'params':loss_function.parameters(), 'lr':beta_lr, 'weight_decay':0}]

            optimizer = torch.optim.Adam(to_optim)
        elif loss == 'proxy_nca_pp':
            loss_function = ProxyNCA_prob(len(data.classes), model.num_features, 3, model.device)

            to_optim = [
                {'params':model.parameters(), 'weight_decay':0},
                {'params':loss_function.parameters(), 'lr': lr_proxies}, # Allows to update automaticcaly the proxies vectors when doing a step of the optimizer
            ]

            optimizer = torch.optim.Adam(to_optim, lr=lr, eps=1)
        elif loss == 'softmax':
            loss_function = NormSoftmax(0.05, len(data.classes), model.num_features, lr_proxies, model.device)

            to_optim = [
                {'params':model.parameters(),'lr':lr,'weight_decay':decay},
                {'params':loss_function.parameters(),'lr':lr_proxies,'weight_decay':decay}
            ]

            optimizer = torch.optim.Adam(to_optim)
        elif loss == 'softtriple':
            # Official - paper implementation
            loss_function = SoftTriple(model.device)
            to_optim = [{"params": model.parameters(), "lr": 0.0001},
                                  {"params": loss_function.parameters(), "lr": 0.01}] 
            optimizer = torch.optim.Adam(to_optim, eps=0.01, weight_decay=0.0001)
        else:
            to_optim = [{'params':model.parameters(),'lr':lr,'weight_decay':decay}] # For byol: lr = 3e-4
            optimizer = torch.optim.Adam(to_optim)
            loss_function = None
        return optimizer, loss_function


def train_model(model, loss_name, epochs, training_dir, batch_size = 32, lr = 0.0001, decay = 0.0004, beta_lr = 0.0005, lr_proxies = 0.00001, sched = 'exponential', gamma = 0.3, parallel = False, informative_samp = True, starting_weights = None, epoch_freq = 20, need_val = True):
    # download the dataset
    if need_val:
        data = TrainingDataset(root = training_dir, model_name = model.model_name, samples_per_class= 2,  informative_samp = informative_samp, need_val=2)
        data_val = TrainingDataset(root = training_dir, model_name = model.model_name, samples_per_class= 2, informative_samp = informative_samp, need_val=1)
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                            shuffle=True, num_workers=12,
                                            pin_memory=True)
        loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size,
                                            shuffle=True, num_workers=12,
                                            pin_memory=True)
        loaders = [loader, loader_val]
        print('Size of the training dataset:', data.__len__(), '|  Size of the validation dataset: ', data_val.__len__() )

        losses_mean = [[],[]]
        losses_std = [[],[]]
    else:   
        data = TrainingDataset(root = training_dir, model_name = model.model_name, samples_per_class= 2, informative_samp = informative_samp, need_val=0)
        print('Size of dataset', data.__len__())
        loaders = [torch.utils.data.DataLoader(data, batch_size=batch_size,
                                            shuffle=True, num_workers=12,
                                            pin_memory=True)]
        losses_mean = [[]]
        losses_std = [[]]
    
    # Creation of the optimizer and the scheduler
    optimizer, loss_function = get_optim(model, data, loss_name, lr, decay, beta_lr, lr_proxies)
    starting_epoch = 0
    if sched == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif sched == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2, epochs],
                                                        gamma=gamma)
    
    range_plot = range(epochs)

    # Creation of the folder to save the weight
    weight_path = create_weights_folder(model.model_name, starting_weights)
    write_info(model, weight_path, lr, decay, beta_lr, lr_proxies, loss_name, epochs, informative_samp, need_val, sched, gamma)
    # Downloading of the pretrained weights and parameters 
    if starting_weights is not None:
        checkpoint = torch.load(starting_weights)
        if parallel:
            model.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        loss_function = checkpoint['loss_function']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        try:
            checkpoint_loss = torch.load(weight_path + '/loss')
            losses_mean = checkpoint_loss['loss_means']
            losses_std = checkpoint_loss['loss_stds']

        except:
            print("Issue with the loss file, it will be started from scratch")
        range_plot = range(starting_epoch, epochs)


    try:
        for epoch in range(starting_epoch, epochs):
            start_time = time.time()
            if need_val:
                loss_list_val = []
                loss_list = []
                loss_lists = [loss_list, loss_list_val]
            else:
                loss_list = []
                loss_lists = [loss_list]
            for j in range(len(loaders)):
                loader = loaders[j]
                for i, (labels, images) in enumerate(loader):
                    if i%1000 == 0 and j ==0:
                        print(i, flush=True)
                    
                    images_gpu = images.to(model.device)
                    labels_gpu = labels.to(model.device)
                    if model.model_name == "deit":
                        out = model.forward(images_gpu.view(-1, 3, 224, 224))
                    else:
                        out = model.forward(images_gpu)
                    if loss_function is None:
                        print("This model requires a specific loss. Please specifiy one. ")
                        exit(-1)
                    loss = loss_function(out, labels_gpu)

                    if j == 0: # training
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                    loss_lists[j].append(loss.item())
            
            if need_val:
                print("epoch {}, loss = {}, loss_val = {}, time {}".format(epoch, np.mean(loss_lists[0]),
                                                        np.mean(loss_lists[1]), time.time() - start_time))
                losses_mean[1].append(np.mean(loss_lists[1]))
                losses_std[1].append(np.std(loss_lists[1]))
            else:
                print("epoch {}, loss = {}, time {}".format(epoch, np.mean(loss_lists[0]),
                                                        time.time() - start_time))
            
            print("\n----------------------------------------------------------------\n")
            losses_mean[0].append(np.mean(loss_lists[0]))
            losses_std[0].append(np.std(loss_lists[0]))
            if sched != None:
                scheduler.step()

            # Saving of the model
            model_saving(model.model, epoch, epochs, epoch_freq, weight_path, optimizer, scheduler, loss, loss_function, loss_list, losses_mean, losses_std)

        if need_val:
            plt.figure()
            plt.errorbar(range_plot, losses_mean[1], yerr=losses_std[1], fmt='o--k',
                        ecolor='lightblue', elinewidth=3)
            plt.savefig(weight_path+"/validation_loss.png")
        plt.figure()
        plt.errorbar(range_plot, losses_mean[0], yerr=losses_std[0], fmt='o--k',
                        ecolor='lightblue', elinewidth=3)
        plt.savefig(weight_path+"/training_loss.png")
                
    
    except KeyboardInterrupt:
        print("Interrupted")
