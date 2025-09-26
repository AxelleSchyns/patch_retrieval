import numpy as np
from PIL import Image
import torch 
import os
import matplotlib.pyplot as plt

# given the filename, return the image object 
def load_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        image = image.resize((224,224))
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return image.reshape(-1)

# Define a function to generate batches of image paths
def batch_image_paths(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i+batch_size]

# Write the information about the training in a text file
def write_info(obj, path, lr, decay, beta_lr, lr_proxies, loss_name, epochs, informative_samp, need_val, sched, gamma ):
    with open(path + "/info.txt", "w") as file:
        file.write("Information about the training of the model\n")
        file.write("Model:" + str(obj.model_name) + "\n")
        file.write("Weight: " + str(obj.weight) + "\n")
        file.write("Loss function: " + str(loss_name) + "\n")
        file.write("Num features: " + str(obj.num_features) + "\n")

        file.write("\n")
        file.write("\n")

        file.write("Learning rate: " + str(lr) + "\n")
        file.write("Decay: " + str(decay) + "\n")
        file.write("Beta learning rate: " + str(beta_lr) + "\n")
        file.write("Learning rate proxies: " + str(lr_proxies) + "\n")
        file.write("Scheduler: " + str(sched) + "\n")
        file.write("Gamma: " + str(gamma) + "\n")
        
        file.write("Epochs: " + str(epochs) + "\n")
        file.write("Informative sampling: " + str(informative_samp) + "\n")
        file.write("Parallel: " + str(obj.parallel) + "\n")
        file.write("Batch size: " + str(obj.batch_size) + "\n")
        file.write("Need validation: " + str(need_val) + "\n")

# Create the folder to save the weights  
def create_weights_folder(model_name, starting_weights = None):
    try:
        os.mkdir("weights_folder")
    except FileExistsError:
        pass
    try:
        os.mkdir("weights_folder/"+model_name)
    except FileExistsError:
        pass
    if starting_weights != None:
        id_start = starting_weights.rfind("version")
        if id_start == -1:
            print("Issue with the format of the folder containing the weight, please check that it is in the form: weights_folder/model_name/version_x")
            exit(-1)
        id_end = starting_weights[id_start:].rfind("/") + id_start
        weight_path = starting_weights[0:id_end]
    else:
        versions = []
        for file in os.listdir("weights_folder/"+model_name):
            id_ = file.find("_")
            if id_ != 7:
                continue
            versions.append(int(file[8:]))
        versions.sort()
        
        count = 0
        for nb in versions:
            if nb != count:
                break
            count += 1
        weight_path = "weights_folder/"+model_name+"/version_"+str(count)
        try:
            os.mkdir(weight_path)
        except FileExistsError:
            print("Issue with the creation of the folder, risk of overwriting existing files.")
    
    return weight_path

# Save the model, the optimizer, the scheduler and the loss   
def model_saving(model, epoch, epochs,  epoch_freq, weight_path, optimizer, scheduler, loss, loss_function, loss_list, loss_mean, loss_stds):
    try:
        model = model.module
    except AttributeError:
        pass

    state = np.isnan(loss_list).any() # Indicate if the model has diverged

    if state == False:
        # Save separate file every epoch_freq epoch
        if epoch == 0 or (epoch+1) % epoch_freq == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'loss_function': loss_function,
            }, weight_path + "/epoch_"+str(epoch))
        
        # Save the last epoch (overwrite the file)
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'loss_function': loss_function,
        }, weight_path + "/last_epoch")

        # Save the loss (overwrite the file)
        torch.save({
            'loss_means': loss_mean,
            'loss_stds': loss_stds,
            }, weight_path + "/loss")
        
        # Save the model corresponding to the lowest validation loss
        if len(loss_mean) > 1:
            if len(loss_mean[1]) == 1 or loss_mean[1][-1] < min(loss_mean[1][:-1]):
                if len(loss_mean[1]) > 1:
                    print("Loss mean: ", loss_mean[1][-1], " min: ", min(loss_mean[1][:-1]))
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'loss_function': loss_function,
                }, weight_path + "/best_model")
    else:
        print("Loss is nan, not saving the model and stopping the training")
        plt.errorbar(range(epoch-1), loss_mean[:-1], yerr=loss_stds[:-1], fmt='o--k',
                ecolor='lightblue', elinewidth=3)
        plt.savefig(weight_path+"/training_loss.png")
        exit(-1)