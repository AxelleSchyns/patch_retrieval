from collections import Counter
import torch
import os
import time

import numpy as np
import pandas as pd

from dataset.dataset import TestDataset
from database.db import Database
from utils.inference_utils import *
from utils.uliege_utils import *

# -------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------
# Define project equivalences
EQUIVALENT_PROJECTS = [
    {"cells_no_aug", "patterns_no_aug"},
    {"mitos2014", "tupac_mitosis"},
    {"janowczyk1", "janowczyk2", "janowczyk5", "janowczyk6", "janowczyk7"}
]

def are_equivalent(proj_im, proj_retr):
    for group in EQUIVALENT_PROJECTS:
        if proj_im in group and proj_retr in group:
            return True
    return False

def compute_uliege(names, query, accuracies, predictions, data, weights):
    similar = names[:5]
    temp = []

    class_im = get_class(query)
    proj_im = get_proj(query)
    idx_class = 0 if len(data.classes) == 1 else data.conversion[class_im]

    # Proportion of correct image at each step
    prop = [1 if get_class(n) == class_im else 0 for n in names]

    # Evolution of proportion (cumulative average)
    ev_prop = [prop[0]]
    for i in range(1, len(names)):
        ev_prop.append((prop[i] + ev_prop[i-1] * i) / (i + 1))

    # Counters for majority vote
    counts = np.zeros(3)  # class, proj, sim

    # Loop over top-5 results
    for j, retr in enumerate(similar):
        class_retr = get_class(retr)
        proj_retr = get_proj(retr)
        temp.append(class_retr)

        # Add top-1 prediction for confusion matrix
        if j == 0:
            unknown_idx = len(data.conversion)
            predictions[0].append(data.conversion.get(class_retr, unknown_idx))

        # Define conditions dynamically
        conditions = [
            (class_retr == class_im, 0),                # same class
            (proj_retr == proj_im, 1),                  # same project
            (are_equivalent(proj_im, proj_retr) or proj_retr == proj_im, 2)     # equivalent project
        ]

        """for cond, k in conditions:
            if cond:
                if j == 0:
                    accuracies[0, k] += weights[idx_class]
                if counts[k] == 0:
                    accuracies[1, k] += weights[idx_class]
                counts[k] += 1"""
        for cond, k in conditions:
            if cond:
                if j == 0:
                    accuracies[0, k] += weights[idx_class]
                if counts[k] == 0:
                    accuracies[1, k] += weights[idx_class]

                counts[k] += 1

    # Majority vote accuracy
    for k, c in enumerate(counts):
        if c > 2:
            accuracies[2, k] += weights[idx_class]
    predictions[1].append(data.conversion[max(set(temp), key = temp.count)])
    return predictions, accuracies, prop

def compute_results(names, query, accuracies, predictions, data, weights):
    similar = names[:5]
    temp = []

    class_im = get_class(query)
    idx_class = 0 if len(data.classes) == 1 else data.conversion[class_im]

    # Counters for majority vote
    count = 0

    # Loop over top-5 results
    for j, retr in enumerate(similar):
        class_retr = get_class(retr)
        temp.append(class_retr)

        # Add top-1 prediction for confusion matrix
        if j == 0:
            unknown_idx = len(data.conversion)
            predictions[0].append(data.conversion.get(class_retr, unknown_idx))
            if class_retr == class_im:
                count += 1
                accuracies[0] += weights[idx_class]
        
        elif class_retr == class_im:
            accuracies[1] += weights[idx_class]
            count += 1
    if count > 2:
        accuracies[2] += weights[idx_class]
    predictions[1].append(data.conversion[max(set(temp), key = temp.count)])
    return predictions, accuracies

# Indexes the data in the database and computes the results
def inference(model, db_name, data_name, data_path, measure, project_name = None, class_name = None):

    # Load database
    database = Database(db_name, model, True)

    # Load data 
    data = TestDataset(data_path, measure, project_name, class_name)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,
                                         num_workers=4, pin_memory=True)
    
    # Load weights 
    if measure == 'weighted':
        weights = data.weights
    elif measure == 'remove':
        weights = np.ones(len(data.conversion))
    else:
        weights = np.ones(len(data.classes))
    
    # Compute the results at the class, the project and the origin level
    if data_name == 'uliege':
        accuracies = np.zeros((3,3))  
    else:
        accuracies = np.zeros((3,1))
    props = np.zeros((10,1)) # Proportion of correct images in the first n retrieved images

    nbr_per_class = Counter()


    ground_truth = []
    predictions = [[], []]
    
    t_search = 0
    t_model = 0
    t_transfer = 0
    t_tot = 0

    # For each image in the dataset, search for the 10 most similar images in the database and compute the accuracy
    for i, (image_tensor, image_name) in enumerate(loader):
        # Search for the 10 most similar images in the database
        t = time.time()
        names, _, t_model_tmp, t_search_tmp, t_transfer_tmp = database.search(image_tensor, 10)

        t_tot += time.time() - t
        t_model += t_model_tmp
        t_transfer += t_transfer_tmp
        t_search += t_search_tmp

        nbr_per_class[get_class(image_name[0])] += 1
        ground_truth.append(data.conversion[get_class(image_name[0])])


        # Compute accuracy 
        if data_name == 'uliege':
            predictions, accuracies, prop = compute_uliege(names, image_name[0], accuracies, predictions, data, weights)
        else:
            predictions, accuracies, prop = compute_results(names, image_name[0], accuracies, predictions, data, weights)
        for el in range(len(prop)):
            props[el] += prop[el]
    
    if measure == 'weighted':
        s = len(data.classes) # In weighted, each result was already divided by the length of the class
    else:
        s = data.__len__()

    accuracies = accuracies / s
    props = props / s

    return [accuracies, t_tot, t_model, t_search, t_transfer]
    
# -------------------------------------------------------------------------------------------
# Main functions
# -------------------------------------------------------------------------------------------
def test_each_class(model, db_name, data_name, data_path,  measure, excel_path):
    classes = sorted(os.listdir(data_path))
    if data_name == 'uliege':
        res = np.zeros((len(classes), 13))
    else:
        res = np.zeros((len(classes), 7))

    # Compute the results for each class
    for i, c in enumerate(classes):
        r = inference(model, db_name, data_name, data_path, measure, project_name = False, class_name = c)
        k = 0
        for j, el in enumerate(r):
            if j == 0:
                accuracies = el 
                if data_name == 'uliege':
                    for l in range(3):
                        for m in range(3):
                            res[i][j+k] = accuracies[m][l]
                            k += 1
                else:
                    for l in range(3):
                        res[i][j+k] = accuracies[l]
                        k += 1
            else:
                res[i][j+k-1] = el
 
    # Write the results in an Excel file
    write_class_results_xlsx(excel_path, data_name, data_path, model, res, classes)

def full_test(model, db_name, data_name, data_path, measure, log_filename, stat = False, nb_exp = 5, project_name = None, class_name = None):
    if stat:
        if data_name == 'uliege':
            accuracies = np.zeros((3,3, nb_exp))
        else:
            accuracies = np.zeros((3,1, nb_exp))
        ts = np.zeros((4,nb_exp))
        for i in range(nb_exp):
            accuracies[:, :, i], ts[0][i], ts[1][i], ts[2][i], ts[3][i] =  inference(model, db_name, data_name, data_path, measure, project_name, class_name)
        write_full_results_txt(log_filename, data_name, data_path , model, measure, accuracies, ts[0], ts[1], ts[2], ts[3], project_name, class_name, nb_exp)
    else:
        accuracies, t_tot, t_model, t_search, t_transfer = inference(model, db_name, data_name, data_path, measure, project_name, class_name)
        write_full_results_txt(log_filename, data_name, data_path , model, measure, accuracies, t_tot, t_model, t_search, t_transfer, project_name, class_name)
        write_full_results_xlsx("./logs/result.xlsx", data_name, data_path, model, measure, accuracies, t_tot, t_model, t_search, t_transfer, project_name, class_name)