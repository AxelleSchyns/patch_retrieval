import datetime
import random
import time
import models
import os

import matplotlib.pyplot as plt
import pandas as pd

from database.db import Database
from argparse import ArgumentParser
from PIL import Image
from torchvision import transforms
from utils.uliege_utils import * 
from models.model import Model
from utils.inference_utils import *

def list_of_strings(string):
    return string.split(',')

def list_of_ints(string):
    return [int(i) for i in string.split(',')]

def random_query_list(path):
    query_list = []
    for c in os.listdir(path):
        if os.path.isdir(os.path.join(path, c)):
            img = random.choice(os.listdir(os.path.join(path,c)))
            img = os.path.join(path,c,img)
            query_list.append(img)
    return query_list

def log_result(log_path, model_name, query_path, top1_path, results_dir):
    """Append results to Excel log file"""
    new_entry = pd.DataFrame([{
        "model": model_name,
        "query": query_path,
        "top1": top1_path,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results_dir": results_dir

    }])

    if os.path.exists(log_path):
        df = pd.read_excel(log_path)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_excel(log_path, index=False)

def multi_query_multi_models(log_path, models_name_list, queries_path, results_dir, weights_dir, path_index, device):

    fig_list, axes_list = [], []

    for query_path in queries_path:
        fig, axes = plt.subplots(3, 6, figsize=(7, 4))
        axes = axes.flatten()
        axes[0].imshow(Image.open(query_path).convert('RGB'))
        axes[0].set_title("Query image", fontsize=8)
        axes[0].axis('off')
        fig_list.append(fig)
        axes_list.append(axes)

    for i, model_name in enumerate(models_name_list):
        # Load model
        weights = get_weight(model_name, weights_dir)
        model = Model(model_name, weights, device)

        # Indexing
        database = Database("db", model)
        t = time.time()
        database.add_dataset(path_index)
        print("T_indexing = "+str(time.time() - t))
        for j, query_path in enumerate(queries_path):
            class_name = get_class(query_path)
            # Retrieve the most similar images
            top1_path, _, _, _, _ = database.search(Image.open(query_path).convert('RGB'), 1)

            # Add to subplot figure
            axes_list[j][i+1].imshow(Image.open(top1_path).convert('RGB'))
            axes_list[j][i+1].set_title(f"{class_name}\n{model_name}", fontsize=6)
            axes_list[j][i+1].axis('off')

            # Log to Excel
            log_result(log_path, model_name, query_path, top1_path, results_dir)

    for i, fig in enumerate(fig_list):
        fig.tight_layout()
        save_path = os.path.join(results_dir, f"all_top1_results_{get_class(queries_path[i])}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

# Retrieve the top1 result to a query image for all models in the given list
def one_query_all_models(log_path, models_name_list, query_path, results_dir, weights_dir, path_index, device):

    # Figure 1: query image alone
    plt.figure()
    plt.imshow(Image.open(query_path).convert('RGB'))
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, "query.png"))

    # Figure 2: query image + nearest image for each model
    fig, axes = plt.subplots(3, 6, figsize=(7, 4))
    axes = axes.flatten()

    # put query in first subplot
    axes[0].imshow(Image.open(query_path).convert('RGB'))
    axes[0].set_title("Query image", fontsize=8)
    axes[0].axis('off')

    for i, model_name in enumerate(models_name_list):
        # Load model
        weights = get_weight(model_name, weights_dir)
        model = Model(model_name, weights, device)

        # Indexing
        database = Database("db", model)
        t = time.time()
        database.add_dataset(path_index)
        print("T_indexing = "+str(time.time() - t))

        # Retrieve the most similar images
        top1_path, _, _, _, _ = database.search(Image.open(query_path).convert('RGB'), 1)

        # Figure 3- nb_models + 2: save top1 image for each model
        plt.figure()
        plt.imshow(Image.open(top1_path).convert('RGB'))
        plt.axis('off')
        plt.savefig(os.path.join(results_dir, f"Top1_{class_name}_{model_name}_{i}.png"))

        # Add to subplot figure
        axes[i].imshow(Image.open(top1_path).convert('RGB'))
        axes[i].set_title(f"{class_name}\n{model_name}", fontsize=6)
        axes[i].axis('off')

        # Log to Excel
        log_result(log_path, model_name, query_path, top1_path, results_dir)

    # Save subplot with all results
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_top1_results.png"))
    plt.close(fig)

# Retrieve the top10 results to a query image for one single model
def one_query_one_model(log_path, model_name, model_weight, device, query_path, results_dir, path_index = None):
    # Load model
    model = Model(model_name, model_weight, device)

    # Indexing
    if path_index is not None:
        database = Database("db", model)
        t = time.time()
        database.add_dataset(path_index)
        print("T_indexing = "+str(time.time() - t))
    else:
        database = Database("db", model, load=True)

    # Retrieve the most similar images
    names, _, _, _, _ = database.search(Image.open(query_path).convert('RGB'), 10)

    # --- Save query image ---
    query_img = Image.open(query_path).convert('RGB')
    query_img.save(os.path.join(results_dir, "query.png"))

    # --- Subplot: query + top-k retrieved images ---
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    axes = axes.flatten()

    # Query in first subplot
    axes[0].imshow(query_img)
    axes[0].set_title("Query image", fontsize=8)
    axes[0].axis("off")

    # Retrieved images
    for i, n in enumerate(names, start=1):
        axes[i].imshow(Image.open(n).convert("RGB"))
        axes[i].set_title(get_class(n), fontsize=6)
        axes[i].axis("off")

    # Hide unused axes
    for ax in axes[len(names)+1:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"top10_{model_name}.png"))
    plt.close(fig)

    # --- Log results to Excel ---
    log_result(log_path, model_name, query_path, names, results_dir)

    print(f"Top-10 retrieval for {model_name} saved and logged.")

          


            
