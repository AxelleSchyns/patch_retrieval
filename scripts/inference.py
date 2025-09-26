import argparse
import torch

from models.model import Model
from database.db import Database
from experiment.quantitative import full_test, test_each_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', default='resnet', type=str, help='Model architecture')
    parser.add_argument(
        '--weights', default=None, type=str, help='Path to the model weights')
    parser.add_argument(
        '--db_name', default='db', type=str, help='Name of the database')
    parser.add_argument(
        '--indexing', action='store_true', help='If set, the database will be indexed with the dataset at path_index')
    parser.add_argument(
        '--path_index', default=None, type=str, help='Path to the dataset to index')
    parser.add_argument(
        '--data_name', default='uliege', type=str, help='Name of the dataset among: uliege, cam, crc')
    parser.add_argument(
        '--query_path', default=None, type=str, help='Path to the query image or folder of images')
    parser.add_argument(
        '--measure', default='all', type=str, help='Measure to use among: all, remove (uliege only), weighted, random')
    parser.add_argument(
        '--filename', default=None, type=str, help='Path to the output file')
    parser.add_argument(
        '--stat', action='store_true', help='If set, statistics will be computed on the results')
    parser.add_argument(
        '--class_split', action='store_true', help='If set, the results will be computed for each class separately (uliege only)')
    parser.add_argument(
        '--project_name', default=None, type=str, help='Name of the project to use for inference')
    parser.add_argument(
        '--class_name', default=None, type=str, help='Name of the class to use for inference')

    args = parser.parse_args()

    # Create the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(model_name = args.model_name, weight = args.weights, device = device)

    # Create the database
    database = Database(filename=args.db_name, model=model, load=not args.indexing, device=device)
    if args.indexing:
        if args.path_index is None:
            print("Please provide a path to the dataset to index with --path_index")
            exit(-1)
        database.add_dataset(args.path_index)
        database.save()
    if args.class_split:
        test_each_class(model, args.db_name, args.data_name, args.query_path, args.measure, "./logs/results_per_class.xlsx")
    else:
        full_test(model, args.db_name, args.data_name, args.query_path, args.measure, log_filename=args.filename, stat = args.stat, project_name=args.project_name, class_name=args.class_name)

        
        """display_cm(model.weight, args.measure, ground_truth, data, predictions)
        display_precision_recall(model.weight, args.measure, ground_truth, predictions[0])
        display_prec_im(model.weight, props, data, args.measure)"""