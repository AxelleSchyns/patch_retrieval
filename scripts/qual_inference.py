import torch
import argparse

from experiment.qualitative import one_query_all_models, one_query_one_model, multi_query_multi_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        type=str,
        nargs='+',
        default='resnet'
    )

    parser.add_argument(
        '--weights'
    )

    parser.add_argument(
        '--db_name',
        default = 'db'
    )
    parser.add_argument(
        '--query_path',
        nargs='+',
        type=str,
    )
    parser.add_argument(
        '--index_path',
        default='index',
        type=str,
        help='Path to the index file'
    )
    parser.add_argument(
        '--excel_path',
        default='log.xlsx',
        type=str,
        help='Path to the Excel log file'
    )
    parser.add_argument(
        '--results_dir',
        default='results',
        type=str,
        help='Path to the results directory'
    )

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    if len(args.query_path) < 2:
        if len(args.model_name) < 2:
            one_query_one_model(args.excel_path, args.model_name[0], args.weight, device, args.query_path[0], args.results_dir, args.index_path)
        else:
            one_query_all_models(args.excel_path, args.model_name, args.query_path[0], args.results_dir, args.weights, args.index_path, device)
    else:
        multi_query_multi_models(args.excel_path, args.model_name, args.query_path, args.results_dir, args.weights, args.index_path, device)
