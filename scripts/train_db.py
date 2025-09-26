import argparse
import torch

from models.model import Model
from database.db import Database

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        default='resnet'
    )

    parser.add_argument(
        '--weights'
    )

    parser.add_argument(
        '--db_name',
        default = 'db'
    )
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Retrieve the pretrained model 
    model = Model(args.model_name, args.weights, device)

    # Create the database
    database = Database(args.db_name, model, load=True)

    # Train the index
    database.train_index()
    database.save()
