import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from helpers.args import ModelArgs
from model import BookRecommendationModel
from dataset import BookDataset
from train import BookRecommendationTrainer, TrainArgs

ARG_PATH = os.path.abspath("arguments")
ENV_ARG_NAME = "ARG"
BOOKS_DATASET_ARG = "DATASET"


def load_model_arguments(args_name: str):
    arg_file = os.path.join(ARG_PATH, args_name)
    arg_data = {}

    with open(arg_file, "r") as f:
        arg_data = json.loads(f.read())

    print(f"Loaded model args: {args_name}")

    for key, value in arg_data.items():
        print(f"{key}: {value}")

    print("")
    return arg_data


def load_arguments():
    args_name = os.getenv(ENV_ARG_NAME, "default.json")
    books_dataset_name = os.getenv(BOOKS_DATASET_ARG, "book_training_dataset.csv")

    # Load the argument models
    arg_data = load_model_arguments(args_name=args_name)
    model_args = ModelArgs(**arg_data)

    return model_args, books_dataset_name


def create_model(model_args: ModelArgs):
    return BookRecommendationModel(model_args)


def create_book_dataset(books_dataset_name: str, device: torch.device):
    book_dataset = BookDataset(books_dataset_name, device)

    book_dataloader = DataLoader(book_dataset, batch_size=16, shuffle=True)

    return book_dataloader


def create_hypers():
    device = torch.device("mps")
    criterion = nn.MSELoss()

    print(f"Using Device: {device}\n")

    return device, criterion


def create_args():
    train_args = TrainArgs(6)
    return train_args


def within_cluster_variance_loss(embeddings, cluster_labels):
    n_clusters = len(torch.unique(cluster_labels))
    cluster_variances = []

    for cluster_id in range(n_clusters):
        cluster_data = embeddings[cluster_labels == cluster_id]
        cluster_mean = cluster_data.mean(dim=0)
        variance = ((cluster_data - cluster_mean) ** 2).mean()
        cluster_variances.append(variance)

    return torch.mean(torch.tensor(cluster_variances))


def train():
    device, criterion = create_hypers()
    model_args, books_dataset_name = load_arguments()

    book_dataloader = create_book_dataset(
        books_dataset_name=books_dataset_name, device=device
    )
    model = create_model(model_args=model_args)

    optimiser = torch.optim.Adam(model.parameters())
    train_args = create_args()

    trainer = BookRecommendationTrainer(
        model=model,
        optimizer=optimiser,
        criterion=criterion,
        dataloader=book_dataloader,
        device=device,
        args=train_args,
    )

    trainer.train()

    # Save the model
    torch.save(model.state_dict(), "cluster_model.pth")


def main():
    train()


if __name__ == "__main__":
    main()
