import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class TrainArgs:
    n_epochs: int


class BookRecommendationTrainer:
    def __init__(self, model, optimizer, criterion, dataloader, device, args):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.args = args

    def _forward_pass(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def _training_step(self, inputs):
        cluster_labels, loss = self._forward_pass(inputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train(self):
        print(self.device)
        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(self.args.n_epochs):
            running_loss = 0.0

            with tqdm(self.dataloader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    inputs = data
                    inputs.to(self.device)

                    loss = self._training_step(inputs)
                    running_loss += loss

                    tepoch.set_description(
                        f"Epoch {epoch+1}, Loss: {running_loss / (i+1):.4f}"
                    )

            print(f"Epoch {epoch+1} Loss: {running_loss / len(self.dataloader):.4f}")
