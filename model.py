import torch
import torch.nn as nn
from helpers.args import ModelArgs


def kmeans_loss(x, centroids, assignments):
    distances = torch.norm(x - centroids[assignments], dim=1)
    return torch.sum(distances**2)


class BERTEmbeddings(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.position_embeddings = nn.Embedding(
            args.max_pos_embeddings, args.hidden_dim
        )
        self.LayerNorm = nn.LayerNorm(args.hidden_dim)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_ids):

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class KMeans(nn.Module):
    def __init__(self, K=10, Niter=10, input_dim=393216):  # Update input_dim
        super().__init__()
        self.K = K
        self.Niter = Niter
        self.centroids = nn.Parameter(
            torch.randn(self.K, input_dim)
        )  # Adjust input_dim

    def forward(self, x):
        N, D = x.shape

        # Detach centroids to prevent gradient flow
        with torch.no_grad():
            centroids = self.centroids.clone()

        x_i = x.view(N, 1, D)
        c_j = centroids.view(1, self.K, D)

        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cl = D_ij.argmin(dim=1).long().view(-1)

        return cl


# TODO: Analyse the benefit of whether self attention is even required
class BookRecommendationModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super(BookRecommendationModel, self).__init__()

        # Create the layers
        self.embeddings = BERTEmbeddings(args=args)
        self.attention = nn.MultiheadAttention(embed_dim=args.hidden_dim, num_heads=8)
        self._cluster = KMeans()

    def forward(self, x):

        # Use the embeddings
        embeddings = self.embeddings(input_ids=x)

        # Process input through attention mechanism
        attn_applied, _ = self.attention(embeddings, embeddings, embeddings)

        # Flatten attention output
        flattened = attn_applied.view(attn_applied.size(0), -1)

        # Generate the cluster labels
        cluster_labels = self._cluster(flattened)

        # Calculate k-means loss
        centroids = self._cluster.centroids
        loss = kmeans_loss(flattened, centroids, cluster_labels)

        return cluster_labels, loss
