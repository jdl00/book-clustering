from dataclasses import dataclass


@dataclass
class ModelArgs:
    vocab_size: int
    input_dim: int
    hidden_dim: int
    pad_token_id: int
    max_pos_embeddings: int
    layer_norm_eps: int
    type_vocab_size: int
    hidden_dropout_prob: float
    n_clusters: int
