"""Randomly initialized, token-preserving modality encoders. No model downloads."""
import math
import torch
from torch import nn


def sequence_positions(length, width, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000) / width))
    out = torch.zeros(length, width, device=device, dtype=dtype)
    out[:, 0::2] = torch.sin(position * frequency)
    out[:, 1::2] = torch.cos(position * frequency[:width // 2])
    return out


def reset_transformer(module, *, bert=False):
    # TransformerEncoder clones its prototype layer; initialize each clone
    # independently rather than starting every depth at identical weights.
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Embedding)):
            if bert or isinstance(child, nn.Embedding):
                nn.init.normal_(child.weight, std=.02)
            else:
                nn.init.xavier_uniform_(child.weight)
            if isinstance(child, nn.Linear) and child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.MultiheadAttention):
            if bert:
                nn.init.normal_(child.in_proj_weight, std=.02)
            else:
                nn.init.xavier_uniform_(child.in_proj_weight)
            if child.in_proj_bias is not None:
                nn.init.zeros_(child.in_proj_bias)


def check_token_budget(count, budget):
    if budget is not None and count > budget:
        raise ValueError(f"{count} tokens exceed remaining max_tokens budget={budget}; adjust explicit data sampling")


class ViTEncoder(nn.Module):
    """Patch projection + 2D positions + pre-LN ViT; returns every patch."""
    def __init__(self, width, heads, layers, mlp_ratio, patch_size, dropout):
        super().__init__()
        self.patch = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(width, heads, mlp_ratio * width, dropout,
                                          batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, layers, nn.LayerNorm(width), enable_nested_tensor=False)
        reset_transformer(self.transformer)

    def forward(self, image, *, token_budget=None):
        if image.ndim != 3 or image.shape[0] != 3 or image.numel() == 0 or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("image must be finite RGB CHW float in [0,1]")
        height, width = image.shape[1:]
        if min(height, width) < self.patch_size or height % self.patch_size or width % self.patch_size:
            raise ValueError("Image dimensions must be positive and divisible by patch_size")
        height, width = height // self.patch_size, width // self.patch_size
        check_token_budget(height * width, token_budget)
        content = self.patch(image[None]).flatten(2).transpose(1, 2)
        dim = content.shape[-1]
        # Separate row/column encodings preserve 2D location, including on
        # rectangular grids, before any patch-to-patch attention takes place.
        y_pos = sequence_positions(height, dim // 2, content.device, content.dtype)
        x_pos = sequence_positions(width, dim - dim // 2, content.device, content.dtype)
        positions = torch.cat((y_pos[:, None].expand(-1, width, -1),
                               x_pos[None].expand(height, -1, -1)), dim=-1).reshape(height * width, dim)
        tokens = self.transformer(self.dropout(content + positions[None]))[0]
        y, x = torch.meshgrid(torch.arange(height, device=content.device, dtype=content.dtype),
                              torch.arange(width, device=content.device, dtype=content.dtype), indexing="ij")
        xy = torch.stack(((x.flatten() + .5) / width, (y.flatten() + .5) / height), dim=-1)
        return tokens, xy


class BertTextEncoder(nn.Module):
    """Byte embeddings + learned positions + bidirectional post-LN/GELU blocks.

    This is BERT-style architecture trained with the policy, without pretrained
    weights, WordPiece vocabulary, CLS/pooling, MLM or NSP heads.
    """
    def __init__(self, width, heads, layers, mlp_ratio, max_tokens, dropout):
        super().__init__()
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(256, width)
        self.position = nn.Embedding(max_tokens, width)
        self.embedding_norm = nn.LayerNorm(width, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(width, heads, mlp_ratio * width, dropout,
                                          batch_first=True, norm_first=False, activation="gelu", layer_norm_eps=1e-12)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        reset_transformer(self, bert=True)

    def forward(self, text, *, token_budget=None):
        if not isinstance(text, str) or not text:
            raise ValueError("text must be nonempty UTF-8 text")
        data = text.encode("utf-8")
        if len(data) > self.max_tokens:
            raise ValueError(f"Text has {len(data)} UTF-8 bytes, exceeding max_text_tokens={self.max_tokens}; shorten explicitly or increase the limit")
        check_token_budget(len(data), token_budget)
        ids = torch.tensor(list(data), device=self.embedding.weight.device, dtype=torch.long)
        positions = torch.arange(len(ids), device=ids.device)
        content = self.dropout(self.embedding_norm(self.embedding(ids) + self.position(positions)))
        return self.transformer(content[None])[0]
