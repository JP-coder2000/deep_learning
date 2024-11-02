# model.py
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, num_heads: int = 4, 
                 num_layers: int = 2, max_seq_length: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = self._create_sinusoidal_embeddings(max_seq_length, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        
        # Inicializar pesos
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _create_sinusoidal_embeddings(self, max_seq_length: int, dim: int) -> nn.Parameter:
        pe = torch.zeros(max_seq_length, dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.size()
        
        if t > self.max_seq_length:
            raise ValueError(f"Input sequence length ({t}) exceeds maximum allowed ({self.max_seq_length})")
        
        # Get embeddings
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding[:t, :]
        
        # Combine and process
        x = self.emb_dropout(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, 
                temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Genera texto usando el modelo."""
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= self.max_seq_length:
                input_ids = input_ids[:, -self.max_seq_length:]
            
            # Get predictions
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm
        x_ln = self.ln1(x)
        attn_output, _ = self.attention(x_ln, x_ln, x_ln)
        x = x + self.dropout(attn_output)
        
        x_ln = self.ln2(x)
        x = x + self.dropout(self.feed_forward(x_ln))
        return x

