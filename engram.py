import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EngramEmbeddings(nn.Module):
    def __init__(
        self, 
        table_size: int, 
        embed_dim: int,
        compression_map: Tensor = None,
        num_heads: int = 8,
        ngram_order: int = 2,
        dtype = None
    ):
        """
        Args:
            table_size: A large prime number representing the size of the lookup table.
            embed_dim: The dimension of the retrieved memory vector.
            num_heads: Number of independent hash functions (K).
            ngram_order: The number of tokens to look back for the N-gram (N).
        """
        super().__init__()
        table_size = table_size * num_heads
        
        self.table_size = table_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ngram_order = ngram_order
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.memory_table = nn.Embedding(table_size, self.head_dim, dtype=dtype)
        self.memory_table.weight.data.zero_()

        primes = torch.randint(1000000, 1000000000, (num_heads, ngram_order))
        self.register_buffer("hash_primes", primes)
        self.register_buffer("compression_map", compression_map)

    def get_ngram_hashes(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extracts N-grams and hashes them deterministically.
        Returns tensor of shape (batch_size, seq_len, num_heads)
        """
        # Pad the beginning of the sequence to maintain length during N-gram extraction
        padded_ids = F.pad(input_ids, (self.ngram_order - 1, 0), value=0)
        
        # Create rolling N-grams: (batch, seq_len, ngram_order)
        ngrams = padded_ids.unfold(dimension=1, size=self.ngram_order, step=1) 
        
        # Broadcast N-grams and primes for batched hash computation
        # ngrams: (batch, seq, 1, ngram_order)
        # primes: (1, 1, heads, ngram_order)
        ngrams = ngrams.unsqueeze(2)
        primes = self.hash_primes.unsqueeze(0).unsqueeze(0)
        
        # Multi-head hash function: sum(token_i * prime_i) % table_size
        hashes = torch.sum(ngrams * primes, dim=-1) % self.table_size
        return hashes

    def get_chaotic_ngram_hashes(self, input_ids: torch.Tensor) -> torch.Tensor: # has the same ammount of collisions as the simpler function on average
        """
        Extracts N-grams and applies a chaotic Murmur-style bitwise hash 
        to maximize the avalanche effect and minimize collisions.
        """
        # 1. Pad and extract rolling N-grams: (batch, seq_len, ngram_order)
        padded_ids = F.pad(input_ids, (self.ngram_order - 1, 0), value=0)
        ngrams = padded_ids.unfold(dimension=1, size=self.ngram_order, step=1).to(torch.int64)
        
        # Broadcast N-grams and primes: (batch, seq, heads, ngram_order)
        ngrams = ngrams.unsqueeze(2)
        primes = self.hash_primes.unsqueeze(0).unsqueeze(0).to(torch.int64)
        

        mixed_ngrams = ngrams * primes
        
        hash_val = mixed_ngrams[..., 0]
        for i in range(1, self.ngram_order):
            # Bitwise XOR across the N-gram dimension
            hash_val = hash_val ^ mixed_ngrams[..., i]
            hash_val = hash_val * 0x84fbca63
            
        hash_val = hash_val ^ (hash_val >> 16)
        hash_val = hash_val * 0x85ebca6b
        hash_val = hash_val ^ (hash_val >> 13)
        hash_val = hash_val * 0xc2b2ae35
        hash_val = hash_val ^ (hash_val >> 15)
        
        # PyTorch int64 is signed, so bitwise ops might result in negative numbers.
        # We must take the absolute value before modulo to prevent negative embedding indices.
        hashes = torch.abs(hash_val) % self.table_size
        
        return hashes

    def forward(self, input_ids: Tensor, with_hashes = False) -> Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) - Compressed token IDs.
            hidden_states: (batch_size, seq_len, hidden_dim) - Current backbone hidden states.
        Returns:
            gated_memory: (batch_size, seq_len, embed_dim) - The memory ready to be added to the residual stream.
        """
        batch_size, seq_len = input_ids.shape
        
        if self.compression_map is not None:
            input_ids = self.compression_map[input_ids]
        hashes = self.get_ngram_hashes(input_ids) 
        retrieved_embeds = self.memory_table(hashes) 

        memory_output = retrieved_embeds.view(batch_size, seq_len, self.embed_dim)
        if with_hashes:
            return memory_output, hashes
        return memory_output # should be on cpu, can be transfered asynchronously to gpu later
