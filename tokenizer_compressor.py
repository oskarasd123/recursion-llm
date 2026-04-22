import torch
from transformers import PreTrainedTokenizer

def create_token_compression_map(tokenizer: PreTrainedTokenizer, as_tensor: bool = False):
    """
    Creates a mapping from all tokenizer IDs to a canonical token ID,
    collapsing variations in casing and leading space markers.
    
    Args:
        tokenizer: A Hugging Face tokenizer instance.
        as_tensor: If True, returns a 1D PyTorch tensor for O(1) array indexing.
                   If False, returns a Python dictionary.
                   
    Returns:
        dict[int, int] or torch.Tensor
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # Dictionary to hold lists of IDs that share the same normalized string
    normalized_to_ids = {}
    
    for token_str, token_id in vocab.items():
        # 1. Normalize the token string
        # Replace common BPE 'Ġ' and SentencePiece ' ' space markers
        clean_str = token_str.replace('Ġ', '').replace(' ', '').lower()
        
        # 2. Group IDs by their normalized string
        if clean_str not in normalized_to_ids:
            normalized_to_ids[clean_str] = []
        normalized_to_ids[clean_str].append(token_id)
        
    # 3. Create the id_map (Original ID -> Canonical ID)
    id_map = {}
    for clean_str, ids in normalized_to_ids.items():
        # We use the lowest ID as the canonical one. 
        # In most tokenizers, lower IDs represent more frequent, base subwords.
        canonical_id = min(ids)
        for token_id in ids:
            id_map[token_id] = canonical_id
            
    # 4. Format the output
    if as_tensor:
        # Create a lookup tensor where index = original_id, value = canonical_id
        # We initialize with sequential IDs as a fallback for any unmapped tokens
        lookup_tensor = torch.arange(vocab_size, dtype=torch.long)
        for orig_id, canon_id in id_map.items():
            lookup_tensor[orig_id] = canon_id
        return lookup_tensor
        
    return id_map

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import time
    import numpy as np
    # Load a common tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    compression_map = create_token_compression_map(tokenizer, True)
    uncompressed = torch.arange(compression_map.size(0), dtype=torch.long)
    changed = compression_map != uncompressed
    compressed = uncompressed[changed].unique()
    for i in range(200):
        print(f"|{tokenizer.decode(compressed[i])}|{tokenizer.decode(compression_map[compressed[i]])}|")