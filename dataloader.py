import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader






class FineWebDataLoader(IterableDataset):
    def __init__(self, tokenizer, subset="sample-10BT", edu=False, max_length=8192, num_val_documents = 1000):
        """
        Args:
            tokenizer: A tokenizer instance (e.g., from Hugging Face or tiktoken).
            subset: The FineWeb subset name (e.g., 'sample-10BT', 'sample-100BT', 'default').
            split: Dataset split to load.
            max_length: Maximum sequence length for tokenization.
        """
        self.subset = subset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb" + ("-edu" if edu else ""), 
            name=subset, 
            split="train", 
            streaming=(subset != "sample-10BT")
        )
        self.num_val_documents = num_val_documents
        self.data_iter = iter(self.dataset)
        self.val_data = ValDataset(self.dataset.take(num_val_documents), tokenizer, max_length, subset)


    def __iter__(self):
        protocol = self.dataset.skip(self.num_val_documents)
        if self.subset != "sample-10BT":
            protocol = protocol.shuffle(buffer_size=50_000, seed=55)
        else:
            protocol = protocol.shuffle(seed=55)
        
        token_buffer = []
        lengths = []
        texts = []
        for example in protocol:
            text = example["text"]
            # Tokenize the text
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            if sum(lengths) + len(tokens) > self.max_length:
                dict = self._prepare_batch(token_buffer, lengths)
                dict["texts"] = texts
                yield dict
                token_buffer = []
                lengths = []
                texts = []

            token_buffer.extend(tokens)
            lengths.append(len(tokens))
            texts.append(text)
    
    def _prepare_batch(self, tokens, lengths):
        # Create cu_seqlens: [0, len1, len1+len2, ...]
        cu_seqlens = torch.tensor([0] + torch.cumsum(torch.tensor(lengths), dim=0).tolist(), dtype=torch.int32)
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max(lengths)
        }


class MaxLenFineWebDataLoader(FineWebDataLoader):
    def __iter__(self):
        protocol = self.dataset.skip(self.num_val_documents)
        if self.subset != "sample-10BT":
            protocol = protocol.shuffle(buffer_size=50_000, seed=55)
        else:
            protocol = protocol.shuffle(seed=55)
        
        token_buffer = []
        lengths = []
        texts = []
        for example in protocol:
            text = example["text"]
            # Tokenize the text
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            if sum(lengths) + len(tokens) > self.max_length:
                split_pos = self.max_length-sum(lengths)
                current = tokens[:split_pos]
                next = tokens[split_pos:]
                token_buffer.extend(current)
                lengths.append(len(current))
                dict = self._prepare_batch(token_buffer, lengths)
                dict["texts"] = texts
                yield dict
                token_buffer = []
                lengths = []
                texts = []
                tokens = next
                
            token_buffer.extend(tokens)
            lengths.append(len(tokens))
            texts.append(text)
    


class ValDataset(FineWebDataLoader):
    def __init__(self, dataset, tokenizer, max_length=8192, subset="sample-10BT"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_val_documents = 0
        self.subset = subset
    
    def __iter__(self):
        return FineWebDataLoader.__iter__(self)

# Example Usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    import time
    import numpy as np
    # Load a common tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token # GPT2 doesn't have a pad token by default
    max_len = 8192//2
    # Initialize the dataloader
    dataloader = MaxLenFineWebDataLoader(tokenizer, subset="sample-10BT", edu=True, max_length=max_len)
    dataloader = DataLoader(dataloader, 1, num_workers=1)

    iterator = iter(dataloader)
    filled = []
    for i in range(256):
        batch = next(iterator)
        length = batch["input_ids"].shape[1]
        fill_frac = length/max_len
        filled.append(fill_frac)
    print(f"fill frac: {np.mean(filled)}\n"\
          f"min filled: {np.min(filled)}"
          )
    