import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader






class FineWebDataLoader(IterableDataset):
    def __init__(self, tokenizer, subset="sample-10BT", edu=False, max_length=8192, num_val_documents = 1000, seed=55):
        """
        Args:
            tokenizer: A tokenizer instance (e.g., from Hugging Face or tiktoken).
            subset: The FineWeb subset name (e.g., 'sample-10BT', 'sample-100BT', 'default').
            max_length: Maximum sequence length for tokenization.
        """
        self.subset = subset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb" + ("-edu" if edu else ""), 
            name=subset, 
            split="train",
        )
        self.num_val_documents = num_val_documents
        self.data_iter = iter(self.dataset)
        self.val_data = ValDataset(self.dataset.take(num_val_documents), tokenizer, max_length, subset, seed)
        self.val_data.__iter__ = type(self).__iter__


    def __iter__(self):
        protocol = self.dataset.skip(self.num_val_documents)
        if self.subset != "sample-10BT":
            protocol = protocol.shuffle(buffer_size=50_000, seed=self.seed)
        else:
            protocol = protocol.shuffle(seed=self.seed)
        
        token_buffer = []
        lengths = []
        texts = []
        for example in protocol:
            text = example["text"]
            # Tokenize the text
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length-1,
            )["input_ids"] + [self.tokenizer.eos_token_id]
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
            protocol = protocol.shuffle(buffer_size=50_000, seed=self.seed)
        else:
            protocol = protocol.shuffle(seed=self.seed)
        
        token_buffer = []
        lengths = []
        texts = []
        for example in protocol:
            text = example["text"]
            # Tokenize the text
            tokens = self.tokenizer(
                text,
                truncation=True
            )["input_ids"] + [self.tokenizer.eos_token_id]
            while sum(lengths) + len(tokens) > self.max_length:
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
        dict = self._prepare_batch(token_buffer, lengths)
        dict["texts"] = texts
        yield dict
    


class ValDataset(FineWebDataLoader):
    def __init__(self, dataset, tokenizer, max_length=8192, subset="sample-10BT", seed=55):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_val_documents = 0
        self.subset = subset
        self.seed=seed
    
    def __iter__(self): # overriden by FineWebDataLoader
        return FineWebDataLoader.__iter__(self)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import time
    import numpy as np
    # Load a common tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token # GPT2 doesn't have a pad token by default
    max_len = 4096
    # Initialize the dataloader
    dataloader = FineWebDataLoader(tokenizer, subset="sample-10BT", edu=True, max_length=max_len)
    dataloader = DataLoader(dataloader, 1, num_workers=1)

    iterator = iter(dataloader)
    filled = []
    for i in range(512):
        batch = next(iterator)
        length = batch["input_ids"].shape[1]
        fill_frac = length/max_len
        filled.append(fill_frac)
        assert length <= max_len
    print(f"fill frac: {np.mean(filled)}\n"\
          f"min filled: {np.min(filled)}\n"\
          f"max filled: {np.max(filled)}"
          )
    