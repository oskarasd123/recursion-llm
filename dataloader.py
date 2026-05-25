import os
import random

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader


class FineWebDataLoader(IterableDataset):
    def __init__(self, tokenizer, subset="sample-10BT", edu=False, max_length=8192, num_val_documents = 0, val=False, seed=55):
        """
        Args:
            tokenizer: A tokenizer instance (e.g., from Hugging Face or tiktoken).
            subset: The FineWeb subset name (e.g., 'sample-10BT', 'sample-100BT', 'default').
            max_length: Maximum sequence length for tokenization.
            num_val_documents: number of documents from the beginning to reserve for validation.
        """
        self.subset = subset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.num_val_documents = num_val_documents
        self.val = val
        self.start = 0
        
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb" + ("-edu" if edu else ""), 
            name=subset,
            split="train",
        ).shuffle(seed=self.seed)
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))


    def __iter__(self):
        token_buffer = []
        lengths = []
        texts = []
        range_start = (0 if self.val else self.num_val_documents)
        range_end = (self.num_val_documents+self.start if self.val else len(self.dataset))
        self.start = 0
        for i in range(range_start, range_end):
            if i % self.world_size != self.rank:
                continue
            example = self.dataset[i]
            text = example["text"]
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length-1,
            )["input_ids"]
            tokens = tokens + [self.tokenizer.eos_token_id]
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
        dict = self._prepare_batch(token_buffer, lengths)
        dict["texts"] = texts
        yield dict
    
    def _prepare_batch(self, tokens, lengths):
        cu_seqlens = torch.tensor([0] + torch.cumsum(torch.tensor(lengths), dim=0).tolist(), dtype=torch.int32)
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max(lengths)
        }


class MaxLenFineWebDataLoader(FineWebDataLoader):
    """ensures batches are fully filled to max_length by cuting documents that don't fit into the current batch"""
    def __iter__(self):
        token_buffer = []
        lengths = []
        texts = []
        range_start = (0 if self.val else self.num_val_documents)
        range_end = (self.num_val_documents+self.start if self.val else len(self.dataset))
        self.start = 0
        for i in range(range_start, range_end):
            if i % self.world_size != self.rank:
                continue
            example = self.dataset[i]
            text = example["text"]
            tokens = self.tokenizer(
                text,
                truncation=True
            )["input_ids"]
            tokens = [self.tokenizer.eos_token_id] + tokens + [self.tokenizer.eos_token_id]
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

class InstructionDataset(FineWebDataLoader):
    def __init__(self, tokenizer, max_length=512):
        self.dataset = load_dataset("tatsu-lab/alpaca", split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    def format_prompt(self, example):
        # Common format for instruction tuning
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        response = example.get("output", "")
        
        prompt = (
            f"### Instruction:\n{instruction}\n\n" + 
            (f"### Input:\n{input_text}\n\n" if input_text else "") + 
            f"### Response:\n{response}"
        )
        return prompt

    def __iter__(self):
        token_buffer = []
        lengths = []
        texts = []
        for i in range(0, len(self.dataset)):
            if i % self.world_size != self.rank:
                continue
            example = self.dataset[i]
            text = self.format_prompt(example)
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length-1,
            )["input_ids"]
            tokens = tokens + [self.tokenizer.eos_token_id]
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
        dict = self._prepare_batch(token_buffer, lengths)
        dict["texts"] = texts
        yield dict

class MixDataset(IterableDataset):
    def __init__(self, ds1, ds2, ratio : float):
        "Randomly takes batches from the datasets with a probability (ratio) to take from the second one."
        self.ds1 = ds1
        self.ds2 = ds2
        self.ratio = ratio
    
    def __iter__(self):
        ds1_iter = iter(self.ds1)
        ds2_iter = iter(self.ds2)
        
        while True:
            if random.random() < self.ratio:
                try:
                    yield next(ds2_iter)
                except StopIteration:
                    ds2_iter = iter(self.ds2)
                    yield next(ds2_iter)
            else:
                try:
                    yield next(ds1_iter)
                except StopIteration:
                    ds1_iter = iter(self.ds1)
                    yield next(ds1_iter)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch.nn.functional as F
    from time import perf_counter
    import numpy as np
    from engram import EngramEmbeddings
    from tokenizer_compressor import create_token_compression_map
    # Load a common tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token # GPT2 doesn't have a pad token by default
    max_len = 16348
    # Initialize the dataloader
    dataset = MaxLenFineWebDataLoader(tokenizer, subset="sample-10BT", edu=True, max_length=max_len, num_val_documents=10000)
    dataloader = DataLoader(dataset, 1, num_workers=1)
    val_dataloader = MaxLenFineWebDataLoader(tokenizer, subset="sample-10BT", edu=True, max_length=max_len, num_val_documents=10000, val=True)
    instruction_dataloader = InstructionDataset(tokenizer, max_len)
    
    compression_map = create_token_compression_map(tokenizer, True)
    ngram_order = 2
    engram = EngramEmbeddings(2**20, 64, compression_map, 1, ngram_order)
    

    hashes_func1 = 0
    hashes_func2 = 0
    ideal_hashes = 0
    ideal_uncompressed_hashes = 0
    all_hashes = torch.tensor([], dtype=torch.int32)
    iterator = iter(dataloader)
    filled = []
    batches = 4096
    print("processing statistics")
    for i in range(batches):
        print(f"\r{i}/{batches}", end="")
        batch = next(iterator)
        length = batch["input_ids"].shape[1]
        fill_frac = length/max_len
        filled.append(fill_frac)
        
        hashes_func1 += engram.get_ngram_hashes(batch["input_ids"]).unique().numel()
        hashes_func2 += engram.get_chaotic_ngram_hashes(batch["input_ids"]).unique().numel()
        
        all_hashes = torch.cat([all_hashes, engram.get_chaotic_ngram_hashes(batch["input_ids"]).unique()]).unique()
        
        input_ids = compression_map[batch["input_ids"]]
        padded_ids = F.pad(input_ids, (ngram_order-1, 0), value=0)
        ngrams = padded_ids.unfold(dimension=1, size=ngram_order, step=1).to(torch.int64) # (batch, seq_len, ngram_order)
        pairs = ngrams[:, :, 0] + ngrams[:, :, 1] * len(tokenizer)
        ideal_hashes += pairs.unique().numel()
        # ideal hashes without id compression
        padded_ids = F.pad(batch["input_ids"], (ngram_order-1, 0), value=0)
        ngrams = padded_ids.unfold(dimension=1, size=ngram_order, step=1).to(torch.int64) # (batch, seq_len, ngram_order)
        pairs = ngrams[:, :, 0] + ngrams[:, :, 1] * len(tokenizer)
        ideal_uncompressed_hashes += pairs.unique().numel()
        assert length <= max_len
    print()
    print(f"fill frac: {np.mean(filled)}\n"\
          f"min filled: {np.min(filled)}\n"\
          f"max filled: {np.max(filled)}"
          )
    print("hash function 1 average unique hashes:", hashes_func1/batches)
    print("hash function 2 average unique hashes:", hashes_func2/batches)
    print("ideal average unique hashes:", ideal_hashes/batches)
    print("ideal average uncompressed unique hashes:", ideal_uncompressed_hashes/batches)
    print(f"unused engram table frac: {1-all_hashes.numel()/engram.table_size}")
    batches = 512
    start_time = perf_counter()
    for i in range(batches):
        batch = next(iterator)
        length = batch["input_ids"].shape[1]
        assert length <= max_len
    end_time = perf_counter()
    print(f"dataloader load rate: {batches/(end_time - start_time):.02f}batches/sec, {batches/(end_time - start_time)*max_len:.02f}tokens/sec")
    val_tokens = 0
    for batch in val_dataloader:
        val_tokens += batch["input_ids"].numel()
    print(f"val tokens: {val_tokens}")
    instruction_tokens = 0
    for batch in instruction_dataloader:
        instruction_tokens += len(batch["input_ids"])
    print(f"instruction tokens: {instruction_tokens}")
    print(batch["texts"][0])