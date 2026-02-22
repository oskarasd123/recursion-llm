import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader



class ValDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=8192, subset="sample-10BT"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_val_documents = 0
        self.subset = subset
    
    def __iter__(self):
        return FineWebDataLoader.__iter__(self)


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
        for example in protocol:
            text = example["text"]
            # Tokenize the text
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                #padding="max_length",
                return_tensors="pt"
            )
            
            yield {
                "input_ids": tokens["input_ids"].squeeze(0),
                "text": text,
            }

# Example Usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    import time
    # Load a common tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token # GPT2 doesn't have a pad token by default
    
    # Initialize the dataloader
    dataloader = FineWebDataLoader(tokenizer, subset="sample-10BT", edu=True)
    dataloader = DataLoader(dataloader, 1, num_workers=1)

    iterator = iter(dataloader)
    text1 = next(iterator)["text"]
    text2 = next(iterator)["text"]

    num_tokens = 0
    num_examples = 0
    above_8192_len = 0
    start_time = time.time()
    for example in dataloader:
        if num_examples%10==0:
            print(f"{num_examples} {num_tokens/max(1, num_examples):.2f} {above_8192_len/max(1, num_examples)} {(time.time()-start_time)/max(1, num_examples):.5f}\r", end="")
        numel = example["input_ids"].numel()
        num_tokens += numel
        if example["text"] in [text1, text2] and num_examples > 3:
            print()
            print(f"repeated text at {num_examples}")
            break
        if numel > 8192:
            above_8192_len += 1
        num_examples += 1
        if num_examples > 20_000_000:
            print("dataset loops")
            break
