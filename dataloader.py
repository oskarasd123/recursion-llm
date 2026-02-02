import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader



class ValDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=8192):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_val_documents = 0
    
    def __iter__(self):
        return FineWebEduDataLoader.__iter__(self)


class FineWebEduDataLoader(IterableDataset):
    def __init__(self, tokenizer, subset="sample-10BT", max_length=8192, num_val_documents = 1000):
        """
        Args:
            tokenizer: A tokenizer instance (e.g., from Hugging Face or tiktoken).
            subset: The FineWeb-Edu subset name (e.g., 'sample-10BT', 'sample-100BT', 'default').
            split: Dataset split to load.
            max_length: Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # We use streaming=True to avoid downloading the entire subset to disk
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name=subset, 
            split="train", 
            streaming=(subset != "sample-10BT")
        )
        self.num_val_documents = num_val_documents
        self.data_iter = iter(self.dataset)
        self.val_data = ValDataset(self.dataset.take(num_val_documents), tokenizer, max_length)


    def __iter__(self):
        for example in self.dataset.skip(self.num_val_documents):
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
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }

def get_fineweb_dataloader(tokenizer, subset="sample-10BT", batch_size=1, max_length=8192):
    ds = FineWebEduDataLoader(tokenizer, subset=subset, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size)

# Example Usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Load a common tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token # GPT2 doesn't have a pad token by default
    
    # Initialize the dataloader
    dataloader = get_fineweb_dataloader(tokenizer, subset="sample-10BT", batch_size=1)
    
    # Fetch one batch
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"First 10 tokens of first sample: {batch['input_ids'][0][:10]}")