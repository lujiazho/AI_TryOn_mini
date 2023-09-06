import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        # tokenize text and encode to specific ids
        # eg. a photograph of -> a photo graph of -> 49406,   320,  8853,   539
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # torch.Size([3, 77]) 3 is batch size and 77 is max_length
        tokens = batch_encoding["input_ids"].to(self.device) 
        
        # encode ids to embeddings
        outputs = self.transformer(input_ids=tokens)
        # torch.Size([3, 77, 768]) 3 is batch size, 77 is max_length, 768 is embedding size
        z = outputs.last_hidden_state

        return z

    def encode(self, text):
        return self(text)


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)