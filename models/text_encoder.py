import numpy as np
import torch
from torch import nn
from torchsummary import summary
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer


class TextEncoder(nn.Module):

    def __init__(self, model_name, device=None, pretrained=True):
        super(TextEncoder, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            config = DistilBertConfig()
            self.model = DistilBertModel(config)

        if device:
            self.device = device

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state.mean(dim=1)
    
    def tokenize(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=50, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        
        return input_ids, attention_mask

    def get_text_features(self, text):
        input_ids, attention_mask = self.tokenize(text)
        return self(input_ids, attention_mask)
    
if __name__ == "__main__":
    model = TextEncoder(model_name="distilbert-base-uncased")
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Total number of parameters = {params}")
    texts = [
        "Hello, my dog is cute.",
        "This is interesting",
    ]
    print(model.get_text_features(texts).shape)