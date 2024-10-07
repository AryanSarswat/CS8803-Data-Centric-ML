from torch import nn
from transformers import DistilBertModel, DistilBertConfig
from torchsummary import summary
import numpy as np

class TextEncoder(nn.Module):

    def __init__(self, model_name, pretrained=True):
        super(TextEncoder, self).__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            config = DistilBertConfig()
            self.model = DistilBertModel(config)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state.mean(dim=1)
    
if __name__ == "__main__":
    model = TextEncoder(model_name="distilbert-base-uncased")
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Total number of parameters = {params}")