import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
