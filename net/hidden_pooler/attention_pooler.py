import torch
from torch import nn
from net.layers.attention_layer import Attention

class AttentionPooler(nn.Module):
    def __init__(self, config):
        super(AttentionPooler, self).__init__()
        config.attention_hidden_size = config.hidden_size
        self.attention = Attention(config)

    def forward(self, hidden_output, attention_mask):
        attention_output = self.attention(hidden_output, attention_mask)
        return attention_output

