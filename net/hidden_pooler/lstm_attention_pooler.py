import torch
from torch import nn
from net.layers.lstm_layer import LSTMEncoder
from net.layers.attention_layer import Attention
class LSTMAttentionPooler(nn.Module):
    def __init__(self, config):
        super(LSTMAttentionPooler, self).__init__()
        self.lstm_encoder = LSTMEncoder(config)
        config.attention_hidden_size = config.hidden_size * 2
        self.attention = Attention(config)
        self.pooler = nn.Linear(config.attention_hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_output, attention_mask):
        lstm_output, batch_max_lengths = self.lstm_encoder(hidden_output, attention_mask)
        attention_mask = attention_mask[:, :batch_max_lengths]
        pooled_output = self.attention(lstm_output, attention_mask)
        pooled_output = self.pooler(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output
