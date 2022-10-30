import torch
from torch import nn
from net.layers.lstm_layer import LSTMEncoder
class LSTMPooler(nn.Module):
    def __init__(self, config):
        super(LSTMPooler, self).__init__()
        self.lstm_encoder = LSTMEncoder(config)
        self.pooler = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_output, attention_mask):
        lstm_output, batch_max_lengths = self.lstm_encoder(hidden_output, attention_mask)
        attention_mask = attention_mask[:, :batch_max_lengths]
        lstm_output = lstm_output * attention_mask.unsqueeze(-1)
        lstm_mean = torch.sum(lstm_output, dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        lstm_max, _ = torch.max(lstm_output + ((1 - attention_mask) * -10000.0).unsqueeze(-1), dim=1)
        pool_cat = torch.cat((lstm_max, lstm_mean), dim=1)
        pool_output = self.pooler(pool_cat)
        pool_output = self.activation(pool_output)
        return pool_output
