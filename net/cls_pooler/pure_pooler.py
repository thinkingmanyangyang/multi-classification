from torch import nn
import torch
from transformers.modeling_bert import BertPooler
class PurePooler(nn.Module):
    def __init__(self, config):
        super(PurePooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_outputs):
        # 这里的[:, 0] 代表第0维取全部，第一维取第0个
        last_cls = hidden_outputs[-1][:, 0]
        pooled_output = self.dense(last_cls)
        pooled_output = self.activation(last_cls)
        return pooled_output
        # return pooled_output