import torch
from torch import nn

class MeanMaxPooler(nn.Module):
    def __init__(self, config):
        super(MeanMaxPooler, self).__init__()
        self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
    def forward(self, hidden_output, attention_mask):
        # 转换成bool类型
        torch.unsqueeze(attention_mask, -1)
        hidden_output = hidden_output * attention_mask.unsqueeze(-1)
        hidden_mean = torch.sum(hidden_output, dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        hidden_max, _ = torch.max(hidden_output + ((1 - attention_mask) * -10000.0).unsqueeze(-1), dim=1)
        pool_cat = torch.cat((hidden_max, hidden_mean), dim=1)
        pool_output = self.pooler(pool_cat)
        return pool_output