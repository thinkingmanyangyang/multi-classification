import torch
from torch import nn
import math
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_weight = nn.Linear(config.attention_hidden_size, 1, bias=False)
        self.head_num = 1
    def forward(self, H, mask):
        # mask (batch_size, seq_length)
        mask = (mask > 0).unsqueeze(-1).repeat(1, 1, self.head_num)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        scores = self.attn_weight(H)
        hidden_size = H.size(-1)
        scores /= math.sqrt(float(hidden_size))
        scores += mask
        probs = nn.Softmax(dim=-2)(scores)
        H = H.transpose(-1, -2)
        output = torch.bmm(H, probs)
        output = torch.reshape(output, (-1, hidden_size * self.head_num))
        return output