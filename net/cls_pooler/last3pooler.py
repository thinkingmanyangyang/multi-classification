from torch import nn
import torch
class Last3ConcatPooler(nn.Module):
    def __init__(self, config):
        super(Last3ConcatPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, hidden_outputs):
        # 这里的[:, 0] 代表第0维取全部，第一维取第0个
        last3cat = torch.cat(
            (hidden_outputs[-1][:, 0],
            hidden_outputs[-2][:, 0],
            hidden_outputs[-3][:, 0]),
            1,
        )
        pooled_output = self.dense(last3cat)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Last3WeightedPooler(nn.Module):
    def __init__(self, config):
        super(Last3WeightedPooler, self).__init__()
        self.avg_weight = nn.Parameter(torch.ones(1, 1, 3))
        self.activation = nn.Tanh()
    def forward(self, hidden_outputs):
        last3cls = torch.stack(
            (hidden_outputs[-1][:, 0],
            hidden_outputs[-2][:, 0],
            hidden_outputs[-3][:, 0]),
            1,
        )

        last3weighted = torch.matmul(
            torch.nn.functional.softmax(self.avg_weight),
            last3cls)
        pooled_output = self.activation(last3weighted)
        return pooled_output.squeeze(1)

class Last3MeanPooler(nn.Module):
    def __init__(self, config):
        super(Last3MeanPooler, self).__init__()
        self.activation = nn.Tanh()
    def forward(self, hidden_outputs):
        last3cls = torch.stack(
            (hidden_outputs[-1][:, 0],
            hidden_outputs[-2][:, 0],
            hidden_outputs[-3][:, 0]),
            1,
        )
        last3mean = torch.mean(last3cls, dim=1)
        pooled_output = self.activation(last3mean)
        return pooled_output



