import torch

class FGM(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emd_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emd_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emd_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emd_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}