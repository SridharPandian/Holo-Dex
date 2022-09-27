import math
import torch
import torch.nn as nn

# Adapted from the official MoCoV3 repository: https://github.com/facebookresearch/moco-v3
class MoCo(nn.Module):
    def __init__(self,
        base_encoder,
        momentum_encoder, 
        predictor,
        first_augment_fn,
        sec_augment_fn,
        temperature=1.0
    ):

        super(MoCo, self).__init__()

        self.base_encoder = base_encoder 
        self.momentum_encoder = momentum_encoder
        self.predictor = predictor 
        self.first_augment_fn = first_augment_fn
        self.sec_augment_fn = sec_augment_fn
        self.temperature = temperature

        # Make sure that momentum encoder will not be updated by gradient
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k): 
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temperature
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temperature)

    @property
    def encoder(self):
        return self.base_encoder

    def forward(self, x, m):
        x1 = self.first_augment_fn(x)
        x2 = self.sec_augment_fn(x)
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():
            self._update_momentum_encoder(m)
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

def adjust_moco_momentum(epoch, momentum, total_epochs):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / total_epochs)) * (1. - momentum)
    return m

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim = 0)
    return output

