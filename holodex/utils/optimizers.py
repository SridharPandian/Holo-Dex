import math
import torch
from torch import optim

# From the official VICReg repository
class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay = 0,
        momentum = 0.9,
        eta = 0.001,
        weight_decay_filter = None,
        lars_adaptation_filter = None,
    ):
        defaults = dict(
            lr = lr,
            weight_decay = weight_decay,
            momentum = momentum,
            eta = eta,
            weight_decay_filter = weight_decay_filter,
            lars_adaptation_filter = lars_adaptation_filter,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                dp = parameter.grad

                if dp is None:
                    continue

                if group["weight_decay_filter"] is None or not group["weight_decay_filter"](parameter):
                    dp = dp.add(parameter, alpha = group["weight_decay"])

                if group["lars_adaptation_filter"] is None or not group["lars_adaptation_filter"](parameter):
                    param_norm = torch.norm(parameter)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, 
                            (group["eta"] * param_norm / update_norm), 
                            one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[parameter]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(parameter)
                mu = param_state["mu"]
                mu.mul_(group["momentum"]).add_(dp)

                parameter.add_(mu, alpha=-group["lr"])

def adjust_learning_rate(options, optimizer, loader, step):
    max_steps = options.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = options.lr * options.batch_size / 256

    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr

def exclude_bias_and_norm(p):
    return p.ndim == 1