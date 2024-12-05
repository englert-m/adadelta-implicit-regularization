import math

import torch
from torch.optim.optimizer import Optimizer, required


class BaseModel(torch.nn.Module):
    """Basic model that provides a method to get the norm of the network to the power
    of the homogeneity exponent of then network when the network consists of
    linear and/or convolutional layers."""

    number_of_layers = None

    @torch.no_grad()
    def get_number_of_layers(self):
        # recursively go through the children and count all linear and convolutional layers
        def count_layers(module):
            count = 0
            for child in module.children():
                if isinstance(child, (torch.nn.Linear, torch.nn.Conv2d)):
                    count += 1
                elif isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)):
                    count += count_layers(child)
            return count

        return count_layers(self)

    @torch.no_grad
    def get_norm_power(self):
        if self.number_of_layers is None:
            self.number_of_layers = self.get_number_of_layers()
        return torch.pow(sum([torch.sum(p**2) for p in self.parameters()]), self.number_of_layers / 2)


class GeneralizedAdadelta(Optimizer):
    def __init__(self, params, lr=1.0, scale=0.0, rho=required):
        # rho should be an iterable which gives a decay factor for each step.

        defaults = dict(lr=lr, rho=rho)
        super(GeneralizedAdadelta, self).__init__(params, defaults)
        self.state = dict()
        for group in self.param_groups:
            group["eps"] = []
            group["delta"] = []
            for p in group["params"]:
                self.state[p] = dict(u=torch.zeros_like(p, requires_grad=False), v=torch.zeros_like(p, requires_grad=False))

                e = -5 * torch.ones_like(p, requires_grad=False) + torch.randn_like(p, requires_grad=False) * scale
                d = -5 * torch.ones_like(p, requires_grad=False) + torch.randn_like(p, requires_grad=False) * scale
                group["eps"].append(10**e)
                group["delta"].append(10**d)

    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            rho = next(group["rho"])
            for p, eps, delta in zip(group["params"], group["eps"], group["delta"]):
                if p.grad is None:
                    continue

                state = self.state[p]
                state["v"].mul_(rho).addcmul_(p.grad, p.grad, value=1 - rho)
                deltax = state["u"].add(eps).div_(state["v"].add(delta)).sqrt_().mul_(p.grad)
                state["u"].mul_(rho).addcmul_(deltax, deltax, value=1 - rho)
                p.add_(deltax, alpha=-group["lr"])

    @torch.no_grad
    def get_reciprocal(self):
        result = []
        for group in self.param_groups:
            for p, eps, delta in zip(group["params"], group["eps"], group["delta"]):
                state = self.state[p]
                result.append(torch.rsqrt(torch.sqrt((state["u"] + eps) / (state["v"] + delta))))
        return torch.cat(result, dim=1).squeeze()


def get_adadelta(params, scale=0.0, lr=1.0):
    def constant_value_generator(value):
        while True:
            yield value

    return GeneralizedAdadelta(params, lr=lr, scale=scale, rho=constant_value_generator(0.9))


def get_adadeltaS(params, scale=0.0, total_steps=5000, lr=1.0):
    decay_schedule = [1 - 0.1 / (1 + math.floor(100 * i / total_steps)) for i in range(total_steps)]
    return GeneralizedAdadelta(params, lr=lr, scale=scale, rho=iter(decay_schedule))
