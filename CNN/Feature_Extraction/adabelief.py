import torch


class AdaBelief(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, s = state["step"], state["m"], state["s"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                diff = g - m
                s.mul_(beta2).addcmul_(diff, diff, value=1 - beta2).add_(group["eps"])
                m_hat = m / (1 - beta1 ** t)
                s_hat = s / (1 - beta2 ** t)
                p.addcdiv_(m_hat, s_hat.sqrt().add_(group["eps"]), value=-group["lr"])
        return loss
