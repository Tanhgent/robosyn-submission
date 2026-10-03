"""muon_lite.py — Muon (Jordan et al., 2024) + aux AdamW hybrid, single-file.

Muon path (params with ndim>=2, conv viewed as (out, -1)):
    buf = mu*buf + g ; g_eff = g + mu*buf (nesterov) ; U = NewtonSchulz5(g_eff)
    p -= lr * U * sqrt(max(1, m/n))
NewtonSchulz5: official quintic coefficients (3.4445, -4.7750, 2.0315), 5 iters, bf16.
Aux AdamW path (1-D params: biases, norms): standard AdamW at aux_ratio*lr
    (official practice pairs Muon with a separately-tuned AdamW; we tie it by a
     fixed ratio so the harness's single-lr sweep remains meaningful).
State (momentum buf / m,v) lives in self.state so bytes_per_param() measures it.
NOTE: MuonClip's QK-Clip is an attention-logit guard (transformer-only, large-scale
      trigger) — no-op for MLP/conv nets, hence plain Muon is the faithful subject.
"""
import torch


def _newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 aux_ratio=0.15, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, aux_ratio=aux_ratio,
                                      betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, mu = g["lr"], g["momentum"]
            b1, b2, eps = g["betas"][0], g["betas"][1], g["eps"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if p.ndim >= 2 and p.shape[0] < 10000:            # ── Muon path (임베딩류 초대형 행렬은 aux로 — 공식 관행)
                    G = p.grad.reshape(p.shape[0], -1)
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(G)
                    st["buf"].mul_(mu).add_(G)
                    g_eff = G.add(st["buf"], alpha=mu) if g["nesterov"] else st["buf"]
                    U = _newton_schulz5(g_eff, g["ns_steps"])
                    scale = max(1.0, G.size(0) / G.size(1)) ** 0.5
                    if g["weight_decay"]:
                        p.mul_(1 - lr * g["weight_decay"])
                    p.add_(U.reshape(p.shape), alpha=-lr * scale)
                else:                                             # ── aux AdamW path
                    _t = st.get("t", None)
                    _t = (int(_t.item()) if hasattr(_t, "item") else int(_t or 0)) + 1
                    st["t"] = torch.tensor(_t, dtype=torch.long); t = _t
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p); st["v"] = torch.zeros_like(p)
                    st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                    st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    alr = lr * g["aux_ratio"]
                    upd = (st["m"] / (1 - b1 ** t)) / ((st["v"] / (1 - b2 ** t)).sqrt() + eps)
                    if g["weight_decay"]:
                        p.mul_(1 - alr * g["weight_decay"])
                    p.add_(upd, alpha=-alr)
