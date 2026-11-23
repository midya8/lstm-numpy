"""Adam and the training loop, both written without a framework."""

import numpy as np

from .lstm import LSTM


def adam_update(params, grads, m, v, t, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
    for k in params:
        m[k] = b1 * m[k] + (1 - b1) * grads[k]
        v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
        mhat = m[k] / (1 - b1 ** t)
        vhat = v[k] / (1 - b2 ** t)
        params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
    return m, v


def clip_grads(grads, limit=5.0):
    for k in grads:
        n = np.linalg.norm(grads[k])
        if n > limit:
            grads[k] *= limit / n
    return grads


def make_batches(rng, text, stoi, batch, seq_len):
    """Random windows of chars. Inputs run one char behind the targets."""
    starts = rng.integers(0, len(text) - seq_len - 1, size=batch)
    codes = np.array([stoi[c] for c in text])
    idx = codes[starts[:, None] + np.arange(seq_len + 1)]
    return idx[:, :-1], idx[:, 1:]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_run(text, steps=3000, batch=32, seq_len=48, hidden=96, seed=11,
              lr=2e-3, log_every=200, rng=None):
    rng = rng or np.random.default_rng(seed)
    chars = sorted(set(text))
    stoi = {c: j for j, c in enumerate(chars)}
    V = len(chars)
    model = LSTM(V, hidden, rng)
    Wo = rng.normal(size=(V, hidden)) / np.sqrt(hidden)
    bo = np.zeros(V)
    params = {"Wx": model.Wx, "Wh": model.Wh, "b": model.b, "Wo": Wo, "bo": bo}
    m = {k: np.zeros_like(p) for k, p in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    losses = []
    for t in range(1, steps + 1):
        xs_idx, ys_idx = make_batches(rng, text, stoi, batch, seq_len)
        xs = [np.eye(V)[xs_idx[:, s]] for s in range(seq_len)]
        hs, cs, caches = model.forward(xs)
        loss = 0.0
        dhs = [np.zeros((batch, hidden)) for _ in range(seq_len)]
        dWo = np.zeros_like(Wo)
        dbo = np.zeros(V)
        scale = 1.0 / (batch * seq_len)
        for s in range(seq_len):
            logits = hs[s] @ Wo.T + bo
            p = softmax(logits)
            loss += -np.mean(np.log(p[np.arange(batch), ys_idx[:, s]] + 1e-12))
            dlogit = p.copy()
            dlogit[np.arange(batch), ys_idx[:, s]] -= 1.0
            dlogit *= scale
            dWo += dlogit.T @ hs[s]
            dbo += dlogit.sum(axis=0)
            dhs[s] += dlogit @ Wo
        dWx, dWh, db, _, _ = model.backward(caches, dhs)
        grads = {"Wx": dWx, "Wh": dWh, "b": db, "Wo": dWo, "bo": dbo}
        clip_grads(grads)
        adam_update(params, grads, m, v, t, lr=lr)
        if t % log_every == 0 or t == 1:
            losses.append((t, loss / seq_len))
    return model, params, chars, losses
