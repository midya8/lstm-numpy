"""Baselines, sampling and the two checks the README reports."""

import numpy as np

from .train import softmax


class NgramLM:
    """Interpolated character n-gram model over orders 3, 2, 1, 0.

    Each order is add-alpha smoothed, then the orders are mixed with
    fixed weights. Strong enough to be a real baseline on repetitive
    text, weak enough to fail at counting parentheses.
    """

    ORDERS = (3, 2, 1)
    WEIGHTS = {3: 0.6, 2: 0.25, 1: 0.1}

    def __init__(self, text, alpha=0.5):
        self.chars = sorted(set(text))
        self.stoi = {c: j for j, c in enumerate(self.chars)}
        self.alpha = alpha
        V = len(self.chars)
        self.counts = {}
        for k in self.ORDERS:
            self.counts[k] = {}
            for i in range(k, len(text)):
                ctx = text[i - k:i]
                self.counts[k].setdefault(ctx, np.zeros(V))
                self.counts[k][ctx][self.stoi[text[i]]] += 1
        uni = np.zeros(V)
        for ch in text:
            uni[self.stoi[ch]] += 1
        self.counts[0] = uni / uni.sum()

    def dist(self, history):
        V = len(self.chars)
        p = self.WEIGHTS[1] * self.counts[0]
        for k in self.ORDERS:
            ctx = history[-k:]
            c = self.counts[k].get(ctx)
            if c is None or c.sum() == 0:
                continue
            pk = (c + self.alpha) / (c.sum() + self.alpha * V)
            p = p + self.WEIGHTS[k] * pk
        return p / p.sum()

    def sample(self, rng, n_strings, max_len=60, stop="\n", temp=1.0):
        out = []
        for _ in range(n_strings):
            s = ""
            while len(s) < max_len:
                p = self.dist(s[-3:]) ** (1.0 / temp)
                p = p / p.sum()
                nxt = self.chars[rng.choice(len(p), p=p)]
                if nxt == stop:
                    break
                s += nxt
            out.append(s)
        return out


def split_train_eval(text, frac=0.9):
    cut = int(len(text) * frac)
    cut = text.index("\n", cut) + 1  # never split a line
    return text[:cut], text[cut:]


def ce_lstm(model, params, chars, eval_text):
    """Next-char cross entropy on held-out text, state reset per line."""
    stoi = {ch: j for j, ch in enumerate(chars)}
    V = len(chars)
    total, n = 0.0, 0
    for line in eval_text.split("\n"):
        if not line:
            continue
        idx = np.array([stoi[ch] for ch in line])
        h = np.zeros((1, model.H))
        c = np.zeros((1, model.H))
        for t in range(len(idx) - 1):
            x = np.eye(V)[idx[t:t + 1]]
            h, c, _ = model.cell(x, h, c)
            logits = h @ params["Wo"].T + params["bo"]
            p = softmax(logits)[0]
            total += -np.log(p[idx[t + 1]] + 1e-12)
            n += 1
    return total / n


def ce_ngram(train_text, eval_text):
    lm = NgramLM(train_text)
    total, n = 0.0, 0
    for line in eval_text.split("\n"):
        if not line:
            continue
        for i in range(len(line)):
            p = lm.dist(line[:i])
            total += -np.log(p[lm.stoi[line[i]]] + 1e-12)
            n += 1
    return total / n


def balanced(strings):
    """Fraction of strings whose parentheses balance, ( counts +1 and ) -1."""
    ok = 0
    for s in strings:
        depth, good = 0, True
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                good = False
                break
        if good and depth == 0:
            ok += 1
    return ok / len(strings)


def sample_lstm(model, params, chars, rng, n_strings=200, temperature=0.8,
                max_len=60, stop="\n"):
    stoi = {ch: j for j, ch in enumerate(chars)}
    V = len(chars)
    out = []
    for _ in range(n_strings):
        s = stop
        h = np.zeros((1, model.H))
        c = np.zeros((1, model.H))
        while len(s) < max_len:
            x = np.eye(V)[np.array([stoi[s[-1]]])]
            h, c, _ = model.cell(x, h, c)
            logits = h[0] @ params["Wo"].T + params["bo"]
            p = softmax(logits[None, :])[0] ** (1.0 / temperature)
            p = p / p.sum()
            nxt = chars[rng.choice(V, p=p)]
            if nxt == stop:
                break
            s += nxt
        out.append(s[len(stop):])
    return out
