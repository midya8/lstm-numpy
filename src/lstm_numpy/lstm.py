"""A one-layer LSTM with the forward and backward passes written out by hand.

Weight layout: Wx has shape (4H, D), Wh has shape (4H, H), b has shape
(4H,). The stacked rows are, in order, input gate, forget gate, output
gate, candidate. Nothing here uses an autograd library.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class LSTM:
    def __init__(self, dim_in, dim_h, rng):
        self.D, self.H = dim_in, dim_h

        def orth(n_in, n_out):
            q, _ = np.linalg.qr(rng.normal(size=(n_out, n_in)))
            return q

        self.Wx = np.zeros((4 * self.H, self.D))
        self.Wh = np.zeros((4 * self.H, self.H))
        self.b = np.zeros(4 * self.H)
        for k in range(4):
            self.Wx[k * self.H:(k + 1) * self.H] = orth(self.D, self.H)
            self.Wh[k * self.H:(k + 1) * self.H] = orth(self.H, self.H)
        # a positive forget bias means the cell starts out remembering
        self.b[self.H:2 * self.H] = 1.0

    def cell(self, x, h, c):
        """One timestep. x is (B, D), h and c are (B, H)."""
        h_prev, c_prev = h, c
        z = x @ self.Wx.T + h @ self.Wh.T + self.b
        zi, zf, zo, zg = np.split(z, 4, axis=1)
        i = sigmoid(zi)
        f = sigmoid(zf)
        o = sigmoid(zo)
        g = np.tanh(zg)
        c_new = f * c_prev + i * g
        t = np.tanh(c_new)
        h_new = o * t
        # the cache keeps the pre-update h and c, which is what the
        # backward pass differentiates against
        cache = (x, h_prev, c_prev, i, f, o, g, t)
        return h_new, c_new, cache

    def forward(self, xs, h0=None, c0=None):
        """xs is a list of (B, D) inputs, one per timestep."""
        h = np.zeros((xs[0].shape[0], self.H)) if h0 is None else h0.copy()
        c = np.zeros((xs[0].shape[0], self.H)) if c0 is None else c0.copy()
        hs, cs, caches = [], [], []
        for x in xs:
            h, c, cache = self.cell(x, h, c)
            hs.append(h)
            cs.append(c)
            caches.append(cache)
        return hs, cs, caches

    def backward(self, caches, dhs):
        """Backprop through time given gradients on each output.

        Returns dWx, dWh, db, dxs (gradients on the inputs, list aligned
        with caches) and the gradient on the initial hidden state.
        """
        T = len(caches)
        B, H = dhs[0].shape
        dWx = np.zeros((4 * H, self.D))
        dWh = np.zeros((4 * H, H))
        db = np.zeros(4 * H)
        dxs = [None] * T
        dh = np.zeros((B, H))
        dc = np.zeros((B, H))
        for t in reversed(range(T)):
            x, h_prev, c_prev, i, f, o, g, tanc = caches[t]
            dh = dh + dhs[t]
            do = dh * tanc * o * (1.0 - o)
            dc = dc + dh * o * (1.0 - tanc ** 2)
            df = dc * c_prev * f * (1.0 - f)
            di = dc * g * i * (1.0 - i)
            dg = dc * i * (1.0 - g ** 2)
            dz = np.concatenate([di, df, do, dg], axis=1)
            dWx += dz.T @ x
            dWh += dz.T @ h_prev
            db += dz.sum(axis=0)
            dxs[t] = dz @ self.Wx
            dh = dz @ self.Wh
            dc = dc * f
        return dWx, dWh, db, dxs, dh
