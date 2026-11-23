import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lstm_numpy.lstm import LSTM
from lstm_numpy.data import build
from lstm_numpy.metrics import balanced


def loss_probe(model, xs, R):
    hs, _, caches = model.forward(xs)
    return sum(float(np.sum(h * R)) for h in hs)


def num_grad(f, p, eps=1e-5):
    g = np.zeros_like(p)
    it = np.nditer(p, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = p[i]
        p[i] = old + eps
        up = f()
        p[i] = old - eps
        dn = f()
        p[i] = old
        g[i] = (up - dn) / (2 * eps)
        it.iternext()
    return g


def test_cell_gradcheck():
    rng = np.random.default_rng(3)
    D, H, B, T = 3, 4, 2, 3
    model = LSTM(D, H, rng)
    xs = [rng.normal(size=(B, D)) for _ in range(T)]
    R = rng.normal(size=(B, H))
    dhs = [R for _ in range(T)]  # dL/dh_t is R for every timestep
    dWx, dWh, db, _, _ = model.backward(model.forward(xs)[2], dhs)
    for name, p in [("Wx", model.Wx), ("Wh", model.Wh), ("b", model.b)]:
        def f():
            return loss_probe(model, xs, R)
        ng = num_grad(f, p)
        g = {"Wx": dWx, "Wh": dWh, "b": db}[name]
        assert np.max(np.abs(ng - g)) < 1e-6, name


def test_carried_state_matches_full_pass():
    rng = np.random.default_rng(7)
    D, H, B = 5, 6, 2
    model = LSTM(D, H, rng)
    xs = [rng.normal(size=(B, D)) for _ in range(6)]
    h1, c1 = None, None
    hs_full, cs_full, _ = model.forward(xs)
    # run the same sequence in two chunks, carrying the state
    h = np.zeros((B, H))
    c = np.zeros((B, H))
    first, cs_a, _ = model.forward(xs[:3], h0=h, c0=c)
    second, _, _ = model.forward(xs[3:], h0=first[-1], c0=cs_a[-1])
    assert np.max(np.abs(second[-1] - hs_full[-1])) < 1e-10
    assert np.max(np.abs(cs_a[-1] - cs_full[2])) < 1e-10


def test_balanced_counter():
    assert balanced(["((1+2))", "3", "()()"]) == 1.0
    assert balanced(["(()", ")("]) == 0.0


def test_training_loss_drops():
    from lstm_numpy.train import train_run
    d = build()
    model, params, chars, losses = train_run(d["parens"], steps=90, batch=8,
                                             seq_len=24, hidden=16, seed=2, log_every=30)
    first = [v for _, v in losses][:3]
    last = [v for _, v in losses][-1]
    assert last < first[0]
