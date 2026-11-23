"""Trains both runs and writes results.json, the loss curves and weights."""

import json
import os
import time

import numpy as np

from .data import build
from .train import train_run
from .metrics import (NgramLM, balanced, ce_lstm, ce_ngram, sample_lstm,
                      split_train_eval)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets")


def run_dataset(name, text, steps, hidden, seed, temperature=0.8):
    train, ev = split_train_eval(text)
    t0 = time.time()
    model, params, chars, losses = train_run(
        train, steps=steps, batch=32, seq_len=48, hidden=hidden, seed=seed,
        lr=2e-3, log_every=100)
    train_seconds = round(time.time() - t0, 1)
    results = {
        "steps": steps, "hidden": hidden, "batch": 32, "seq_len": 48,
        "seed": seed, "train_chars": len(train), "eval_chars": len(ev),
        "train_seconds": train_seconds,
        "ce_lstm": round(ce_lstm(model, params, chars, ev), 4),
        "ce_ngram": round(ce_ngram(train, ev), 4),
    }
    rng = np.random.default_rng(seed + 1)
    samples = sample_lstm(model, params, chars, rng, n_strings=200,
                          temperature=temperature)
    lm = NgramLM(train)
    ng_samples = lm.sample(rng, 200)
    if name == "parens":
        results["balanced_lstm"] = round(balanced(samples), 4)
        results["balanced_ngram"] = round(balanced(ng_samples), 4)
    results["sample_lstm"] = samples[:5]
    results["sample_ngram"] = ng_samples[:5]
    np.savez(os.path.join(ASSETS, f"weights_{name}.npz"), **params)
    return model, chars, losses, results


def main():
    os.makedirs(ASSETS, exist_ok=True)
    data = build()
    curves, out = {}, {}
    for name, steps, hidden, seed in [("sentences", 3000, 96, 11),
                                      ("parens", 3000, 96, 23)]:
        model, chars, losses, results = run_dataset(name, data[name], steps,
                                                    hidden, seed)
        out[name] = results
        curves[name] = losses
        print(name, json.dumps(results)[:200])
    with open(os.path.join(ASSETS, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, name in zip(axes, ["sentences", "parens"]):
        t, v = zip(*curves[name])
        ax.plot(t, v)
        ax.set_title(f"{name}: training loss")
        ax.set_xlabel("step")
        ax.set_ylabel("cross entropy")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_loss.png"), dpi=110)


if __name__ == "__main__":
    main()
