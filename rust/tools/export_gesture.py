"""Converte os pesos de uma MLP binária treinada pelo projeto para JSON."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def export(source, destination):
    # Joblib usa pickle. Abra somente pesos de origem conhecida.
    model = joblib.load(source)
    if hasattr(model, "steps"):
        scaler, mlp = (step for _, step in model.steps)
        mean, scale = scaler.mean_, scaler.scale_
    else:
        mlp = model
        mean, scale = np.zeros(126), np.ones(126)
    if (
        list(mlp.classes_) != [0, 1]
        or mlp.n_features_in_ != 126
        or mlp.out_activation_ != "logistic"
    ):
        raise ValueError("Esperada MLP binária (classes 0 e 1), com 126 entradas.")
    layers = [
        {"weights": w.tolist(), "biases": b.tolist()}
        for w, b in zip(mlp.coefs_, mlp.intercepts_, strict=True)
    ]
    payload = {
        "activation": mlp.activation,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "layers": layers,
    }
    destination.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
