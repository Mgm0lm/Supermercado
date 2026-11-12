"""Inferência do classificador de produtos com o mesmo runtime do OCR."""

import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import onnxruntime


class ProductClassifier:
    def __init__(self, path):
        path = Path(path).resolve()
        if path.suffix.lower() == ".h5":
            python = os.environ.get("CLASSIFIER_PYTHON") or (
                "python" if getattr(sys, "frozen", False) else sys.executable
            )
            exporter = Path(__file__).with_name("export_classifier.py")
            converted = subprocess.run(
                [python, str(exporter), str(path), "-"],
                capture_output=True, timeout=180,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if converted.returncode:
                raise ValueError(converted.stderr.decode("utf-8", "replace").strip())
            source = converted.stdout
        elif path.suffix.lower() == ".onnx":
            source = str(path)
        else:
            raise ValueError("O classificador deve ser um arquivo .h5 ou .onnx.")

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = onnxruntime.InferenceSession(source, options, providers=["CPUExecutionProvider"])
        (entry,) = self.session.get_inputs()
        (output,) = self.session.get_outputs()
        batch, height, width, channels = entry.shape
        if (batch, channels) != (1, 3) or not all(
            isinstance(side, int) and side > 0 for side in (height, width)
        ):
            raise ValueError("Esperada entrada fixa [1, altura, largura, 3] no classificador.")
        self.size = (width, height)
        self.input_name = entry.name

        aliases = {"Miojo": "Nissin Miojo", "Coca-Cola": "Coca 600ml"}
        self.labels = {}
        labels = path.with_name("labels.txt")
        for line in labels.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            index, label = line.split(maxsplit=1)
            label = label.strip()
            index = int(index)
            if index in self.labels:
                raise ValueError(f"Índice repetido em {labels}: {index}")
            self.labels[index] = aliases.get(label, label)
        if (
            len(output.shape) != 2 or output.shape[0] != 1
            or not isinstance(output.shape[1], int) or output.shape[1] < 2
            or set(self.labels) != set(range(output.shape[1]))
        ):
            raise ValueError(f"Os índices de {labels} devem corresponder às saídas do modelo.")

    def predict(self, frame, box, threshold, products):
        left, top, right, bottom = box
        crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
        if not crop.size:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_LINEAR)
        batch = (rgb.astype(np.float32) / 127.5 - 1.0)[None, ...]
        (predictions,) = self.session.run(None, {self.input_name: batch})
        if not np.isfinite(predictions).all():
            return None
        index = int(np.argmax(predictions[0]))
        label, probability = self.labels[index], float(predictions[0, index])
        if label in products and label.casefold() != "nada" and threshold <= probability <= 1.0:
            return label, probability
