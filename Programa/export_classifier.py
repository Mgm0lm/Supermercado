"""Exporta um classificador Keras H5 para o formato usado pelo caixa."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def load_model(path):
    os.environ["KERAS_BACKEND"] = "torch"
    import h5py
    import keras

    try:
        return keras.models.load_model(path, compile=False)
    except (TypeError, ValueError) as original:
        with h5py.File(path, "r") as file:
            config = json.loads(file.attrs["model_config"])

        def repair(node):
            if isinstance(node, dict):
                if node.get("class_name") == "DepthwiseConv2D":
                    node["config"].pop("groups", None)
                if node.get("class_name") == "Functional":
                    # Keras 2 armazenava até conexões individuais em listas.
                    for key in ("input_layers", "output_layers"):
                        ports = node["config"].get(key)
                        if isinstance(ports, list) and len(ports) == 1:
                            node["config"][key] = ports[0]
                for value in node.values():
                    repair(value)
            elif isinstance(node, list):
                for value in node:
                    repair(value)

        repair(config)
        models = {"Sequential": keras.Sequential, "Functional": keras.Model, "Model": keras.Model}
        model_type = models.get(config["class_name"])
        if model_type is None:
            raise original
        model = model_type.from_config(config["config"])
        model.load_weights(path)
        return model


def export(source, destination):
    import torch

    torch.set_num_threads(2)
    model = load_model(source)
    shape = model.input_shape
    if not isinstance(shape, tuple) or len(shape) != 4 or shape[-1] != 3 or not all(shape[1:]):
        raise ValueError("Esperada entrada RGB NHWC com altura e largura fixas.")
    if len(model.output_shape) != 2 or model.output_shape[-1] < 2:
        raise ValueError("Esperada uma saída com as probabilidades das classes.")
    torch.onnx.export(
        model.eval(), (torch.zeros((1, *shape[1:]), dtype=torch.float32),), str(destination),
        input_names=["image"], output_names=["probabilities"], opset_version=17, dynamo=False,
    )


if __name__ == "__main__":
    if os.name == "nt" and os.environ.get("_PYI_APPLICATION_HOME_DIR"):
        # O interpretador externo herda a pasta de DLLs do executável empacotado.
        import ctypes
        ctypes.windll.kernel32.SetDllDirectoryW(None)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination")
    args = parser.parse_args()
    if args.destination == "-":
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "classifier.onnx"
            with redirect_stdout(sys.stderr):
                export(args.source, output)
            sys.stdout.buffer.write(output.read_bytes())
    else:
        export(args.source, Path(args.destination))
