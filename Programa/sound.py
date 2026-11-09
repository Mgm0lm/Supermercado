"""Gera um bip curto (estilo leitor de caixa de mercado) como .wav,
sem depender de nenhum arquivo de audio externo.
"""

import math
import os
import struct
import tempfile
import wave


def _sine_samples(freq, duration_ms, sample_rate=44100, volume=0.6):
    n_samples = max(1, int(sample_rate * duration_ms / 1000))
    fade = max(1, int(n_samples * 0.08))  # fade in/out curto pra evitar "click"
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = math.sin(2 * math.pi * freq * t)
        if i < fade:
            value *= i / fade
        elif i > n_samples - fade:
            value *= (n_samples - i) / fade
        samples.append(int(max(-1.0, min(1.0, value)) * volume * 32767))
    return samples


def generate_checkout_beep(freq=1800, duration_ms=110, volume=0.6, path=None):
    """Gera um bip simples e salva como .wav temporario. Retorna o caminho."""
    if path is None:
        fd, path = tempfile.mkstemp(suffix="_beep.wav", prefix="pdv_")
        os.close(fd)

    sample_rate = 44100
    samples = _sine_samples(freq, duration_ms, sample_rate, volume)

    with wave.open(path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack("<" + "h" * len(samples), *samples))

    return path
