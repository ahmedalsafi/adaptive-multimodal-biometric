"""
benchmark_runtime.py
--------------------
Unified runtime benchmark for the adaptive multimodal biometric system
(paper Table 6). Run this LOCALLY on the target CPU (CPU-only, not on Colab/GPU).

Usage:
    Place this script in the same folder as the model files, then run:
        python benchmark_runtime.py

Requirements: tensorflow, numpy, psutil

Models required (place in same directory as this script):
    face_model_final_fixedfeb.keras
    hand_model_final_v2.keras
"""

import os
import gc
import time
import numpy as np
import tensorflow as tf
import psutil

# ------------------------------------------------------------------ config
FACE_KERAS   = "face_model_final_fixedfeb.keras"
HAND_KERAS   = "hand_model_final_v2.keras"
IMG_SIZE     = 200
N_RUNS       = 200       # timed passes per model (median reported)
WARMUP       = 10
HAND_INVOKED = 21        # clean test set: 21/505 hand invocations
N_PAIRS      = 505
P_HAND       = HAND_INVOKED / N_PAIRS   # 0.0416

WORKDIR = os.path.dirname(os.path.abspath(__file__))
dummy   = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)


# ------------------------------------------------------------------ helpers
def kib(path): return os.path.getsize(path) / 1024
def mib(path): return os.path.getsize(path) / 1024 / 1024

def time_keras(model, n=N_RUNS):
    # Wrap in tf.function so the graph is traced once, not rebuilt each call
    infer = tf.function(lambda x: model(x, training=False))
    infer(dummy)                    # trace once
    for _ in range(WARMUP):
        infer(dummy)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        infer(dummy)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.std(t))

def time_tflite(path, n=N_RUNS):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    idx = interp.get_input_details()[0]["index"]
    for _ in range(WARMUP):
        interp.set_tensor(idx, dummy); interp.invoke()
    t = []
    for _ in range(n):
        interp.set_tensor(idx, dummy)
        t0 = time.perf_counter()
        interp.invoke()
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.std(t))

def convert_tflite(model, out, quantize=False):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    open(out, "wb").write(conv.convert())
    return out

def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def adaptive(f, h): return f + P_HAND * h
def speedup(f, h):  return (f + h) / adaptive(f, h)


# ------------------------------------------------------------------ run
print("=" * 68)
print("RUNTIME BENCHMARK  (200 single-sample passes, median, CPU-only)")
print(f"P_hand = {HAND_INVOKED}/{N_PAIRS} = {P_HAND:.4f}")
print("=" * 68)

gc.collect()
base_ram = rss_mb()
face_k = tf.keras.models.load_model(FACE_KERAS)
hand_k = tf.keras.models.load_model(HAND_KERAS)
_ = face_k(dummy); _ = hand_k(dummy)
keras_infer_ram = rss_mb()

kf, kf_s = time_keras(face_k)
kh, kh_s = time_keras(hand_k)
keras_size = mib(FACE_KERAS)

f_tfl = convert_tflite(face_k, os.path.join(WORKDIR, "face_f32.tflite"))
h_tfl = convert_tflite(hand_k, os.path.join(WORKDIR, "hand_f32.tflite"))
gc.collect(); base2 = rss_mb()
tf_face, tf_face_s = time_tflite(f_tfl)
tf_hand, tf_hand_s = time_tflite(h_tfl)
tfl_ram = rss_mb() - base2
tfl_size = mib(f_tfl)

f_q = convert_tflite(face_k, os.path.join(WORKDIR, "face_q.tflite"), quantize=True)
h_q = convert_tflite(hand_k, os.path.join(WORKDIR, "hand_q.tflite"), quantize=True)
gc.collect(); base3 = rss_mb()
q_face, q_face_s = time_tflite(f_q)
q_hand, q_hand_s = time_tflite(h_q)
q_ram = rss_mb() - base3
q_size_kib = kib(f_q)


# ------------------------------------------------------------------ report
def row(name, size_str, ram, f, fs, h, hs):
    a = adaptive(f, h); s = speedup(f, h)
    print(f"{name:<22}{size_str:>10}{'+'+format(ram,'.1f')+' MB':>12}"
          f"{f:>7.2f}±{fs:.2f}{h:>7.2f}±{hs:.2f}{a:>8.2f}{s:>7.2f}x")

print(f"\n{'Format':<22}{'Size':>10}{'RAM':>12}{'Face(ms)':>12}"
      f"{'Hand(ms)':>10}{'Adap':>8}{'Speedup':>8}")
print("-" * 84)
row("Keras (float32)",    f"{keras_size:.2f} MiB", keras_infer_ram - base_ram, kf, kf_s, kh, kh_s)
row("TFLite (float32)",   f"{tfl_size:.2f} MiB",   tfl_ram, tf_face, tf_face_s, tf_hand, tf_hand_s)
row("TFLite (quantized)", f"{q_size_kib:.0f} KiB", q_ram,   q_face,  q_face_s,  q_hand,  q_hand_s)
print("-" * 84)
print(f"Quantized size reduction vs Keras float32: "
      f"{(1 - q_size_kib/(keras_size*1024))*100:.1f}%")
print("\nNote: Results may vary with system load. "
      "Paper Table 6 was measured on an unloaded CPU.")
