"""Regenerate tests/data/snic_reference_parity.npz from the original SNIC code.

Downloads snic.c/snic.h of the authors' reference implementation
(github.com/achanta/SNIC, pinned below), builds it as a shared library with the
compiler setuptools finds, runs SNIC_main (doRGBtoLAB=0) on the cases below and
stores the inputs, the seeds the reference placed (from its FindSeeds) and its
labels. The reference code is not redistributed and cdts does not use its seed
grid; only its outputs are kept, and the tests feed cdts the same seeds.

    python tests/data/make_snic_parity_fixtures.py
"""
import ctypes
import os
import sys
import tempfile
import urllib.request

import numpy as np

COMMIT = "f145cac83eb3158ee3bb09540016ce502e1b2366"
BASE = f"https://raw.githubusercontent.com/achanta/SNIC/{COMMIT}/snic_python_interface/"
HERE = os.path.dirname(os.path.abspath(__file__))


def build_reference(workdir):
    from setuptools._distutils.ccompiler import new_compiler
    from setuptools._distutils.sysconfig import customize_compiler

    for name in ("snic.c", "snic.h"):
        urllib.request.urlretrieve(BASE + name, os.path.join(workdir, name))
    cc = new_compiler()
    customize_compiler(cc)
    msvc = cc.compiler_type == "msvc"
    objs = cc.compile([os.path.join(workdir, "snic.c")], output_dir=workdir,
                      extra_postargs=["/O2"] if msvc else ["-O2", "-fPIC", "-ffp-contract=off"])
    lib = os.path.join(workdir, "snic_ref" + (".dll" if sys.platform == "win32" else ".so"))
    cc.link(cc.SHARED_LIBRARY, objs, lib, export_symbols=["SNIC_main", "FindSeeds"] if msvc else None,
            extra_postargs=["/DLL"] if msvc else ["-shared"])
    ref = ctypes.CDLL(lib)
    ref.SNIC_main.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_double, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    ref.FindSeeds.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    return ref


def reference_seeds(ref, height, width, n_segments):
    """(row, col) of the reference's seeds, in its cluster order."""
    cap = 2 * n_segments + 64
    kx = np.zeros(cap, dtype=np.int32)
    ky = np.zeros(cap, dtype=np.int32)
    n = ctypes.c_int(0)
    ref.FindSeeds(width, height, n_segments, kx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                  ky.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.byref(n))
    return np.column_stack([ky[:n.value], kx[:n.value]]).astype(np.int16)


def run_reference(ref, img, n_segments, compactness):
    f, h, w = img.shape
    buf = np.ascontiguousarray(img, dtype=np.float64).copy()
    labels = np.empty(h * w, dtype=np.int32)
    numlabels = ctypes.c_int(0)
    ref.SNIC_main(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), w, h, f, n_segments, compactness, 0,
                  labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.byref(numlabels))
    return labels.reshape(h, w), numlabels.value


def make_cases():
    rng = np.random.default_rng(20170721)
    cases = {}
    # 8-bit multiband image, the reference's native use (as integers the
    # feature sums are exact, whatever the summation order)
    cases["uint8_rgb"] = (rng.integers(0, 256, (3, 48, 64)).astype(np.float32), 60, 10.0)
    # piecewise-constant image: many equal distances, exercises heap tie order
    blocks = rng.integers(0, 4, (2, 9, 11)).astype(np.float32)
    cases["piecewise_ties"] = (np.kron(blocks, np.ones((1, 6, 6), np.float32))[:, :50, :61], 40, 5.0)
    # white noise, real-valued
    cases["noise_float"] = (rng.normal(size=(5, 45, 37)).astype(np.float32), 30, 0.5)
    # spatially smooth, time-series-like cube: 12 dates x 2 bands of an
    # NDVI-like seasonal curve whose phase drifts across the image
    t = np.linspace(0, 1, 12)[:, None, None, None]
    phase = rng.normal(size=(1, 1, 40, 52)).cumsum(2).cumsum(3) * 0.01
    cube = np.concatenate([np.sin(2 * np.pi * t + phase), np.cos(2 * np.pi * t + phase)], axis=1)
    cube = cube + 0.02 * rng.normal(size=cube.shape)
    cases["time_series_cube"] = (cube.reshape(24, 40, 52).astype(np.float32), 50, 0.5)
    # compactness 0: pure feature-space growth
    cases["compactness_zero"] = (rng.normal(size=(2, 33, 29)).astype(np.float32), 12, 0.0)
    # single channel, strong compactness, two segments (one seed would crash
    # the reference: its pop() never removes a heap's last node, so the first
    # pop reads an uninitialised pixel index)
    cases["single_band_two_seeds"] = (rng.normal(size=(1, 21, 25)).astype(np.float32), 2, 40.0)
    return cases


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workdir:
        ref = build_reference(workdir)
        out = {}
        for name, (img, n_segments, compactness) in make_cases().items():
            labels, numlabels = run_reference(ref, img, n_segments, compactness)
            seeds = reference_seeds(ref, img.shape[1], img.shape[2], n_segments)
            assert labels.min() >= 0, f"{name}: reference left unlabelled pixels"
            assert len(seeds) == numlabels
            out[f"{name}__image"] = img
            out[f"{name}__seeds"] = seeds
            out[f"{name}__compactness"] = np.float64(compactness)
            out[f"{name}__labels"] = labels.astype(np.int16)
            print(f"{name}: {img.shape}, {numlabels} segments")
        np.savez_compressed(os.path.join(HERE, "snic_reference_parity.npz"), **out)


if __name__ == "__main__":
    main()
