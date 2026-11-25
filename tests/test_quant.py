#!/usr/bin/env python3
"""Compare gfxi's weight decoders with the gguf Python package, bit for bit.

usage: test_quant.py <gfxi> [--raw <type-id>:<blocks.bin> ...] <model.gguf> [...]

For every weight encoding found in the given models, one tensor is decoded by both
implementations. `--raw` checks an encoding from a file of raw block data, for encodings
that only occur in models too large to keep around as test fixtures.
"""
import os, subprocess, sys, tempfile
import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT

def compare(name, ours, ref):
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    if ours.shape != ref.shape:
        return f"FAIL {name}: {ours.size} elements, reference has {ref.size}"
    if not np.array_equal(ours, ref):
        bad = np.flatnonzero(ours != ref)
        return f"FAIL {name}: {bad.size}/{ours.size} differ, max abs diff {np.abs(ours - ref).max():.3g} (first at {bad[0]})"
    return f"ok   {name}: {ours.size} elements identical"

def main():
    exe, args = sys.argv[1], sys.argv[2:]
    failed, seen = 0, set()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.f32")
        while args:
            a = args.pop(0)
            if a == "--raw":
                tid, path = args.pop(0).split(":", 1)
                try:
                    qt = QT(int(tid))
                except ValueError:
                    print(f"skip type id {tid}: unknown to the installed gguf package"); continue
                raw = np.fromfile(path, dtype=np.uint8)
                raw = raw[: raw.size - raw.size % gguf.GGML_QUANT_SIZES[qt][1]]
                try:
                    ref = gguf.quants.dequantize(raw, qt)
                except NotImplementedError:
                    print(f"skip {qt.name}: the gguf package has no reference decoder"); continue
                subprocess.run([exe, "dequant-raw", tid, path, out], check=True)
                msg = compare(f"{qt.name} (raw sample)", np.fromfile(out, dtype=np.float32), ref)
            else:
                reader = gguf.GGUFReader(a)
                msg = None
                for t in reader.tensors:
                    qt = t.tensor_type
                    if qt in seen or qt.name.startswith("I") and not qt.name.startswith("IQ"):
                        continue
                    seen.add(qt)
                    try:
                        ref = np.asarray(t.data) if qt == QT.F32 else gguf.quants.dequantize(np.asarray(t.data), qt)
                    except NotImplementedError:
                        print(f"skip {qt.name}: the gguf package has no reference decoder"); continue
                    subprocess.run([exe, "dequant", a, t.name, out], check=True)
                    m = compare(f"{qt.name} ({os.path.basename(a)}:{t.name})", np.fromfile(out, dtype=np.float32), ref)
                    print(m); failed += m.startswith("FAIL")
                continue
            print(msg); failed += msg.startswith("FAIL")
    print("FAILED" if failed else "all decoders match")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
