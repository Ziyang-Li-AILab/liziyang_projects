#!/usr/bin/env python3
"""Convert the official MANO pickles to chumpy-free plain-numpy pickles.

The official MANO release (https://mano.is.tue.mpg.de, ``mano_v1_2/models/``)
stores ``shapedirs`` as a lazy chumpy expression and ``J_regressor`` as a scipy
sparse matrix, so unpickling those files normally requires ``chumpy`` — which
no longer imports on modern numpy. This one-off converter unpickles them
WITHOUT chumpy or scipy installed (stub classes capture the pickled state and
the arrays are reconstructed from it) and writes pickles that contain only
plain numpy arrays and builtins, which ``smplx`` then loads directly.

Usage (once, after downloading the official MANO release):

  python scripts/convert_mano_pkls.py --src /path/to/mano_v1_2/models \
      --dst data/mano/mano

Afterwards ``data/mano`` is the default MANO root the code uses (override with
``ACE_EGO_HAND_MANO_DIR``).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


class _Stub:
    """Placeholder that captures a pickled object's state for later resolution."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["__state__"] = state


class _ChStub(_Stub):
    """chumpy.ch.Ch: the wrapped array lives in the ``x`` attribute."""


class _SelectStub(_Stub):
    """chumpy.reordering.Select: value == a.ravel()[idxs].reshape(preferred_shape)."""


class _CscStub(_Stub):
    """scipy.sparse csc_matrix: state == {_shape, data, indices, indptr, ...}."""


class _StubUnpickler(pickle.Unpickler):
    """Route chumpy / scipy.sparse classes to stubs; everything else is normal."""

    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _SelectStub if name == "Select" else _ChStub
        if module.startswith("scipy.sparse"):
            return _CscStub
        return super().find_class(module, name)


def _resolve(v):
    """Stub -> plain numpy array (recursing into nested chumpy expressions)."""
    if isinstance(v, _SelectStub):
        a = np.asarray(_resolve(v.a))
        out = a.ravel()[np.asarray(v.idxs)]
        pref = getattr(v, "preferred_shape", None)
        return out.reshape(pref) if pref is not None else out
    if isinstance(v, _ChStub):
        return np.asarray(_resolve(v.x))
    if isinstance(v, _CscStub):
        n_row, n_col = v._shape
        data = np.asarray(v.data)
        indices = np.asarray(v.indices)
        indptr = np.asarray(v.indptr)
        dense = np.zeros((n_row, n_col), dtype=data.dtype)
        for col in range(n_col):
            lo, hi = indptr[col], indptr[col + 1]
            dense[indices[lo:hi], col] = data[lo:hi]
        return dense
    return v


def convert(src: Path, dst: Path) -> None:
    with open(src, "rb") as f:
        d = _StubUnpickler(f, encoding="latin1").load()
    out = {k: _resolve(v) for k, v in d.items()}
    left = {k: type(v).__name__ for k, v in out.items() if isinstance(v, _Stub)}
    if left:
        raise RuntimeError(f"{src.name}: unresolved pickled objects: {left}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump(out, f, protocol=2)
    fields = ", ".join(
        f"{k}{list(v.shape)}" for k, v in sorted(out.items())
        if isinstance(v, np.ndarray))
    print(f"{src} -> {dst}\n  {fields}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True,
                    help="dir with the official MANO_LEFT.pkl / MANO_RIGHT.pkl "
                         "(e.g. mano_v1_2/models)")
    ap.add_argument("--dst", default="data/mano/mano",
                    help="output dir (default: data/mano/mano, the dir "
                         "smplx.create reads with the default MANO root)")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    n = 0
    for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
        if (src / name).exists():
            convert(src / name, dst / name)
            n += 1
        else:
            print(f"! {src / name} not found, skipped")
    if n == 0:
        print("nothing converted — point --src at the dir holding the MANO pkls")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
