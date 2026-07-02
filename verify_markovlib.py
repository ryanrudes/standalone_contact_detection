"""Verify markovlib's engines reproduce ``contact/hmm.py`` and ``contact/hsmm.py`` bit-for-bit.

markovlib (the vendored general Markov-process library under ``markovlib/``) is meant to back the same
inference the contact engines implement. This standalone harness runs both on identical random inputs and
asserts agreement (the ``verify_standalone.py`` discipline):

* **HMM** — ``forward_backward`` posterior + log-likelihood and the ``viterbi`` path, homogeneous ``(S,S)``
  and time-varying ``(T,S,S)`` transitions, to ~1e-9.
* **HSMM** — the right-censored explicit-duration ``hsmm_viterbi`` MAP path, exactly.

The contact modules are loaded directly under a stub ``contact`` package (they are pure numpy/scipy, and
``hsmm.py`` only needs ``from .hmm import logsumexp``), so contact's heavy ``__init__`` never runs and
neither package imports the other — markovlib stays contact-free.

Run (uses markovlib's light env, not contact's heavy MuJoCo one):

    uv run --project markovlib python verify_markovlib.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np

import markovlib as mk
from markovlib.duration import NegBinomDuration

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_contact_modules():
    """Load ``contact.hmm`` then ``contact.hsmm`` under a stub package (so the relative import resolves)."""
    if "contact" not in sys.modules:
        package = types.ModuleType("contact")
        package.__path__ = [os.path.join(HERE, "contact")]  # type: ignore[attr-defined]
        sys.modules["contact"] = package
    loaded = {}
    for name in ("hmm", "hsmm"):
        full = f"contact.{name}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(HERE, "contact", f"{name}.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["hmm"], loaded["hsmm"]


def _check_hmm(hmm, rng: np.random.Generator) -> tuple[int, int, float, float]:
    n = fails = 0
    worst_gamma = worst_ll = 0.0
    for time_varying in (False, True):
        for _ in range(500):
            n_states = int(rng.integers(2, 6))
            n_steps = int(rng.integers(2, 30))
            log_init = np.log(rng.dirichlet(np.ones(n_states)))
            if time_varying:
                rows = [np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]) for _ in range(n_steps)]
                log_trans = np.log(np.stack(rows))  # contact (T, S, S)
            else:
                log_trans = np.log(np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]))
            log_em = np.log(rng.uniform(0.05, 1.0, size=(n_steps, n_states)))

            gamma_c, ll_c = hmm.forward_backward(log_em, log_trans, log_init)
            path_c = hmm.viterbi(log_em, log_trans, log_init)
            mk_trans = log_trans if log_trans.ndim == 2 else log_trans[: n_steps - 1]  # markovlib (T-1, S, S)
            model = mk.DiscreteChain(log_init, mk_trans)
            result = mk.smooth(model, log_em)
            path_m = mk.decode(model, log_em)

            d_gamma = float(np.max(np.abs(gamma_c - result.gamma)))
            d_ll = abs(float(ll_c) - result.loglik)
            worst_gamma, worst_ll = max(worst_gamma, d_gamma), max(worst_ll, d_ll)
            n += 1
            if not (d_gamma <= 1e-9 and d_ll <= 1e-9 and np.array_equal(path_c, path_m)):
                fails += 1
    return n, fails, worst_gamma, worst_ll


def _check_hsmm(hsmm, rng: np.random.Generator) -> tuple[int, int]:
    n = fails = 0
    for _ in range(1000):
        n_states = int(rng.integers(2, 6))
        n_steps = int(rng.integers(2, 30))
        max_dur = int(rng.integers(2, 9))  # small enough that censoring is active
        concentration = float(rng.uniform(0.5, 3.0))
        means = rng.uniform(1.5, 5.0, size=n_states)
        log_init = np.log(rng.dirichlet(np.ones(n_states)))
        log_trans = np.log(np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]))
        log_em = np.log(rng.uniform(0.05, 1.0, size=(n_steps, n_states)))

        path_c = hsmm.hsmm_viterbi(log_em, log_trans, log_init, means, concentration, max_dur)
        gamma_c, ll_c = hsmm.hsmm_posteriors(log_em, log_trans, log_init, means, concentration, max_dur)
        durations = tuple(NegBinomDuration(float(m), concentration) for m in means)
        model = mk.SemiMarkovChain(log_init, log_trans, durations, max_dur)
        path_m = mk.decode(model, log_em)
        post = mk.smooth(model, log_em)
        n += 1
        path_ok = np.array_equal(path_c, path_m)
        post_ok = float(np.max(np.abs(gamma_c - post.gamma))) <= 1e-9 and abs(float(ll_c) - post.loglik) <= 1e-9
        if not (path_ok and post_ok):
            fails += 1
            if fails <= 5:
                print(f"  HSMM FAIL S={n_states} T={n_steps} max_dur={max_dur}: path_ok={path_ok} post_ok={post_ok}")
    return n, fails


def main() -> int:
    hmm, hsmm = _load_contact_modules()
    rng = np.random.default_rng(0)

    n_hmm, f_hmm, worst_gamma, worst_ll = _check_hmm(hmm, rng)
    print(
        f"{'PASS' if f_hmm == 0 else 'FAIL'}: HMM  {n_hmm - f_hmm}/{n_hmm} match contact/hmm.py bit-for-bit "
        f"(forward_backward gamma+loglik, viterbi path); worst |Δgamma|={worst_gamma:.2e}, |Δloglik|={worst_ll:.2e}."
    )

    n_hsmm, f_hsmm = _check_hsmm(hsmm, rng)
    print(
        f"{'PASS' if f_hsmm == 0 else 'FAIL'}: HSMM {n_hsmm - f_hsmm}/{n_hsmm} match contact/hsmm.py bit-for-bit "
        f"(explicit-duration viterbi path + per-frame posteriors)."
    )
    return 1 if (f_hmm or f_hsmm) else 0


if __name__ == "__main__":
    sys.exit(main())
