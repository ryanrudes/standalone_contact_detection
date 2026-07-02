# Rung 0 — the naive starting point

`main.py` is the original single-file chi-squared detector that [THEORY.md](../THEORY.md)
§0 opens with and then dismantles: the "close and not moving" idea, plus the hand-tuned
cleanup heuristics the derivation replaces one by one. It is kept runnable, with its own
tests (`test_main.py`, collected by the main suite), purely for contrast with what the
theory builds — it is *not* the current method. The real pipeline lives in
[`contact/`](../contact/) (the estimator) and [`oracle/`](../oracle/) (truth + scoring).

```bash
uv run python rung0/main.py        # run the toy end to end (writes contact_timeline.png here)
uv run pytest rung0                # its tests only
```
