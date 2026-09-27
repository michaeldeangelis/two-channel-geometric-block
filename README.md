# Two-channel geometric block

Softmax returns the same vector whether two neighbors carry it or twenty. I added a second channel that sums the evidence instead of averaging it. I kept the channel because it beat geometric softmax and geometric softmax plus a neighbor count. A count by itself did not.

I then split those channels into same-chain and cross-chain. On a task whose label is the product of the two sums, the split won.

The definition is in [hypothesis.md](hypothesis.md). The decision rule is in [experiments.md](experiments.md).

## Results

Stage 1. Held-out mean squared error, three seeds, 1200 steps. A predictor that always says zero scores 4.299. Lower is better.

| Variant | What it tests | MSE |
|---|---|---|
| A | Geometric softmax | 0.731 ± 0.018 |
| B | Softmax plus an ungated neighbor count | 0.720 ± 0.009 |
| R | Softmax plus a gated count, no vector sum | 0.694 ± 0.018 |
| D | Softmax plus gated vector sum | 0.149 ± 0.016 |
| C | Gated vector sum only | 0.122 ± 0.006 |

The vector sum is the part that works. Putting softmax beside it does not improve on the sum alone.

Variant E. The label is the product of a same-chain sum and a cross-chain sum. Chain identity is already a pair feature, so the unsplit models are allowed to use it. Label variance is 4.876.

| Variant | MSE |
|---|---|
| C, chains mixed in one sum | 2.570 ± 0.034 |
| D, chains mixed | 1.455 ± 0.930 |
| E, channels split by chain | 0.301 ± 0.011 |

D was uneven across seeds, and two seeds were still falling at the step limit. C had already flattened well above E. Seed curves are in `results/`.

These are synthetic geometric tasks. I have not trained this block for inverse folding.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_block.py
python run_stage1.py
python run_stage_e.py
```

CPU is enough. Python 3.10 or newer. The numbers above used PyTorch 2.14.0.

`test_block.py` checks behavior, not usefulness. Identical neighborhoods collapse under softmax. A distant node leaves the local channels unchanged. A rigid motion leaves the scalar output unchanged.

## Layout

| Path | Contents |
|---|---|
| `block.py` | Variants A–E |
| `synthetic.py` | Tasks |
| `test_block.py` | Invariance and locality checks |
| `run_stage1.py` | Cardinality ablation |
| `run_stage_e.py` | Chain-split ablation |
| `results/` | Metrics for each seed |

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
