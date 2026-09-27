# Two-channel geometric block

A geometric residue block that keeps two facts separate: which neighbors matter, and how much interaction evidence is present.

On a synthetic task where the label is a signed sum over a query-selected neighborhood, the gated vector sum beat geometric softmax and geometric softmax plus a neighbor count. Splitting those channels into same-chain and cross-chain then beat the unsplit block when the label was the product of the two sums.

The block is defined in [hypothesis.md](hypothesis.md). The decision rule and the full write-up are in [experiments.md](experiments.md).

## Results

Stage 1. Held-out mean squared error, three seeds, 1200 steps. A zero predictor scores 4.299. Lower is better.

| Variant | What it tests | MSE |
|---|---|---|
| A | Geometric softmax | 0.731 ± 0.018 |
| B | Softmax plus an ungated neighbor count | 0.720 ± 0.009 |
| R | Softmax plus a gated count, no vector sum | 0.694 ± 0.018 |
| D | Softmax plus gated vector sum | 0.149 ± 0.016 |
| C | Gated vector sum only | 0.122 ± 0.006 |

An ungated count does not fix softmax. A gated scalar count does not either. The vector sum does. Adding softmax on top of that sum does not help.

Variant E. The label is the product of a same-chain sum and a cross-chain sum. Chain identity is already a pair feature, so the unsplit models are allowed to use it. Label variance is 4.876.

| Variant | MSE |
|---|---|
| C, chains mixed in one sum | 2.570 ± 0.034 |
| D, chains mixed | 1.455 ± 0.930 |
| E, channels split by chain | 0.301 ± 0.011 |

Seed-level curves are in `results/`. These are synthetic geometric tasks. Inverse folding has not been run.

## Question

If every neighbor value is the same vector, softmax returns that vector for two neighbors or for twenty. The first experiment asks whether an accumulated-evidence channel beats both geometric attention and geometric attention with a count feature. Beating softmax alone is not the claim. Cardinality-preserving attention and Principal Neighbourhood Aggregation already place a sum beside a mean.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_block.py
python run_stage1.py
python run_stage_e.py
```

CPU is enough. Python 3.10 or newer. The published numbers used PyTorch 2.14.0.

`test_block.py` checks the block's behavior: identical neighborhoods collapse under softmax, a distant node leaves the local channels unchanged, and a rigid motion leaves the scalar output unchanged.

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

Use the Cite this repository button, which reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
