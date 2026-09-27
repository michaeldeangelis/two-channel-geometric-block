# Two-channel geometric block

A geometric residue block with two local channels: geometry-biased softmax, and an unnormalized gated sum that keeps how much interaction evidence is present.

Private research snapshot. No license is granted.

## Question

If every neighbor value is the same vector, softmax aggregation returns that vector whether the neighborhood has two neighbors or twenty. The experiment asks whether an explicit accumulated-evidence channel beats both geometric softmax and geometric softmax plus a neighbor count.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_block.py
python run_stage1.py
python run_stage_e.py
```

CPU is enough. Python 3.10 or newer. The runs that produced `results/` used PyTorch 2.14.0, three seeds, and 1200 steps.

## Results

Details and the decision rule are in [experiments.md](experiments.md). The block definition is in [hypothesis.md](hypothesis.md).

Stage 1 passed. The gated vector sum beat geometric softmax and softmax plus an ungated count. A gated count without the vector sum did not. Softmax did not improve on the gated sum alone.

Variant E passed on a second task. Splitting the same channels into same-chain and cross-chain beat the unsplit block when the label was the product of those two sums.

## Layout

| Path | Contents |
|---|---|
| `block.py` | Variants A–E |
| `synthetic.py` | Synthetic tasks |
| `test_block.py` | Invariance and locality checks |
| `run_stage1.py` | Cardinality ablation |
| `run_stage_e.py` | Chain-split ablation |
| `results/` | Seed-level metrics |

## Cite

GitHub reads `CITATION.cff`.
