# Two-channel geometric block

On heterodimer inverse folding, splitting the block by chain made recovery worse. Attention did not help either.

The synthetic tasks still stand, and they point the other way. A gated vector sum cut held-out MSE from 0.731 ± 0.018 to 0.122 ± 0.006. A neighbor count did not. On a label that is the product of a same-chain sum and a cross-chain sum, the split cut MSE from 2.570 ± 0.034 to 0.301 ± 0.011. That label matches the split. Amino-acid recovery does not.

The definition is in [hypothesis.md](hypothesis.md). The decision rule and the reading of both results are in [experiments.md](experiments.md).

## Inverse folding

Twenty PDB heterodimers. Each chain is 40–160 residues, resolution is at most 2.5 Å, and each complex has at least eight cross-chain CA contacts within 8 Å. Homologs were clustered at 30% identity on the shorter chain, and whole clusters stayed on one side of the split. Fifteen complexes trained. Five were held out: `7OWD`, `7S6O`, `7MIC`, `7QGS`, `7LVS`. That is 830 test residues, 253 of them interface.

Features are coordinates, chain id, and within-chain index. The amino acid is the label. `test_inverse.py` checks that shuffling the labels leaves the logits unchanged. The block cutoff is 10 Å. Matched budget: 200 steps, three seeds. Variant D also ran to 400. `±` is the sample standard deviation across seeds, with the n−1 denominator. Record: `results/inverse.json`. A uniform guess scores 0.05 recovery and NLL ln(20) ≈ 2.996.

| Variant | Recovery | Interface | Non-interface | NLL |
|---|---|---|---|---|
| A geometric softmax | 0.111 ± 0.007 | 0.153 ± 0.016 | 0.092 ± 0.006 | 2.912 ± 0.007 |
| C gated sum | 0.124 ± 0.005 | 0.158 ± 0.014 | 0.109 ± 0.005 | 2.889 ± 0.007 |
| S chain-split gated sum | 0.100 ± 0.015 | 0.104 ± 0.020 | 0.098 ± 0.013 | 2.911 ± 0.005 |
| D hybrid | 0.123 ± 0.011 | 0.162 ± 0.017 | 0.106 ± 0.016 | 2.890 ± 0.011 |
| E chain-split hybrid | 0.102 ± 0.008 | 0.107 ± 0.029 | 0.101 ± 0.011 | 2.922 ± 0.011 |

C beats S, and D beats E, on whole-complex recovery and on interface recovery. The gap is largest at the interface. C and D overlap on every number, so the attention branch did not earn a place. C is barely ahead of A on whole-complex recovery. Their interface ranges overlap.

S and E both reached interface recovery near 0.19 at step 50, then fell. At step 400, D was worse than at step 200 (recovery 0.102, interface 0.149). Extra steps do not explain the split's loss. Fifteen training complexes are not a general claim about proteins. They are enough to say this split did not transfer. This is not a SolubleMPNN comparison.

## Synthetic cardinality

Stage 1. Held-out mean squared error, three seeds, 1200 steps. A predictor that always says zero scores 4.299. Lower is better.

| Variant | What it tests | MSE |
|---|---|---|
| A | Geometric softmax | 0.731 ± 0.018 |
| B | Softmax plus an ungated neighbor count | 0.720 ± 0.009 |
| R | Softmax plus a gated count, no vector sum | 0.694 ± 0.018 |
| D | Softmax plus gated vector sum | 0.149 ± 0.016 |
| C | Gated vector sum only | 0.122 ± 0.006 |

The vector sum is the part that works. Putting softmax beside it does not improve on the sum alone. Multiplying a uniform softmax average by the neighbor count recovers an ordinary sum. It does not, in general, recover a query-selected sum. B failed anyway.

## Synthetic chain split

The label is the product of a same-chain sum and a cross-chain sum. Chain identity is already a pair feature, so the unsplit models are allowed to use it. Label variance is 4.876.

| Variant | MSE |
|---|---|
| C, chains mixed in one sum | 2.570 ± 0.034 |
| D, chains mixed | 1.455 ± 0.930 |
| E, channels split by chain | 0.301 ± 0.011 |

E won that task. An unsplit vector sum can still represent the two chain sums on separate coordinates. The gap shows easier learning under that budget, not that the unsplit block is unable to represent the product. D's seeds were 0.76, 1.10, and 2.51, and two were still falling at step 1200.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_block.py
python test_inverse.py
python run_stage1.py
python run_stage_e.py
python run_inverse.py
```

CPU is enough. Python 3.10 or newer. The numbers above used PyTorch 2.14.0. `run_inverse.py` downloads PDB files into `data/`, which is gitignored.

`test_block.py` checks behavior, not usefulness. Identical neighborhoods collapse under softmax. A distant node leaves the local channels unchanged. A rigid motion leaves the scalar output unchanged.

## Layout

| Path | Contents |
|---|---|
| `block.py` | Variants A–E, R, and S |
| `synthetic.py` | Synthetic tasks |
| `inverse_fold.py` | Heterodimer inverse-folding head |
| `proteins.py` | PDB fetch, homology clusters, split |
| `test_block.py` | Invariance and locality checks |
| `test_inverse.py` | Label leakage check |
| `run_stage1.py` | Cardinality ablation |
| `run_stage_e.py` | Synthetic chain-split ablation |
| `run_inverse.py` | Heterodimer screen |
| `results/` | Metrics for each seed |

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
