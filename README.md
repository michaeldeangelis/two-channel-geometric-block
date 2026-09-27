# Two-channel geometric block

On 14 heterodimer families that did not choose the architecture, a gated sum beat geometric softmax on negative log-likelihood for 13. The mean gap was 0.023 nats. Recovery did not move, and the interface did not either.

A chain split had already lost on an earlier screen. I paused it. Attention still matches the gated sum and does not improve it.

Most of the gap against a uniform guess was amino-acid frequency. Leucine is 0.097 of the training residues, and predicting it recovers 0.096 with no coordinates. Softmax was 0.004 nats better than that table. The gated sum was 0.027 nats better. Setting every coordinate to zero removed the gain.

The definition is in [hypothesis.md](hypothesis.md). The decision rule and the reading of each result are in [experiments.md](experiments.md).

## New families

72 PDB heterodimers, 46 homology clusters, split 44 / 14 / 14 into train, validation, and test. Whole clusters only. The five complexes from the first inverse-folding screen stayed in train. Worst cross-partition identity was 0.255, measured as the best ungapped window on the shorter chain, either chain against either chain. The checkpoint is the lowest unweighted mean of per-complex validation NLL. Test complexes were scored once. `±` in this table is the sample standard deviation across the 14 test complexes. Seed standard deviation is training variability and is in `results/scale.json`. Uniform NLL is ln(20) ≈ 2.996.

| Model | Test NLL | Recovery | Interface NLL |
|---|---|---|---|
| F training frequencies | 2.920 ± 0.045 | 0.096 ± 0.037 | 2.932 ± 0.077 |
| A geometric softmax | 2.916 ± 0.051 | 0.096 ± 0.024 | 2.947 ± 0.113 |
| C gated sum | 2.893 ± 0.038 | 0.100 ± 0.019 | 2.936 ± 0.124 |
| Z gated sum, coordinates removed | 2.932 ± 0.030 | 0.089 ± 0.031 | 2.952 ± 0.066 |
| D hybrid | 2.892 ± 0.040 | 0.100 ± 0.026 | — |

Paired C minus A, seeds averaged first: NLL −0.023 (13/14), recovery +0.004 (8/14), interface NLL −0.011 (9/14). The one NLL loss is `7T8I`. A has 23,561 parameters and C has 22,278, so the NLL edge is not a wider model. Z uses C's weights shape with coordinates set to zero. Chain id and sequence index remain. C beat Z on all 14 complexes.

Several seeds were best before the last step. On those runs validation NLL rose afterward, and on some of them training NLL was still falling. The table uses the validation checkpoint.

## First inverse-folding screen

Twenty heterodimers, 15 train and 5 test, 200 steps, no validation checkpoint. The chain split was worse, especially at the interface. Record: `results/inverse.json`. `±` here is across three seeds.

| Variant | Recovery | Interface | NLL |
|---|---|---|---|
| A geometric softmax | 0.111 ± 0.007 | 0.153 ± 0.016 | 2.912 ± 0.007 |
| C gated sum | 0.124 ± 0.005 | 0.158 ± 0.014 | 2.889 ± 0.007 |
| S chain-split gated sum | 0.100 ± 0.015 | 0.104 ± 0.020 | 2.911 ± 0.005 |
| D hybrid | 0.123 ± 0.011 | 0.162 ± 0.017 | 2.890 ± 0.011 |
| E chain-split hybrid | 0.102 ± 0.008 | 0.107 ± 0.029 | 2.922 ± 0.011 |

That screen chose the architecture, so those five test complexes are not the test set above. S and E are paused.

## Synthetic tasks

Stage 1. Held-out mean squared error, three seeds, 1200 steps. A predictor that always says zero scores 4.299. Lower is better. `±` is across seeds.

| Variant | What it tests | MSE |
|---|---|---|
| A | Geometric softmax | 0.731 ± 0.018 |
| B | Softmax plus an ungated neighbor count | 0.720 ± 0.009 |
| R | Softmax plus a gated count, no vector sum | 0.694 ± 0.018 |
| D | Softmax plus gated vector sum | 0.149 ± 0.016 |
| C | Gated vector sum only | 0.122 ± 0.006 |

The vector sum is the part that works. Putting softmax beside it does not improve on the sum alone. Multiplying a uniform softmax average by the neighbor count recovers an ordinary sum. It does not, in general, recover a query-selected sum. B failed anyway.

The chain-split label is the product of a same-chain sum and a cross-chain sum. Label variance is 4.876.

| Variant | MSE |
|---|---|
| C, chains mixed in one sum | 2.570 ± 0.034 |
| D, chains mixed | 1.455 ± 0.930 |
| E, channels split by chain | 0.301 ± 0.011 |

E won that task. An unsplit vector sum can still represent the two chain sums on separate coordinates. The gap shows easier learning under that budget. It did not transfer to inverse folding.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_block.py
python test_inverse.py
python test_scale.py
python run_stage1.py
python run_stage_e.py
python run_inverse.py
python run_scale.py
```

CPU is enough. Python 3.10 or newer. The numbers above used PyTorch 2.14.0. The inverse-folding scripts download PDB files into `data/`, which is gitignored.

`test_block.py` checks behavior, not usefulness. Identical neighborhoods collapse under softmax. A distant node leaves the local channels unchanged. A rigid motion leaves the scalar output unchanged. `test_inverse.py` checks that amino-acid labels do not change the logits. `test_scale.py` checks the homology audit, the frequency baseline, and the coordinate ablation.

## Layout

| Path | Contents |
|---|---|
| `block.py` | Variants A–E, R, and S |
| `synthetic.py` | Synthetic tasks |
| `inverse_fold.py` | Heterodimer head, frequency baseline, coordinate ablation |
| `proteins.py` | PDB fetch, homology clusters, partition audit |
| `test_block.py` | Invariance and locality checks |
| `test_inverse.py` | Label leakage check |
| `test_scale.py` | Audit, frequency baseline, coordinate ablation |
| `run_stage1.py` | Cardinality ablation |
| `run_stage_e.py` | Synthetic chain-split ablation |
| `run_inverse.py` | First heterodimer screen |
| `run_scale.py` | C versus A on new families |
| `results/` | Metrics for each seed and each complex |

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
