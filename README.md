# Two-channel geometric block

On a locked test of 48 homology clusters, the gated sum beat geometric softmax by 0.033 nats of whole-complex negative log-likelihood. A cluster bootstrap put that gap between 0.027 and 0.039 nats. That is about 3.2% lower perplexity. Recovery moved by half a percentage point.

The 14 complexes that originally favored the gated sum were not in this test. They stayed in the training record. Attention was not retrained. The chain split stays paused.

The definition is in [hypothesis.md](hypothesis.md). The decision rule and the reading of each result are in [experiments.md](experiments.md).

## Locked test

240 PDB heterodimers, 106 clusters. Split 142 / 50 / 48 into train, validation, and test, written down before training. Every test cluster is one complex. Worst cross-partition identity was 0.275. Whole-complex NLL is the primary endpoint. The interval resamples clusters 2000 times and reports the 2.5 and 97.5 percentiles. It is not the standard deviation across complexes. Record: `results/confirm.json`.

Widths match the development run: softmax hidden 104 (23,561 parameters), gated sum hidden 80 (22,278). Three seeds. The checkpoint is the lowest validation NLL.

| Comparison | Effect | Bootstrap interval | Clusters favoring the gated sum |
|---|---|---|---|
| Gated sum − softmax, NLL | −0.033 | [−0.039, −0.027] | 45/48 |
| Gated sum − training frequencies, NLL | −0.048 | [−0.055, −0.041] | 45/48 |
| Gated sum − softmax, recovery | +0.006 | [+0.001, +0.010] | 30/48 |
| Gated sum − softmax, interface NLL | −0.030 | [−0.041, −0.019] | 38/48 |

Test NLL is 2.846 for the gated sum, 2.878 for softmax, and 2.894 for the training-set amino-acid frequencies. Softmax itself beats that frequency table by 0.015 nats, interval [−0.020, −0.011]. The gated sum's gap over softmax is additional. Recovery is 0.118 against 0.112, with the frequency baseline at 0.105. Interface NLL also favors the gated sum on this set. The smaller development set did not resolve the interface.

A 0.033 nat drop can raise the probability of the correct residue without changing the top prediction. That is why recovery barely moves.

## Development screen

14 complexes, after a 44 / 14 / 14 split. The gated sum was better on NLL for 13 of 14, by 0.023 nats. The across-complex scatter of 0.027 was descriptive, not a confidence interval. Recovery and interface NLL were not resolved there. Beating the frequency table showed information beyond amino-acid composition. It did not, by itself, show that the information was geometric.

The coordinate ablation zeros positions before the cutoff. Distances become 0 and every residue becomes a neighbor of every other residue, so the 10 Å graph is replaced by a complete graph. Sequence separation and chain identity remain. That run is not a proof that the ablated model saw no geometry. It was not repeated here.

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

The chain-split label is the product of a same-chain sum and a cross-chain sum. Label variance is 4.876. The split won that task (MSE 0.301 ± 0.011 against 2.570 ± 0.034). An unsplit sum can still represent the two chain sums on separate coordinates. The gap did not transfer to inverse folding.

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
python run_confirm.py
```

CPU is enough. Python 3.10 or newer. The numbers above used PyTorch 2.14.0. The inverse-folding scripts download PDB files into `data/`, which is gitignored. `run_confirm.py` reuses `data/split_confirm.json` if it is already there.

## Layout

| Path | Contents |
|---|---|
| `block.py` | Variants A–E, R, and S |
| `synthetic.py` | Synthetic tasks |
| `inverse_fold.py` | Heterodimer head, frequency baseline, coordinate ablation |
| `proteins.py` | PDB fetch, homology clusters, partition audit |
| `run_confirm.py` | Locked C versus A confirmation |
| `run_scale.py` | Development screen on 14 complexes |
| `results/` | Metrics for each seed, complex, and cluster |

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
