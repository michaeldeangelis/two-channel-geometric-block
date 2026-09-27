# Two-channel geometric block

Dividing the gated sum by its gate mass raised whole-complex negative log-likelihood by 0.036 nats. Putting the mass back recovered 0.007 of that. The mean plus the mass stayed 0.029 nats worse than the sum. A 95% cluster bootstrap put both gaps above zero.

The mean and the mass determine the sum, and the block's mixer is an MLP, so the product is representable. The network did not use it as well as the sum it was handed. That is easier optimization, not missing information.

The likelihood advantage of the gated sum over geometric softmax remains the earlier result: 0.033 nats on 48 locked clusters, interval 0.027 to 0.039. This run did not repeat that comparison.

The definition is in [hypothesis.md](hypothesis.md). The decision rule and the reading of each result are in [experiments.md](experiments.md).

## Sum versus mean

480 PDB heterodimers, 196 clusters. The locked test is 30 complexes that were not in any earlier validation or test split. Each test cluster is one complex. Worst cross-partition identity was 0.293. C, N, and Q share the gate, the message vectors, and 22,278 parameters. N divides the sum by (1e-8 + ρ). Q does that and restores log(1+ρ), which C already had beside the sum. Seeds were averaged before the cluster resample. The interval is the 2.5 and 97.5 percentiles of 2000 draws. It does not include retraining. Record: `results/magnitude.json`.

| Comparison | Effect on NLL | 95% cluster interval | Clusters |
|---|---|---|---|
| Mean − sum | +0.036 | [+0.029, +0.042] | mean worse on 29/30 |
| Mean-plus-mass − sum | +0.029 | [+0.022, +0.036] | mean-plus-mass worse on 27/30 |
| Mean-plus-mass − mean | −0.007 | [−0.011, −0.003] | mass better on 23/30 |

Test NLL is 2.862 for the sum, 2.898 for the mean, and 2.891 for the mean plus mass. Whole-complex recovery is 0.107, 0.094, and 0.098. Interface NLL is 2.903, 2.933, and 2.932. The mass slot does not close the interface gap.

By construction, the mean times (1e-8 + ρ) equals the sum.

## Locked softmax comparison

On 48 homology clusters held out of training, the gated sum beat geometric softmax by 0.033 nats. The same style of cluster interval was 0.027 to 0.039. The gated sum also beat training-set amino-acid frequencies by 0.048 nats. Recovery moved by half a percentage point. Interface NLL favored the gated sum on that set; an earlier 14-complex screen had been too noisy to say so.

Lower NLL means a higher geometric mean of the probabilities on the correct residues. It does not mean every correct residue received a higher probability, and it does not by itself explain the recovery change.

The coordinate ablation zeros positions before the cutoff. Distances become 0 and the 10 Å neighborhood becomes a complete graph, so it mixes geometry with connectivity. It was not used in this comparison.

## Synthetic tasks

A gated vector sum cut held-out MSE from 0.731 ± 0.018 to 0.122 ± 0.006 against geometric softmax. A neighbor count did not. On a label that is the product of a same-chain sum and a cross-chain sum, an explicit chain split cut MSE from 2.570 ± 0.034 to 0.301 ± 0.011. That split lost on inverse folding and stays paused. `±` there is across three seeds. Tables are in [experiments.md](experiments.md).

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
python run_magnitude.py
```

CPU is enough. Python 3.10 or newer. The numbers above used PyTorch 2.14.0. Inverse-folding scripts download PDB files into `data/`, which is gitignored. A locked split file is reused when it is already present.

## Layout

| Path | Contents |
|---|---|
| `block.py` | Variants, including the normalized mean N and the mean plus mass Q |
| `run_magnitude.py` | Sum versus mean on a fresh locked test |
| `run_confirm.py` | Locked gated-sum versus softmax confirmation |
| `results/` | Metrics for each seed, complex, and cluster |

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
