# Two-channel geometric block

Inside a three-block encoder, replacing the gated sum with a normalized mean plus log mass raised held-out negative log-likelihood by 0.012 nats. A 95% cluster bootstrap put that gap between 0.004 and 0.021. The same substitution in the one-block model was 0.029 nats, on a different locked split.

On one fixed one-block checkpoint, rebuilding the sum from the mean and log mass changed logits by at most 8.3e-7. The algebra matches the masking and fusion code. The training gap is not an implementation mismatch.

The fusion layer receives log(1+ρ), so the rebuilt sum is the mean times (1e-8 + exp(log(1+ρ)) − 1). The inputs determine the sum. A finite MLP has to learn that map, and separately trained gates need not match, because the aggregation changes the gradients. Directly supplying the unnormalized aggregate improves held-out likelihood here even when gate mass stays available. That does not separate easier optimization, activation scale, regularization, or generalization.

The likelihood advantage of the gated sum over geometric softmax remains the earlier result: 0.033 nats on 48 locked clusters, interval 0.027 to 0.039. This run did not repeat that comparison.

The definition is in [hypothesis.md](hypothesis.md). The decision rule and the reading of each result are in [experiments.md](experiments.md).

## Deeper encoder

Three blocks, residue width 64, message width 64, mixer hidden 128. Both aggregations have 203,034 parameters. The only change is the sum versus the mean with log(1+ρ) kept. 429 train, 21 validation, 30 test, from the same 480 heterodimers. Every earlier validation or test complex stayed out of this test. All 30 test clusters are singletons. Worst cross-partition identity was 0.277. Seeds were averaged before the cluster resample. The interval is the 2.5 and 97.5 percentiles of 2000 draws. It does not include retraining. Record: `results/deep.json`.

| Comparison | Effect on NLL | 95% cluster interval | Clusters |
|---|---|---|---|
| Mean-plus-mass − sum | +0.012 | [+0.004, +0.021] | mean-plus-mass worse on 21/30 |

Test NLL is 2.858 for the sum and 2.870 for the mean plus mass. Whole-complex recovery is 0.113 and 0.112; that recovery interval includes zero. Interface NLL is 2.872 and 2.895, a paired gap of +0.023 with interval [+0.004, +0.040], mean worse on 21/30. The sum’s three seeds checkpointed at step 350. Two mean-plus-mass seeds were still best at step 400.

## Algebraic check

The one-block magnitude run did not save weights, so one gated-sum seed was trained on that locked split, then left fixed. Replacing its sum by the mean and the explicit reconstruction above, on the 30 test complexes, moved no logit by more than 8.3e-7. Mean absolute difference 5.4e-8 across 128,600 logits. Tolerance was 1e-4. Record: `results/reconstruct.json`.

## Sum versus mean

480 PDB heterodimers, 196 clusters. The locked test is 30 complexes that were not in any earlier validation or test split. Each test cluster is one complex. Worst cross-partition identity was 0.293. C, N, and Q share the gate, the message vectors, and 22,278 parameters. N divides the sum by (1e-8 + ρ). Q does that and restores log(1+ρ), which C already had beside the sum. Record: `results/magnitude.json`.

| Comparison | Effect on NLL | 95% cluster interval | Clusters |
|---|---|---|---|
| Mean − sum | +0.036 | [+0.029, +0.042] | mean worse on 29/30 |
| Mean-plus-mass − sum | +0.029 | [+0.022, +0.036] | mean-plus-mass worse on 27/30 |
| Mean-plus-mass − mean | −0.007 | [−0.011, −0.003] | mass better on 23/30 |

Test NLL is 2.862 for the sum, 2.898 for the mean, and 2.891 for the mean plus mass. Whole-complex recovery is 0.107, 0.094, and 0.098. Interface NLL is 2.903, 2.933, and 2.932. The mass slot does not close the interface gap. By construction, the mean times (1e-8 + ρ) equals the sum. The log-mass form above is what the mixer actually sees.

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
python check_reconstruct.py
python run_deep.py
```

CPU is enough. Python 3.10 or newer. The numbers above used PyTorch 2.14.0. Inverse-folding scripts download PDB files into `data/`, which is gitignored. A locked split file is reused when it is already present. `check_reconstruct.py` needs the magnitude lock. `run_deep.py` needs the earlier locks so it can keep those complexes out of the test.

## Layout

| Path | Contents |
|---|---|
| `block.py` | Variants, including the normalized mean, the mean plus mass, and the reconstruction check |
| `inverse_fold.py` | One-block folder and the three-block stack |
| `run_deep.py` | Sum versus mean-plus-mass in the three-block encoder |
| `check_reconstruct.py` | Fixed-checkpoint logit comparison |
| `run_magnitude.py` | One-block sum versus mean on a fresh locked test |
| `run_confirm.py` | Locked gated-sum versus softmax confirmation |
| `results/` | Metrics for each seed, complex, and cluster |

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
