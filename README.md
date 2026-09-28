# Two-channel geometric block

## Abstract

I trained a small network to name each amino acid in a two-chain protein from backbone geometry, with the sequence itself hidden. Describing each neighbor with a local backbone frame, rather than a single distance, beat a guess based only on how common each amino acid is. On 24 separate protein families that gain was 0.395 nats: about 48% higher geometric-mean probability of the correct residue, and about 33% lower perplexity, on every family. Adding the neighbor messages up did not separate from averaging them while keeping their total weight. That gap was 0.009 nats, and the uncertainty range includes no difference. I kept the sum because it was better in earlier, weaker versions of this block. I did not show that the frames erased that earlier advantage. A chain split and a hybrid that also used attention were tried and dropped. On those same 24 families I compared this model with ProteinMPNN. Both named residues from the backbone, and neither was given the native sequence. ProteinMPNN got 0.476 of the residues right. This model got 0.231. The difference was 0.245, with a range from 0.221 to 0.265, and ProteinMPNN was ahead on every family. The test structures match, and neither model was given the native sequence. The training data, the model size, and the chance of seeing a related protein do not match, so I do not pin that gap on the architecture. The coordinate-only guess recovered slightly more native residues than sequences drawn at temperature 0.1. Naming the native residue is a different question from whether a new sequence would work in an experiment. This phase is closed.

On 24 new singleton families, the gated sum and the mean plus log mass are still not separated. The gap is 0.009 nats. A 95% cluster interval is −0.001 to +0.018, and the mean plus mass is worse on 17 of 24. Every test cluster has one complex, so the cluster-weighted mean and the complex-weighted mean are the same number. The sum stays the working aggregation because of the earlier results. This run does not show that it won.

The same frame-edge model beats training-set frequencies by 0.395 nats on those families, interval 0.361 to 0.432, better on 24 of 24. That raises the geometric mean of the correct-residue probabilities by exp(0.395) ≈ 1.484, about 48.4% higher. Perplexity falls by exp(−0.395) ≈ 0.674, about 32.6% lower. Those two percentages are not the same. The budget was 800 steps, set from the earlier validation curves before this test was scored. One sum seed and two mean-plus-mass seeds are still best at step 800. The run stops. Record: `results/families.json`.

On the same 24 families, vanilla ProteinMPNN (`v_48_020`) recovers 0.476 of native residues from backbone coordinates alone. The frame model recovers 0.231. ProteinMPNN minus the frame model is +0.245, interval +0.221 to +0.265, ProteinMPNN ahead on 24 of 24. The frame recovery was recomputed with the locked training rule, and the recomputed NLL matched the stored run with drift 0. Same structures and no native-sequence input make this a matched evaluation. Training data, capacity, and possible homolog exposure differ, so the gap is not attributed to the architecture. Eight ProteinMPNN sequences at temperature 0.1 recovered 0.466. The coordinate-only argmax recovered slightly more native residues. Native recovery is a different endpoint from the quality of a generated sequence. SolubleMPNN was not run. This is not a SolubleMPNN result. This phase is closed. Record: `results/reference.json`.

The definition is in [hypothesis.md](hypothesis.md). The decision rule and the reading of each result are in [experiments.md](experiments.md).

## Twenty-four families

Released 2022-06-02 through 2026-09-01, same heterodimer filters, resolution at most 2.5 Å. Forty-three new clusters qualified. The test is the smallest 24, each a single complex. Train 510. The checkpoint set is the previous frame-model validation, 89 complexes after new homologs of that set were kept with it. Worst cross-partition identity 0.279. Depth stays 3. Both models use the frame edges. Record: `results/families.json`.

| Comparison | Effect on NLL | 95% interval | Complexes |
|---|---|---|---|
| Mean-plus-mass − sum | +0.009 | [−0.001, +0.018] | mean-plus-mass worse on 17/24 |
| Sum − frequencies | −0.395 | [−0.432, −0.361] | sum better on 24/24 |

## Frame edges

Three blocks, the same depth as the previous encoder. Each edge keeps the 10 Å CA window and adds a 16-bin distance basis, the neighbor CA in the residue’s N-CA-C frame, and the relative rotation of the two frames. Sequence separation and chain identity stay. P uses the gated sum on that same window with the geometric channels set to zero. It does not turn the window into a complete graph. F is the training-set amino-acid frequency. C, Q, and P have 223,770 parameters each. 366 train, 73 validation, 39 test. The test clusters have sizes 3, 3, 4, 4, 6, 7, and 12, and each cluster counts equally. Worst cross-partition identity was 0.279. Every sum seed and every mean-plus-mass seed selected step 400. The run stops there. Record: `results/strong.json`.

| Comparison | Effect on NLL | 95% cluster interval | Complexes |
|---|---|---|---|
| Sum − frequencies | −0.280 | [−0.327, −0.228] | sum better on 39/39 |
| Sum − frame channels off | −0.218 | [−0.261, −0.169] | sum better on 39/39 |
| Mean-plus-mass − sum | +0.006 | [−0.011, +0.030] | mean-plus-mass worse on 15/39 |

Test NLL is 2.633 for the sum and 2.634 for the mean plus mass when each complex has one vote. That paired difference is 0.001 nat. The +0.006 nat in the table gives each of the seven clusters one vote. Under one weighting, the average of the paired differences equals the difference of the averages. The 0.280 nat frequency gap raises correct-residue geometric-mean probability by exp(0.280) ≈ 1.323, about 32.3% higher, and lowers perplexity by exp(−0.280) ≈ 0.756, about 24.4%. Recovery is 0.176 for the sum and 0.073 for frequencies, interval 0.083 to 0.133 on the difference. Recovery and interface NLL do not separate the sum from the mean plus mass. The aggregation interval has room for a modest advantage in either direction. Seven clusters do not show that frame features removed the earlier advantage. The sum stays the working aggregation on the earlier evidence. Every sum seed and every mean-plus-mass seed selected step 400, so that result is a fixed budget and is not reopened.

## Closed series

Doubling the earlier three-block budget to 800 steps left a 0.013 nat gap, interval 0.004 to 0.022, on one already examined set of 30 complexes. On a fixed one-block checkpoint, rebuilding the sum from the mean and log mass changed logits by at most 8.3e-7. That is equivalence at fixed weights. The fusion layer receives log(1+ρ), so the rebuilt sum is the mean times (1e-8 + exp(log(1+ρ)) − 1). Directly supplying the unnormalized aggregate improved held-out likelihood in that weaker encoder even when gate mass stayed available. The 0.029 nat one-block gap and the 0.012 nat three-block gap use different complexes, so that difference is not evidence that depth reduced the effect.

The likelihood advantage of the gated sum over geometric softmax remains the earlier result: 0.033 nats on 48 locked clusters, interval 0.027 to 0.039. This run did not repeat that comparison.

## Eight hundred steps

Same architecture, same lock, same seeds. The validation curve through step 400 matches the earlier run exactly. Checkpointing is still the lowest validation NLL, and the test is not used to stop. This is a follow-up on an already examined test. Record: `results/deep_budget.json`.

| Comparison | Effect on NLL | 95% cluster interval | Clusters |
|---|---|---|---|
| Mean-plus-mass − sum | +0.013 | [+0.004, +0.022] | mean-plus-mass worse on 22/30 |

Test NLL is 2.836 for the sum and 2.850 for the mean plus mass. Interface NLL differs by +0.020, interval 0.005 to 0.036, mean worse on 17 of 30. Recovery’s interval includes zero. Selected steps are 800, 450, and 750 for the sum, and 800, 750, and 750 for the mean plus mass.

## Deeper encoder

Three blocks, residue width 64, message width 64, mixer hidden 128. Both aggregations have 203,034 parameters. The only change is the sum versus the mean with log(1+ρ) kept. 429 train, 21 validation, 30 test, from the same 480 heterodimers. Every earlier validation or test complex stayed out of this test. All 30 test clusters are singletons. Worst cross-partition identity was 0.277. Seeds were averaged before the cluster resample. The interval is the 2.5 and 97.5 percentiles of 2000 draws. It does not include retraining. Record: `results/deep.json`.

| Comparison | Effect on NLL | 95% cluster interval | Clusters |
|---|---|---|---|
| Mean-plus-mass − sum | +0.012 | [+0.004, +0.021] | mean-plus-mass worse on 21/30 |

Test NLL is 2.858 for the sum and 2.870 for the mean plus mass. That 0.012 nat gap is a factor of exp(0.012) ≈ 1.012 on the geometric mean of the correct-residue probabilities, about 1.2% higher, and about 1.2% lower perplexity. Whole-complex recovery is 0.113 and 0.112; that recovery interval includes zero. Interface NLL is 2.872 and 2.895, a paired gap of +0.023 with interval [+0.004, +0.040], mean worse on 21/30. The sum’s three seeds checkpointed at step 350. Two mean-plus-mass seeds were still best at step 400. That 400-step boundary is what the 800-step follow-up checks. Greater depth shows the aggregation result survives a larger encoder. It does not show a more capable inverse folder.

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
python run_deep_budget.py
python run_strong.py
python run_families.py
python run_reference.py
```

CPU is enough. Python 3.10 or newer. The numbers above used PyTorch 2.14.0. Inverse-folding scripts download PDB files into `data/`, which is gitignored. A locked split file is reused when it is already present. `check_reconstruct.py` needs the magnitude lock. `run_deep.py` needs the earlier locks so it can keep those complexes out of the test. `run_deep_budget.py` reuses the deep lock and does not make a new split. `run_strong.py` reads N, CA, and C from those PDB files and keeps every earlier validation and test complex out of its test. `run_families.py` appends heterodimers released after 2022-06-01 and holds out new clusters only. `run_reference.py` needs a [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) checkout at `data/ProteinMPNN` or `./ProteinMPNN`, with the vanilla `v_48_020` weights. It scores those 24 backbones and retrains the gated sum only to recover the test recovery that was not stored. It stops if the recomputed NLL drifts from `results/families.json` by more than 0.01 nat.

## Layout

| Path | Contents |
|---|---|
| `block.py` | Variants, including the normalized mean, the mean plus mass, and the reconstruction check |
| `inverse_fold.py` | One-block folder and the three-block stack |
| `run_deep.py` | Sum versus mean-plus-mass in the three-block encoder |
| `run_deep_budget.py` | Same comparison at 800 steps, on that same test |
| `frames.py` | Backbone-frame edge features |
| `run_strong.py` | Frame edges, composition gate, then sum versus mean-plus-mass |
| `run_families.py` | That aggregation question on 24 new singleton families |
| `run_reference.py` | Native recovery of that model against ProteinMPNN |
| `check_reconstruct.py` | Fixed-checkpoint logit comparison |
| `run_magnitude.py` | One-block sum versus mean on a fresh locked test |
| `run_confirm.py` | Locked gated-sum versus softmax confirmation |
| `results/` | Metrics for each seed, complex, and cluster |

## Cite

Use the Cite this repository button. It reads `CITATION.cff`.

## License

[MIT](LICENSE). Copyright 2026 Michael DeAngelis.
