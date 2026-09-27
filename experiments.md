# First experiment

Goal: isolate the architectural claim. No protein weights and no GPU.

Hold fixed: geometric inputs, training data, decoder, and approximate parameter budget. Change only the aggregation.

## Variants

| Id | Variant | Question |
|---|---|---|
| A | Local geometric softmax | Baseline |
| B | Softmax plus neighbor-count features | Does counting alone fix cardinality? |
| C | Gated sum only | Is selective attention necessary? |
| D | Softmax plus gated sum, plus `log(1 + rho)` | Do the channels complement each other? |
| E | D, split into same-chain and cross-interface | Does the protein distinction help? |

Run A–D first. Run E only if D beats A and D beats B. Do not add a global channel in this round.

## Stage 1 — synthetic cardinality

Build a task whose label depends on accumulated contributions, not on the average neighbor. Minimum cases:

- Identical neighbor states, count 2 versus count 20. Softmax must be unable to separate them from its aggregated message alone. The dataset must still be solvable by a sum.
- Mixed signs, so a sum of gated messages can cancel and a count cannot.
- A cutoff: nodes beyond `c(r) = 0` must not affect local channels.

Pass: D beats A and D beats B on held-out synthetic labels. C is reported, not required to win.

Kill: B matches D within noise. Stop. Write the result in this file and do not start Stage 2.

## Stage 1 result — pass

Ran 26 Sep 2026 on CPU. Record: `results/stage1.json`. Three seeds, 1200 steps, matched parameter budget (7.1k–7.6k). Held-out MSE, zero-predictor baseline 4.299.

| Variant | Val MSE | Skill |
|---|---|---|
| A geometric softmax | 0.731 ± 0.018 | 0.830 |
| B softmax plus ungated count | 0.720 ± 0.009 | 0.833 |
| C gated sum only | 0.122 ± 0.006 | 0.972 |
| D softmax plus gated sum | 0.149 ± 0.016 | 0.965 |
| R softmax plus gated count, no vector sum | 0.694 ± 0.018 | 0.839 |

Pre-registered rule: D beats a baseline only when `mean_D + std_D < mean_baseline - std_baseline`. D beats A and D beats B. Decision: **pass**.

What the pass does and does not say:

- An ungated neighbor count does not close the gap. B matches A.
- A gated scalar count does not close it either. R matches B. The vector sum is carrying the result, not `log(1+rho)` alone.
- Softmax does not complement the gated sum on this task. C is ahead of D. Selective attention was not required.
- This is a synthetic half-space sum, not inverse folding.

## Stage 2 — behavior checks

Passed in `test_block.py` on the same block. These do not show biological usefulness. They show the block does what it claims.

- Rigid rotation and translation leave scalar outputs unchanged.
- Adding disconnected distant nodes leaves `m_select`, `m_sum`, and `rho` unchanged.
- `log(1 + rho)` is still present after any normalization inside the MLP.

## Variant E result — split helps on a product task

Unlocked because D beat A and B. Ran the same day, CPU, three seeds, 1200 steps. Record: `results/stage_e.json`.

Label is the product of a same-chain half-space sum and a cross-chain half-space sum. Chain identity is already a pair feature, so the unsplit gated sum is allowed to select on it. Parameter counts sit within about 5% (C 9286, D 9442, E 9714). Label variance 4.876.

| Variant | Val MSE |
|---|---|
| C gated sum, chains mixed | 2.570 ± 0.034 |
| D softmax plus gated sum, chains mixed | 1.455 ± 0.930 |
| E same channels, split by chain | 0.301 ± 0.011 |

E beats C and E beats D under the same non-overlapping-seed rule. On this task, at this budget, the split learned faster and finished lower.

`±` is the sample standard deviation across three seeds, with the n−1 denominator. It is not a standard error and not a confidence interval. Non-overlapping seed ranges were a screening rule fixed before the run. Three seeds do not make that screen a statistical conclusion.

D's seeds were 0.76, 1.10, and 2.51, and two were still falling at step 1200. The comparison of E with D is provisional until D is given a longer diagnostic run.

## How to read the synthetic results

The empirical gaps stand. Two interpretations do not.

An ungated count beside softmax did not close the Stage 1 gap, and neither did a gated scalar density. Multiplying a uniform softmax average by the neighbor count recovers an ordinary sum of per-neighbor values. It does not, in general, recover a query-selected sum. The failure of B is the empirical fact. The uniform-attention identity is the qualification.

An unsplit vector sum that sees chain identity can represent the two chain sums on separate coordinates: put the same-chain contribution in one component and the cross-chain contribution in another, and the sum keeps both. Variant E makes that split explicit. The unsplit model has to learn it. The Stage E gap is evidence of easier optimization under the tested budget, not proof that the unsplit block cannot represent the product.

Softmax still has no evidence in its favor on these tasks. C beat D. Stage 3 includes the chain-split gated sum with no attention, so a win for E cannot be credited to an attention branch the task did not need.

## Stage 3 — inverse folding

Multichain complexes only. A monomer set cannot test chain separation.

Comparisons, at a matched step budget:

| Comparison | Question |
|---|---|
| A vs C | Which aggregation works on structures? |
| C vs D | Does attention add anything? |
| C vs S | Does an explicit chain split help a gated sum? |
| D vs E | Does that help depend on attention? |

S is C computed separately on same-chain and cross-chain neighbors. No softmax.

Rules for the run:

- Node and pair features are coordinates, chain identity, and within-chain index. Amino-acid identity is not an input. Shuffling the labels must leave the logits unchanged.
- Train and test complexes share no chain pair above 30% identity on the shorter chain.
- Report interface recovery separately from the whole complex. An interface residue has a CA on the other chain within 8 Å.
- The block cutoff is 10 Å.
- D also trains for twice as many steps. That curve is a diagnostic. The comparison table uses the matched budget, so extra optimization time cannot count as a win.
- `±` remains the sample standard deviation across seeds.
- This is not a comparison to SolubleMPNN. A win against geometric softmax is not that comparison.

The run used 20 heterodimers from the PDB (two protein chains, each 40–160 residues, resolution ≤ 2.5 Å, released 2012–2022, at least eight cross-chain CA contacts within 8 Å). Clustering at 30% identity left 17 clusters. The split is 15 train and 5 test (`7OWD`, `7S6O`, `7MIC`, `7QGS`, `7LVS`): 830 test residues, 253 of them interface. Matched budget 200 AdamW steps, three seeds. D also ran to 400. `±` is the sample standard deviation across seeds.

Uniform guessing has recovery 1/20 = 0.05 and NLL ln(20) ≈ 2.996.

| Variant | Recovery | Interface | Non-interface | NLL |
|---|---|---|---|---|
| A geometric softmax | 0.111 ± 0.007 | 0.153 ± 0.016 | 0.092 ± 0.006 | 2.912 ± 0.007 |
| C gated sum | 0.124 ± 0.005 | 0.158 ± 0.014 | 0.109 ± 0.005 | 2.889 ± 0.007 |
| S chain-split gated sum | 0.100 ± 0.015 | 0.104 ± 0.020 | 0.098 ± 0.013 | 2.911 ± 0.005 |
| D hybrid | 0.123 ± 0.011 | 0.162 ± 0.017 | 0.106 ± 0.016 | 2.890 ± 0.011 |
| E chain-split hybrid | 0.102 ± 0.008 | 0.107 ± 0.029 | 0.101 ± 0.011 | 2.922 ± 0.011 |

Same non-overlapping-seed screen as Stage 1, now applied in the direction of higher recovery and lower NLL. C's whole-complex recovery and NLL sit just above A's: the recovery gap clears the screen by about 0.001. Their interface ranges overlap, so geometric softmax versus gated sum is still unresolved where the chains meet. C and D overlap on every reported number. Attention did not add a detectable gain. C beats S, and D beats E, on both whole-complex recovery and interface recovery. The explicit chain split was worse, and the gap is largest at the interface, which is where the split was supposed to help. Three seeds still do not make the screen a statistical conclusion. The split gap is the one that is large relative to the seed scatter.

That does not prove the unsplit block is a more powerful representation. S and E both reached interface recovery near 0.18–0.19 at step 50, then fell. At the preregistered budget they had lost that gain while the unsplit models held. D's extra 200 steps made it worse, not better (recovery 0.102, interface 0.149 at step 400). Extra optimization time does not rescue the hybrid, and it does not explain the split's loss.

Absolute skill is small. The best NLL is still close to ln(20). Fifteen training complexes cannot support a claim about protein modeling in general. They can support the narrower claim this stage asked for: on multichain inverse folding, with amino-acid identity kept out of the features and homologs kept out of the test split, the chain split did not help and attention did not help. The synthetic inductive bias did not transfer.

This is not a SolubleMPNN result.

## How to read Stage 3

The numbers above stand. Several readings do not.

C and D overlapping is the firm part: at this budget, attention had no demonstrated benefit. S and E trailing their unsplit counterparts is also a result about the implementations that were trained, including the width those branches received when the parameter counts were matched. It is not a proof that a chain split cannot represent an interface.

C's edge over A cleared the old seed screen by about 0.001. That is too thin to call an architectural win, and the interface ranges overlapped. Uniform guessing is a weak baseline, because amino-acid frequencies are unequal. Three seeds measure training variability. They do not measure uncertainty across proteins. The five test complexes were the independent units, and they have now been used to choose an architecture, so they are retired from being the final test.

The curves do not by themselves say why recovery fell. A later step with worse validation and better training loss would be overfitting. Both getting worse would be optimization. D's extra 200 steps showed that extending that run did not help. They did not show that the hybrid had been optimized. A shift toward common amino acids can also move recovery. That needs the training loss, the validation NLL, and the predicted residue mix.

A categorical amino acid is one output. It can still depend on several distinct inputs. The output format does not explain the split.

## Stage 5 — C versus A on new families

S and E are paused. D is trained as a control at the same budget, not as the comparison. The question is whether the unsplit gated sum beats geometric softmax on complexes that did not choose the architecture.

Rules fixed before the run:

- Train, validation, and test are whole homology clusters. The Stage 3 test ids (`7OWD`, `7S6O`, `7MIC`, `7QGS`, `7LVS`) are forced into train. An audit must show no chain, in either order, shares 30% identity or more with a chain in another partition. The identity is the best ungapped window on the shorter chain.
- The checkpoint is the lowest unweighted mean of per-complex validation NLL. Test complexes are scored once, at that checkpoint.
- The primary summary is the paired per-complex difference, each complex one vote. Seed standard deviation is training variability only.
- F is the training-set amino-acid frequency: NLL uses those frequencies, recovery uses the most common training residue.
- Z is C with coordinates set to zero. Chain id and sequence index remain. It asks whether the block is using 3D distance.
- Parameter counts for A, C, and D are matched to the same rule as before. Z uses C's width, so it has C's parameter count.
- This is still not a SolubleMPNN comparison.

Ran on CPU after those rules were written. Record: `results/scale.json`. 72 heterodimers, 46 clusters. Split 44 train / 14 validation / 14 test. Worst cross-partition identity 0.255. The five Stage 3 test ids are in train. Checkpoint on validation NLL, then one look at the test complexes. `±` below, when it describes complexes, is the sample standard deviation across the 14 test complexes. Seed standard deviations are labeled separately and are training variability only.

Training-set frequencies put leucine at 0.097. That constant predictor is F. Uniform NLL is ln(20) ≈ 2.996. Parameter counts: A 23561, C 22278, Z 22278, D 24442.

| Model | Test NLL | Recovery | Interface NLL |
|---|---|---|---|
| F training frequencies | 2.920 ± 0.045 | 0.096 ± 0.037 | 2.932 ± 0.077 |
| A geometric softmax | 2.916 ± 0.051 | 0.096 ± 0.024 | 2.947 ± 0.113 |
| C gated sum | 2.893 ± 0.038 | 0.100 ± 0.019 | 2.936 ± 0.124 |
| Z gated sum, coordinates removed | 2.932 ± 0.030 | 0.089 ± 0.031 | 2.952 ± 0.066 |
| D hybrid, control | 2.892 ± 0.040 | 0.100 ± 0.026 | — |

Paired C minus A, one vote per complex, seeds averaged first:

| Difference | Mean | Across complexes | Complexes favoring C |
|---|---|---|---|
| NLL | −0.023 | 0.027 | 13/14 |
| Recovery | +0.004 | 0.016 | 8/14 |
| Interface NLL | −0.011 | 0.055 | 9/14 |

The NLL direction is consistent. The size is small: 0.023 nats, against a complex-to-complex scatter of 0.027. That scatter describes the 14 complexes. It is not a confidence interval, and it is not a cluster-level uncertainty. The one complex where A won is `7T8I` (+0.036). Recovery does not show the same advantage. Interface NLL does not either. A drop of 0.023 nats is a perplexity ratio of exp(−0.023) ≈ 0.977, about 2.3% lower. The model can assign more probability to the correct residue without making it the argmax.

C is 0.027 nats better than F on average, and better on 11 of 14 complexes. A is 0.004 nats better than F, on 9 of 14. Beating F means the model has predictive information beyond the training-set amino-acid frequencies. It does not by itself say that information is geometric. Chain length and sequence position could carry it.

Z loses to C on all 14 complexes (mean NLL difference −0.039 for C minus Z) and is worse than F on 9 of 14. Z zeros coordinates before the cutoff. Every distance feature becomes 0, and every masked pair falls inside the window, so the 10 Å neighborhood is replaced by a complete graph. Sequence separation and chain identity stay. The comparison removes explicit distances and changes which neighbors exist. It supports that those two changes matter together. It does not show that Z contains no geometry.

D matches C. Attention still does not add a detectable gain. A has more parameters than C (23561 against 22278), so C's NLL edge is not a wider model.

Several seeds selected a checkpoint before step 400. On those runs, validation NLL was higher at the last step. On C seed 2, Z seed 0, and D seed 2, training NLL was still falling while validation NLL rose. That is overfitting, and the test table uses the earlier checkpoint. It does not show that training had converged.

## Stage 6 — confirmation on a locked test

The architecture stays C. A is the geometric-softmax control. F is the training-set frequency baseline. D, Z, S, and E are not retrained.

The Stage 5 validation and test complexes are a development record. They are forced into train. They are not the confirmation test.

Rules fixed before the run:

- Same widths as Stage 5: A at hidden 104, C at hidden 80. Same optimizer, 400 steps, three seeds, and the same validation-NLL checkpoint.
- Whole-complex NLL is the primary endpoint. Recovery and interface NLL are secondary.
- The split is written to disk before training. A rerun uses that split.
- Homology clusters are the unit. A cluster's effect is the mean of its complexes. The reported effect is the mean across clusters. Uncertainty is a cluster bootstrap: 2000 resamples, clusters drawn with replacement, 2.5 and 97.5 percentiles. That interval is not the across-complex standard deviation.
- An audit must still show no cross-partition chain at or above 30% identity, in either chain order.

Ran on CPU after those rules were written. Record: `results/confirm.json`. 240 heterodimers, 106 clusters. The locked split is 142 train / 50 validation / 48 test. All 48 test clusters are singletons. Worst cross-partition identity 0.275. The 33 development ids, including the Stage 5 test, are in train and in neither validation nor test. Widths and parameter counts match Stage 5.

The effect is the mean of cluster means. Seeds were averaged inside each complex before clusters were resampled. The interval is the 2.5 and 97.5 percentiles of 2000 cluster draws, a 95% bootstrap percentile interval. It is uncertainty across these held-out clusters. It does not include retraining, and it does not include a different training partition. Because every test cluster is one complex, the resample is also a complex resample.

| Comparison | Effect | Bootstrap interval | Clusters favoring C |
|---|---|---|---|
| C − A, whole-complex NLL | −0.033 | [−0.039, −0.027] | 45/48 |
| C − F, whole-complex NLL | −0.048 | [−0.055, −0.041] | 45/48 |
| C − A, recovery | +0.006 | [+0.001, +0.010] | 30/48 |
| C − A, interface NLL | −0.030 | [−0.041, −0.019] | 38/48 |

Primary endpoint: the likelihood advantage holds. exp(−0.033) ≈ 0.968, about 3.2% lower perplexity than softmax. Test NLL is 2.846 for C, 2.878 for A, and 2.894 for F. A itself beats F by 0.015, interval [−0.020, −0.011], so on this larger set softmax is no longer indistinguishable from the frequency table. C's remaining gap over A is on top of that.

Lower NLL means a higher geometric mean of the probabilities assigned to the correct residues. It does not mean every correct residue received a higher probability. It also does not mathematically explain the recovery change. Recovery records only whether a change crossed the argmax.

Recovery moves by about half a percentage point (0.118 against 0.112, frequency baseline 0.105). The interval excludes zero. The predictions are still close to the common amino acids. Interface NLL also favors C here; the Stage 5 interface comparison was too noisy to say that. It is a secondary endpoint. There is no contradiction between those two interface outcomes.

Checkpoints were steps 300, 350, and 400. On the earlier stops, training NLL was still falling while validation NLL had turned up. The table uses those checkpoints.

This is a confirmed, small likelihood advantage for the same gated sum. It is not a useful inverse-folding model. The confirmation stays closed.

## Stage 7 — sum versus normalized mean

C beats A. That does not isolate accumulated magnitude as the reason. C, N, and Q use the same gate and the same message vectors.

- C keeps the implemented gated sum, including its existing log(1+ρ) slot.
- N replaces that sum with the mean, sum divided by (1e-8 + ρ), and the mass slot is zero.
- Q uses that same mean and puts log(1+ρ) back.

Same hidden width as C, 80. Same optimizer, 400 steps, three seeds, and the validation-NLL checkpoint. Whole-complex NLL is primary. The Stage 6 validation and test complexes stay out of this test. The interval is the same 95% cluster bootstrap, with seeds averaged first.

Reading, fixed before the run: if N is worse than C and Q closes that gap, the mass channel is enough to recover the sum. If N and Q are both worse than C, handing the network the sum is easier than handing it the mean and the mass separately. If N matches C, dividing by the gate mass did not matter.

Ran on CPU after those rules were written. Record: `results/magnitude.json`. 480 heterodimers, 196 clusters. Locked split 430 train / 20 validation / 30 test. The test complexes were not in any earlier validation or test split. All 30 test clusters are singletons. Worst cross-partition identity 0.293. C, N, and Q each have 22,278 parameters. By construction, the mean times (1e-8 + ρ) equals the sum.

| Comparison | Effect on NLL | 95% cluster interval | Clusters |
|---|---|---|---|
| N − C | +0.036 | [+0.029, +0.042] | N worse on 29/30 |
| Q − C | +0.029 | [+0.022, +0.036] | Q worse on 27/30 |
| Q − N | −0.007 | [−0.011, −0.003] | Q better on 23/30 |

Test NLL is 2.862 for C, 2.898 for N, and 2.891 for Q. Dividing by the gate mass hurts. Putting log(1+ρ) back helps a little and does not recover the sum. C already had that mass slot beside the sum, so Q versus C changes only the vector, from the sum to the mean. The mean and the mass determine the sum, and the fuse is an MLP, so the product is representable. The trained mean-plus-mass model did not match the trained sum. That weakens an account in which the only difference is information the mean throws away.

Whole-complex recovery is 0.107 for C, 0.094 for N, and 0.098 for Q. Interface NLL is 2.903, 2.933, and 2.932. The mass slot does not close the interface gap. Seeds were averaged before the cluster resample. The interval does not include retraining variation.

The fusion layer receives s = log(1+ρ), not ρ itself. Rebuilding the sum is mean times (eps + exp(s) − 1).

The inputs still determine the sum. A finite MLP has to learn that nonlinear map. Having the information and having an equally easy computation are different. A fixed-weight reconstruction can match while separately trained models still differ. Different gradient trajectories, activation scale, regularization, and generalization all remain compatible. The reconstruction check establishes equivalence at fixed weights. It does not identify which of those produced the training gap.

The claim this supports: in this architecture and training regime, directly supplying an unnormalized gated aggregate improves held-out likelihood relative to a normalized aggregate, even when gate mass remains available. It does not separate easier optimization, activation scale, implicit regularization, or generalization of the supplied product.

## Stage 8 — the same substitution in a deeper encoder

No further one-block variants. Three blocks, residue width 64, message width 64, mixer hidden 128. The only change is C versus Q: the sum, or the mean with log(1+ρ) kept. Same 400 steps, three seeds, and validation-NLL checkpoint. Whole-complex NLL is primary. Complexes already used as validation or test stay out of this test. The interval is the same 95% cluster bootstrap.

A fixed trained one-block C is also checked: replace its sum by the mean and the explicit reconstruction above, and compare logits. The tolerance is 1e-4 absolute. The magnitude run did not save weights, so this check fits one seed on that same locked split and then does not fit again.

Ran on CPU after those rules were written. Record: `results/deep.json` and `results/reconstruct.json`.

The reconstruction check used seed 0 of the one-block C, checkpoint step 300. Across 128,600 test logits the largest absolute difference was 8.3e-7, and the mean absolute difference was 5.4e-8. Rebuilding the sum from the mean and log mass matches the implemented sum inside the masking and fusion code. The training gap is not an algebraic mismatch.

The deeper run locked 429 train / 21 validation / 30 test from the same 480 heterodimers. All 30 test clusters are singletons. None of them was in an earlier validation or test split. Worst cross-partition identity 0.277. C and Q each have 203,034 parameters.

| Comparison | Effect on NLL | 95% cluster interval | Clusters |
|---|---|---|---|
| Q − C | +0.012 | [+0.004, +0.021] | Q worse on 21/30 |

Test NLL is 2.858 for the sum and 2.870 for the mean plus mass. Whole-complex recovery is 0.113 and 0.112. Interface NLL is 2.872 and 2.895; the paired interface gap is +0.023 with interval [+0.004, +0.040], and the mean is worse on 21/30. Recovery’s interval includes zero, [−0.006, +0.004]. Recovery is not separated. Seeds were averaged first. The interval does not include retraining or a new partition.

Test NLL 2.858 versus 2.870 is a gap of 0.012 nats. That raises the geometric mean of the correct-residue probabilities by exp(0.012) ≈ 1.012, about 1.2% higher, under the same complex weighting as the NLL. Equivalently, perplexity is about 1.2% lower. exp(−0.012) is the perplexity ratio, not the probability ratio.

C’s three seeds all checkpointed at step 350, with validation NLL higher at step 400. Two Q seeds were still best at step 400. The 0.012 nat figure is smaller than the 0.029 nat one-block gap, and the complexes and the split are different, so that difference is not evidence that depth reduced the effect. Greater depth shows that the aggregation result survives a larger encoder. It does not show a more capable inverse folder. The reconstruction match does not identify upstream gradients as the cause.

The advantage of supplying the unnormalized aggregate survives this deeper encoder. It does not say which of optimization, scale, regularization, or generalization produces it.

## Stage 8 follow-up — same test, twice the steps

The 400-step result stands. Two mean-plus-mass seeds selected step 400, so their best validation score may lie later. This follow-up retrains both C and Q from the same seeds for 800 steps, twice the original budget, chosen before the run. Checkpointing stays the lowest unweighted mean of per-complex validation NLL. The test is not used to stop. The test set is the one already examined in `results/deep.json`. No new split. No one-block variants.

Reading, fixed before the run: if the selected mean-plus-mass checkpoints move past 400 and the Q−C interval still excludes zero, the advantage survived the extra training. If the interval includes zero, the extra training closed the gap. If a seed is still best at step 800, the boundary is still open and this follow-up stops there.

Ran on CPU after those rules were written. Record: `results/deep_budget.json`. Validation NLL through step 400 matches the 400-step run exactly, so this is that trajectory continued.

Selected steps are 800, 450, and 750 for the sum, and 800, 750, and 750 for the mean plus mass. The two mean-plus-mass seeds that had selected step 400 now select step 750. One seed of each aggregation is still best at step 800. The boundary stays open for those two seeds, and this follow-up stops.

| Comparison | Effect on NLL | 95% cluster interval | Clusters |
|---|---|---|---|
| Q − C | +0.013 | [+0.004, +0.022] | Q worse on 22/30 |

Test NLL is 2.836 for the sum and 2.850 for the mean plus mass. The factor on the geometric mean of the correct-residue probabilities is exp(0.013) ≈ 1.013, about 1.3% higher, with the same complex weighting as the NLL. Perplexity is about 1.3% lower. exp(−0.013) is the perplexity ratio. Interface NLL differs by +0.020, interval [+0.005, +0.036], with the mean worse on 17 of 30. Recovery differs by −0.004, interval [−0.011, +0.003], which includes zero.

The extra training did not close the gap. The 400-step result was not an artifact of stopping those two mean-plus-mass seeds on their last checkpoint.
