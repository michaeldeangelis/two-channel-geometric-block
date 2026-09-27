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
