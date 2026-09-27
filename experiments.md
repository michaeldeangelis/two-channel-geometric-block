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

E beats C and E beats D under the same non-overlapping-seed rule. Decision: the split earns its keep on this task.

D is unstable across seeds (0.76, 1.10, 2.51) and two seeds were still falling at step 1200. C had flattened well above E. A longer budget might narrow D's best seed toward E. It does not erase the gap at the budget that was fixed in advance.

## Stage 3 — inverse folding (separate budget, not before Stage 1 passes)

Family-separated split. Report held-out likelihood, sequence recovery, and slices by neighborhood density and protein length. Temperature is a sampler only. Compare to SolubleMPNN only after A–D exist under the same decoder and budget. A win against an untrained baseline is not a win against SolubleMPNN.
