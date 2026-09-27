# Frozen hypothesis

## Corrections already accepted

- Attention weights are not physical forces. Softmax does not inherently distort geometry. What it can discard is magnitude.
- There is no universal interaction cutoff. A smooth `c(r)` is a computational window, not a claim that physics ends there. Long-range effects stay in a later global channel, not in a larger local cutoff by default.
- Low sampling temperature picks high model probability. It is not a minimum-energy or solubility procedure. SolubleMPNN’s documented distinction is training on soluble proteins.
- AlphaFold 3’s pair-weighted averaging is one operation in its MSA module. This block is not “AF3 with softmax removed.”

## The gap

If every neighbor value in a set is the same vector `v`, then

```
sum_j softmax(s)_j v = v
```

whether the neighborhood has 2 neighbors or 20. The normalized average cannot report that difference by itself. Other features might put the count back. That is why a count-feature control is mandatory.

Prior art already includes cardinality-preserving attention and Principal Neighbourhood Aggregation. Adding a sum next to attention is not a contribution. A contribution would be a specific block that beats both geometric softmax and geometric softmax plus a simple count, on a task where the label actually depends on accumulated interactions.

## Block

Each residue has a state `h_i`. Each pair has geometric features `e_ij`: backbone distances, relative orientations (including a handedness-sensitive feature), sequence separation, and chain identity.

`c(r)` decreases smoothly to 0 at a computational cutoff. Every local sum includes a self-edge so the softmax denominator is defined.

Selective channel:

```
s_ij = (q_i · k_j) / sqrt(d) + b(e_ij)
a_ij = c(r_ij) exp(s_ij) / sum_k c(r_ik) exp(s_ik)
m_select = sum_j a_ij V(h_j, e_ij)
```

Accumulated-evidence channel. Gates do not share a budget:

```
g_ij = sigmoid(G(h_i, h_j, e_ij))
m_sum = sum_j c(r_ij) g_ij Phi(h_j, e_ij)
rho   = sum_j c(r_ij) g_ij
```

`Phi` may have positive and negative components. This is accumulated learned evidence, not an energy and not a force.

Update:

```
h' = h + MLP(h, m_select, m_sum, log(1 + rho), m_global)
```

`m_global` is omitted in the first experiment. When any channel is normalized, `log(1 + rho)` stays outside that normalization.

## Protein-specific variant (later)

Run the two local channels separately on same-chain neighbors and on cross-interface neighbors. A residue can then represent strong internal support, strong partner support, both, or neither. That split is an empirical question, not part of the first kill test. The heterodimer screens in experiments.md ran it. The split lost, and it is paused. The comparison that remains is the gated sum against geometric softmax.

## Inverse folding head (later)

```
p(a_i | X, a_<i) = softmax(W h_i / T)
```

`T` is a sampling temperature. Sequence quality is measured directly. Temperature is not a physical objective.

Scalar predictions use rotation- and translation-invariant features. If a later head emits coordinate updates, those updates have to transform with the input frame. The first experiment does not emit coordinates.
