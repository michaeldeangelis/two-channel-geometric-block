"""Variant E: same-chain vs cross-chain channels, unlocked because D beat A and B.

The label is the product of the two half-space sums. Chain identity is already
a pair feature, so an unsplit gated sum is allowed to select on it.

Same decision rule as Stage 1. E has to beat C, the Stage 1 winner, not only D.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from block import TwoChannelBlock, parameter_count
from synthetic import H_DIM, sample_interface_batch

ROOT = Path(__file__).resolve().parent
STEPS = 1200
BATCH = 128
LR = 2e-3
SEEDS = (0, 1, 2)
MSG_DIM = 16
BASE_HIDDEN = 64
VARIANTS = ("C", "D", "E")


def hidden_for_budget(target: int) -> dict[str, int]:
    widths = {"E": BASE_HIDDEN}
    for variant in VARIANTS:
        if variant == "E":
            continue
        hidden = BASE_HIDDEN
        count = parameter_count(TwoChannelBlock(H_DIM, MSG_DIM, MSG_DIM, hidden, variant))
        while count < int(0.90 * target) and hidden < 512:
            hidden += 8
            count = parameter_count(TwoChannelBlock(H_DIM, MSG_DIM, MSG_DIM, hidden, variant))
        widths[variant] = hidden
    return widths


def evaluate(model: TwoChannelBlock, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        y_hat, _ = model(batch["h"], batch["pos"], batch["mask"], chain=batch["chain"])
        return torch.mean((y_hat - batch["y"]) ** 2).item()


def train_one(variant: str, hidden: int, seed: int, val: dict[str, torch.Tensor]) -> dict[str, float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(20_000 + seed)
    model = TwoChannelBlock(H_DIM, MSG_DIM, MSG_DIM, hidden, variant)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    last = math.inf
    for step in range(1, STEPS + 1):
        batch = sample_interface_batch(BATCH, gen)
        y_hat, _ = model(batch["h"], batch["pos"], batch["mask"], chain=batch["chain"])
        loss = torch.mean((y_hat - batch["y"]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            last = evaluate(model, val)
            print(f"  {variant} seed {seed} step {step:4d}  val_mse {last:.4f}", flush=True)
            model.train()
    return {"val_mse": last, "params": float(parameter_count(model)), "hidden": float(hidden)}


def _std(xs: list[float]) -> float:
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / max(len(xs) - 1, 1))


def beats(challenger: list[float], baseline: list[float]) -> bool:
    return (sum(challenger) / len(challenger) + _std(challenger)) < (
        sum(baseline) / len(baseline) - _std(baseline)
    )


def main() -> None:
    target = parameter_count(TwoChannelBlock(H_DIM, MSG_DIM, MSG_DIM, BASE_HIDDEN, "E"))
    widths = hidden_for_budget(target)
    print("parameter budget target", target)
    print("hidden widths", widths)
    val = sample_interface_batch(2000, torch.Generator().manual_seed(54321))
    var_y = torch.var(val["y"], unbiased=False).item()
    print(f"label mean {val['y'].mean().item():.3f}  variance {var_y:.4f}")

    runs: dict[str, list[dict[str, float]]] = {v: [] for v in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            runs[variant].append(train_one(variant, widths[variant], seed, val))

    summary = {}
    for variant, rows in runs.items():
        mses = [r["val_mse"] for r in rows]
        summary[variant] = {
            "mse_mean": sum(mses) / len(mses),
            "mse_std": _std(mses),
            "mse_seeds": mses,
            "params": rows[0]["params"],
            "hidden": rows[0]["hidden"],
        }
        print(
            f"{variant}  mse {summary[variant]['mse_mean']:.4f} ± {summary[variant]['mse_std']:.4f}"
            f"  params {int(rows[0]['params'])}"
        )

    e = [r["val_mse"] for r in runs["E"]]
    verdict = {
        "E_beats_C": beats(e, [r["val_mse"] for r in runs["C"]]),
        "E_beats_D": beats(e, [r["val_mse"] for r in runs["D"]]),
    }
    if verdict["E_beats_C"]:
        decision = "split-helps"
    else:
        decision = "split-not-needed"
    print("verdict", verdict, "decision", decision)
    path = ROOT / "results" / "stage_e.json"
    path.write_text(
        json.dumps(
            {
                "task": "product of same-chain and cross-chain half-space sums",
                "steps": STEPS,
                "seeds": list(SEEDS),
                "label_variance": var_y,
                "summary": summary,
                "verdict": verdict,
                "decision": decision,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
