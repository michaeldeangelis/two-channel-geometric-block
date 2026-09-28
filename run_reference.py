"""Native recovery of the frame model against ProteinMPNN.

The rules are in experiments.md, Stage 11. They were written before this
script scored either model. The 24-family test is not used to stop or to
choose a checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from frames import attach_backbone
from inverse_fold import tensor_batch
from proteins import AA3, PDB_DIR, ROOT, load_cache
from run_confirm import cluster_interval
from run_families import LOCK, STEPS, WIDTH, build
from run_scale import _mean, train_one

RESULTS = Path(__file__).resolve().parent / "results"
FAMILIES = RESULTS / "families.json"
OUT = RESULTS / "reference.json"
NLL_TOL = 0.01
N_SAMPLES = 8
TEMPERATURE = 0.1
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA1_TO_3 = {one: three for three, one in AA3.items()}


def _mpnn_root() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "data" / "ProteinMPNN",
        Path(__file__).resolve().parent / "ProteinMPNN",
    ]
    for path in candidates:
        if (path / "protein_mpnn_utils.py").exists():
            return path
    raise SystemExit("ProteinMPNN checkout not found next to this script or under data/")


def _oxygens(comp: dict) -> dict[tuple[str, int], tuple[float, float, float]]:
    path = PDB_DIR / f"{comp['id']}.pdb"
    found = {}
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or line[16] not in (" ", "A"):
            continue
        if line[12:16].strip() != "O":
            continue
        found[(line[21], int(line[22:26]))] = (
            float(line[30:38]), float(line[38:46]), float(line[46:54]),
        )
    return found


def write_backbone(comp: dict, path: Path) -> None:
    """Cached residues only. Residue numbers restart at 1 on each chain."""
    oxy = _oxygens(comp)
    lines = []
    serial = 1
    for part in comp["chains"]:
        for i, aa in enumerate(part["seq"]):
            resi = int(part["resseq"][i])
            key = (part["chain"], resi)
            if key not in oxy:
                raise SystemExit(f"{comp['id']} missing O at {key}")
            coords = {
                "N": part["n"][i],
                "CA": part["xyz"][i],
                "C": part["c"][i],
                "O": oxy[key],
            }
            for name in ("N", "CA", "C", "O"):
                x, y, z = coords[name]
                atom = f" {name:<3}" if len(name) < 4 else name
                lines.append(
                    f"ATOM  {serial:5d} {atom} {AA1_TO_3[aa]} {part['chain']}{i + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]:>1}\n"
                )
                serial += 1
    path.write_text("".join(lines) + "END\n")


def _load_mpnn():
    root = _mpnn_root()
    sys.path.insert(0, str(root))
    from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

    device = torch.device("cpu")
    checkpoint = torch.load(
        root / "vanilla_model_weights" / "v_48_020.pt", map_location=device, weights_only=True,
    )
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, parse_PDB, tied_featurize, device


def _recovery(pred: torch.Tensor, native: torch.Tensor, keep: torch.Tensor) -> float:
    denom = int(keep.sum())
    if denom == 0:
        return float("nan")
    return float(((pred == native) & keep).sum() / denom)


def score_proteinmpnn(test: list[dict]) -> list[dict]:
    model, parse_PDB, tied_featurize, device = _load_mpnn()
    folder = ROOT / "reference_pdb"
    folder.mkdir(exist_ok=True)
    rows = []
    for index, comp in enumerate(test):
        path = folder / f"{comp['id']}.pdb"
        write_backbone(comp, path)
        letters = [part["chain"] for part in comp["chains"]]
        parsed = parse_PDB(str(path), input_chain_list=letters)[0]
        for part in comp["chains"]:
            got = parsed[f"seq_chain_{part['chain']}"]
            if got != part["seq"]:
                raise SystemExit(f"{comp['id']} chain {part['chain']} parsed as {got} not {part['seq']}")
        chain_dict = {parsed["name"]: (sorted(letters), [])}
        featured = tied_featurize([parsed], device, chain_dict)
        X, S, mask = featured[0], featured[1], featured[2]
        chain_M, chain_encoding, residue_idx = featured[4], featured[5], featured[12]
        chain_M_pos = featured[10]
        masked_letters = featured[8][0]
        masked_lengths = featured[9][0]
        with torch.no_grad():
            logp = model.unconditional_probs(X, mask, residue_idx, chain_encoding)
            pred = logp[..., :20].argmax(dim=-1)
            native = S.clamp(max=19)
            keep = mask.bool() & (S < 20)
            unconditional = _recovery(pred[0], native[0], keep[0])
            torch.manual_seed(10_000 + index)
            randn = torch.randn(N_SAMPLES, X.shape[1], device=device)
            Xb = X.expand(N_SAMPLES, -1, -1, -1).contiguous()
            Sb = S.expand(N_SAMPLES, -1).contiguous()
            mask_b = mask.expand(N_SAMPLES, -1).contiguous()
            chain_b = chain_M.expand(N_SAMPLES, -1).contiguous()
            enc_b = chain_encoding.expand(N_SAMPLES, -1).contiguous()
            idx_b = residue_idx.expand(N_SAMPLES, -1).contiguous()
            pos_b = chain_M_pos.expand(N_SAMPLES, -1).contiguous()
            omit = np.zeros(21, np.float32)
            bias = np.zeros(21, np.float32)
            zeros = torch.zeros(N_SAMPLES, X.shape[1], 21, device=device)
            sampled = model.sample(
                Xb, randn, Sb, chain_b, enc_b, idx_b, mask=mask_b, temperature=TEMPERATURE,
                omit_AAs_np=omit, bias_AAs_np=bias, chain_M_pos=pos_b,
                omit_AA_mask=torch.zeros(N_SAMPLES, X.shape[1], 21, dtype=torch.int32, device=device),
                pssm_coef=torch.zeros(N_SAMPLES, X.shape[1], device=device),
                pssm_bias=zeros, pssm_multi=0.0,
                pssm_log_odds_flag=False,
                pssm_log_odds_mask=torch.ones(N_SAMPLES, X.shape[1], 21, device=device),
                pssm_bias_flag=False, bias_by_res=zeros,
            )["S"]
            sample_rates = [
                _recovery(sampled[j], native[0], keep[0]) for j in range(N_SAMPLES)
            ]
        batch = tensor_batch([comp])
        iface = batch["interface"][0]
        by_letter = {}
        cursor = 0
        for part in comp["chains"]:
            n = len(part["seq"])
            by_letter[part["chain"]] = iface[cursor : cursor + n]
            cursor += n
        pieces = []
        for letter, n in zip(masked_letters, masked_lengths):
            if len(by_letter[letter]) != int(n):
                raise SystemExit(f"{comp['id']} chain {letter} length mismatch")
            pieces.append(by_letter[letter])
        iface_alpha = torch.cat(pieces)
        rows.append({
            "id": comp["id"],
            "n": int(keep[0].sum()),
            "unconditional": unconditional,
            "sampled": _mean(sample_rates),
            "unconditional_interface": _recovery(pred[0], native[0], keep[0] & iface_alpha),
        })
        print(
            f"  {comp['id']}  unconditional {unconditional:.3f}  sampled {rows[-1]['sampled']:.3f}",
            flush=True,
        )
    return rows


def _take(complexes, ids):
    by_id = {c["id"]: c for c in complexes}
    return [by_id[pdb_id] for pdb_id in ids]


def recompute_frame(train, val, test, locked_runs: list[dict]) -> tuple[list[dict], bool]:
    """Return per-seed test rows, and whether validation NLL reproduced."""
    rows = []
    for seed, locked in enumerate(locked_runs):
        done = train_one("C", "C", WIDTH, True, seed, train, val, test, build=build, steps=STEPS)
        history = {row["step"]: row["val_nll"] for row in done["history"]}
        locked_hist = {row["step"]: row["val_nll"] for row in locked["history"]}
        drift = max(abs(history[step] - locked_hist[step]) for step in locked_hist)
        print(f"  seed {seed} val-nll drift {drift:.6f} best step {done['best_step']}", flush=True)
        if drift > NLL_TOL:
            return rows, False
        rows.append(done.pop("test"))
    return rows, True


def main() -> None:
    locked = json.loads(FAMILIES.read_text())
    if locked["test_ids"] != json.loads(LOCK.read_text())["test"]:
        raise SystemExit("family lock and families.json disagree on the test")
    pooled = load_cache()
    kept = []
    for comp in pooled:
        if comp["id"] in set(locked["train_ids"]) | set(locked["val_ids"]) | set(locked["test_ids"]):
            if not attach_backbone(comp):
                raise SystemExit(f"missing backbone on a locked complex {comp['id']}")
            kept.append(comp)
    train = _take(kept, locked["train_ids"])
    val = _take(kept, locked["val_ids"])
    test = _take(kept, locked["test_ids"])
    print("scoring ProteinMPNN", flush=True)
    mpnn = score_proteinmpnn(test)
    partial = {
        "task": "frame-edge gated sum versus ProteinMPNN native recovery",
        "proteinmpnn": {
            "checkpoint": "vanilla v_48_020",
            "soluble": False,
            "unconditional": "argmax over 20 amino acids, no amino-acid context",
            "sampled": f"{N_SAMPLES} sequences at temperature {TEMPERATURE}, seed 10000+index",
            "mean_unconditional": _mean([row["unconditional"] for row in mpnn]),
            "mean_sampled": _mean([row["sampled"] for row in mpnn]),
            "rows": mpnn,
        },
    }
    OUT.write_text(json.dumps(partial, indent=2) + "\n")
    print("recomputing frame recovery", flush=True)
    frame_rows, reproduced = recompute_frame(train, val, test, locked["runs"]["C"])
    if not reproduced:
        partial["frame_recovery"] = "not reported; validation NLL did not reproduce"
        OUT.write_text(json.dumps(partial, indent=2) + "\n")
        raise SystemExit("validation NLL drifted by more than 0.01 nat")
    by_locked = {row["id"]: row["C_nll"] for row in locked["paired"]}
    paired = []
    for i, mpnn_row in enumerate(mpnn):
        nlls = [seed[i]["nll"] for seed in frame_rows]
        recs = [seed[i]["recovery"] for seed in frame_rows]
        if abs(_mean(nlls) - by_locked[mpnn_row["id"]]) > NLL_TOL:
            partial["frame_recovery"] = "not reported; test NLL did not reproduce"
            OUT.write_text(json.dumps(partial, indent=2) + "\n")
            raise SystemExit(f"test NLL drifted on {mpnn_row['id']}")
        paired.append({
            "id": mpnn_row["id"],
            "frame": _mean(recs),
            "proteinmpnn": mpnn_row["unconditional"],
            "proteinmpnn_sampled": mpnn_row["sampled"],
            "mpnn_minus_frame": mpnn_row["unconditional"] - _mean(recs),
            "nll_drift": _mean(nlls) - by_locked[mpnn_row["id"]],
        })
    effect = cluster_interval([[row["mpnn_minus_frame"]] for row in paired])
    partial["nll_mean_abs_drift"] = _mean([abs(row["nll_drift"]) for row in paired])
    partial["frame_mean"] = _mean([row["frame"] for row in paired])
    partial["difference"] = effect
    partial["paired"] = paired
    OUT.write_text(json.dumps(partial, indent=2) + "\n")
    print(
        f"MPNN-frame {effect['point']:+.3f} [{effect['lo']:+.3f}, {effect['hi']:+.3f}] "
        f"frame {_mean([row['frame'] for row in paired]):.3f} "
        f"mpnn {partial['proteinmpnn']['mean_unconditional']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
