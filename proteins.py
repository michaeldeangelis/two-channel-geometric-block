"""Small heterodimers from the PDB. No amino-acid features are stored as inputs.

A complex is kept when it has exactly two protein chains, each 40–160 residues,
and at least eight cross-chain CA contacts within 8 Å. Homologs are clustered
if either chain matches either chain of another complex at 30% identity or more,
measured as the best ungapped window over the shorter chain.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

ROOT = Path(__file__).resolve().parent / "data"
PDB_DIR = ROOT / "pdb"
INTERFACE_A = 8.0
MIN_LEN = 40
MAX_LEN = 160


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "two-channel-block"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def search_entries(n: int = 60, start: int = 0) -> list[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "equals",
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "range",
                        "value": {"from": 90, "to": 320},
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "range",
                        "value": {"from": "2012-01-01", "to": "2022-06-01"},
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": n},
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}],
        },
    }
    body = json.dumps(query).encode()
    req = urllib.request.Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "two-channel-block"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    return [hit["identifier"] for hit in payload.get("result_set", [])]


def parse_pdb(text: str) -> dict | None:
    """Two chains of CA coordinates and sequences. Amino acids are labels only."""
    residues: dict[tuple[str, int], dict] = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        name = line[12:16].strip()
        if name != "CA":
            continue
        resn = line[17:20].strip()
        if resn not in AA3:
            continue
        chain = line[21]
        resi = int(line[22:26])
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        residues[(chain, resi)] = {"aa": AA3[resn], "xyz": xyz}
    chains: dict[str, list] = {}
    for (chain, resi), rec in residues.items():
        chains.setdefault(chain, []).append((resi, rec))
    if len(chains) != 2:
        return None
    built = []
    for chain, items in chains.items():
        items.sort()
        if not MIN_LEN <= len(items) <= MAX_LEN:
            return None
        built.append({
            "chain": chain,
            "seq": "".join(rec["aa"] for _, rec in items),
            "xyz": [rec["xyz"] for _, rec in items],
            "resseq": [float(resi) for resi, _ in items],
        })
    return {"chains": built}


def _contacts(a: list[tuple[float, float, float]], b: list[tuple[float, float, float]]) -> int:
    n = 0
    for x, y, z in a:
        for u, v, w in b:
            if (x - u) ** 2 + (y - v) ** 2 + (z - w) ** 2 <= INTERFACE_A ** 2:
                n += 1
                break
    return n


def identity(a: str, b: str) -> float:
    if len(a) > len(b):
        a, b = b, a
    best = 0
    span = len(b) - len(a) + 1
    for shift in range(span):
        window = b[shift : shift + len(a)]
        matches = sum(x == y for x, y in zip(a, window))
        if matches > best:
            best = matches
    return best / len(a)


def homologous(left: dict, right: dict, cutoff: float = 0.30) -> bool:
    seqs_l = [c["seq"] for c in left["chains"]]
    seqs_r = [c["seq"] for c in right["chains"]]
    return any(identity(a, b) >= cutoff for a in seqs_l for b in seqs_r)


def cluster(complexes: list[dict]) -> list[list[int]]:
    parent = list(range(len(complexes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(complexes)):
        for j in range(i):
            if homologous(complexes[i], complexes[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(complexes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# Stage 3's test complexes chose the architecture. They stay out of later val/test.
DEV_IDS = ("7OWD", "7S6O", "7MIC", "7QGS", "7LVS")


def split_clusters(clusters: list[list[int]], test_frac: float = 0.25) -> dict:
    ordered = sorted(clusters, key=len, reverse=True)
    n = sum(len(g) for g in ordered)
    target = max(1, int(round(n * test_frac)))
    test: list[int] = []
    train: list[int] = []
    for group in ordered:
        if len(test) < target:
            test.extend(group)
        else:
            train.extend(group)
    if not train:
        train, test = test[:-1], test[-1:]
    return {"train": train, "test": test}


def split_three(
    complexes: list[dict],
    clusters: list[list[int]],
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> dict:
    """Whole clusters only. Stage 3 test ids are forced into train.

    Validation and test are filled with the smallest remaining clusters so
    one large family does not become the entire held-out set.
    """
    dev = set(DEV_IDS)
    train: list[int] = []
    flexible: list[list[int]] = []
    for group in clusters:
        if any(complexes[i]["id"] in dev for i in group):
            train.extend(group)
        else:
            flexible.append(group)
    n = len(complexes)
    target_val = max(1, int(round(n * val_frac)))
    target_test = max(1, int(round(n * test_frac)))
    val: list[int] = []
    test: list[int] = []
    for group in sorted(flexible, key=len):
        if len(test) < target_test:
            test.extend(group)
        elif len(val) < target_val:
            val.extend(group)
        else:
            train.extend(group)
    if not train or not val or not test:
        raise RuntimeError(f"split failed train={len(train)} val={len(val)} test={len(test)}")
    return {"train": train, "val": val, "test": test}


def max_cross_identity(left: dict, right: dict) -> float:
    seqs_l = [c["seq"] for c in left["chains"]]
    seqs_r = [c["seq"] for c in right["chains"]]
    return max(identity(a, b) for a in seqs_l for b in seqs_r)


def audit_partitions(complexes: list[dict], split: dict, cutoff: float = 0.30) -> dict:
    """Homology must not cross train, val, and test, in either chain order."""
    violations = []
    worst = 0.0
    names = ("train", "val", "test")
    for a_i, a in enumerate(names):
        for b in names[a_i + 1 :]:
            for i in split[a]:
                for j in split[b]:
                    left = complexes[i]
                    right = complexes[j]
                    swapped = {"chains": list(reversed(right["chains"]))}
                    direct = max_cross_identity(left, right)
                    flipped = max_cross_identity(left, swapped)
                    link = max(direct, flipped)
                    if link > worst:
                        worst = link
                    if link >= cutoff:
                        violations.append({
                            "left": left["id"],
                            "right": right["id"],
                            "parts": [a, b],
                            "identity": round(link, 4),
                        })
    dev_leaked = [
        complexes[i]["id"]
        for part in ("val", "test")
        for i in split[part]
        if complexes[i]["id"] in DEV_IDS
    ]
    return {
        "cutoff": cutoff,
        "worst_cross_identity": worst,
        "violations": violations,
        "dev_ids_in_val_or_test": dev_leaked,
        "passed": not violations and not dev_leaked,
    }


def load_cache() -> list[dict]:
    path = ROOT / "complexes.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def build_cache(limit: int = 24, update_split: bool = True) -> list[dict]:
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    start = 0
    page = 200
    seen = 0
    while len(kept) < limit:
        ids = search_entries(page, start)
        print(f"search page {start} returned {len(ids)}", flush=True)
        if not ids:
            break
        start += len(ids)
        for pdb_id in ids:
            seen += 1
            if len(kept) >= limit:
                break
            path = PDB_DIR / f"{pdb_id}.pdb"
            if not path.exists():
                try:
                    path.write_bytes(_fetch(f"https://files.rcsb.org/download/{pdb_id}.pdb"))
                except Exception:
                    continue
            parsed = parse_pdb(path.read_text(errors="ignore"))
            if parsed is None:
                continue
            contacts = _contacts(parsed["chains"][0]["xyz"], parsed["chains"][1]["xyz"])
            back = _contacts(parsed["chains"][1]["xyz"], parsed["chains"][0]["xyz"])
            if contacts + back < 8:
                continue
            parsed["id"] = pdb_id
            parsed["interface_contacts"] = contacts + back
            kept.append(parsed)
            print(f"kept {len(kept)}/{limit} {pdb_id} contacts {contacts + back} after {seen}", flush=True)
        if len(ids) < page:
            break
    (ROOT / "complexes.json").write_text(json.dumps(kept))
    if update_split:
        groups = cluster(kept)
        (ROOT / "split.json").write_text(json.dumps(split_clusters(groups), indent=2))
    return kept
