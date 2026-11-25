#!/usr/bin/env python3
"""Find a project record whose outcomes are unusual for similar projects."""
import argparse
import csv
import math
import statistics
from pathlib import Path


TARGETS = ("Effort-", "Defects-", "MonthsX")
NUMERIC = ("Kloc+",) + TARGETS


def _median(values):
    return statistics.median(values) if values else 0.0


def _mad(values, center):
    return _median([abs(x - center) for x in values])


def _num(row, field):
    return float(row[field].strip())


def _profile(row, fields):
    return tuple(row[f].strip().lower() for f in fields)


def score_rows(rows):
    """Return rows sorted by an interpretable leave-one-out anomaly score."""
    if len(rows) < 3:
        raise ValueError("at least three records are required")
    categorical = [f for f in rows[0] if f not in NUMERIC]
    # Similarity is based on categorical project descriptors and size band.
    profiles = [_profile(r, categorical) for r in rows]
    sizes = [_num(r, "Kloc+") for r in rows]
    ranked = []
    for i, row in enumerate(rows):
        peers = [j for j in range(len(rows)) if j != i and profiles[j] == profiles[i]]
        if len(peers) < 2:
            peers = [j for j in range(len(rows)) if j != i]
        # Use the closest historical sizes, avoiding a single arbitrary peer.
        peers.sort(key=lambda j: abs(math.log1p(sizes[j]) - math.log1p(sizes[i])))
        peers = peers[:max(3, min(8, len(peers)))]
        residuals = []
        detail = {}
        for field in TARGETS:
            peer_values = [_num(rows[j], field) for j in peers]
            center = _median(peer_values)
            spread = 1.4826 * _mad(peer_values, center)
            if spread == 0:
                spread = max(abs(center) * 0.05, 1.0)
            z = abs(_num(row, field) - center) / spread
            residuals.append(z)
            detail[field] = z
        ranked.append({"row": row, "score": sum(residuals) / len(residuals), "detail": detail,
                       "peer_count": len(peers)})
    return sorted(ranked, key=lambda x: (-x["score"], int(x["row"]["idX"])))


def load(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    parser = argparse.ArgumentParser(description="spot an inconsistent NASA93 project record")
    parser.add_argument("csv_path")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)
    ranked = score_rows(load(args.csv_path))
    print("rank,id,score,effort_z,defects_z,months_z,peers")
    for item in ranked[:max(1, args.top)]:
        d = item["detail"]
        print(f"{ranked.index(item)+1},{item['row']['idX']},{item['score']:.3f},"
              f"{d['Effort-']:.3f},{d['Defects-']:.3f},{d['MonthsX']:.3f},{item['peer_count']}")
    winner = ranked[0]
    print(f"spot_record={winner['row']['idX']} anomaly_score={winner['score']:.3f}")


if __name__ == "__main__":
    main()
