#!/usr/bin/env python3
"""Convert the real-features YAML reference -> long-format parquet keyed by
(role, city, metric, stratum-4-tuple, sample), + the MANDATORY floor-reproduction teeth on
the NEW format.

A format change is a NEW artifact — the YAML's floor-reproduction proof does NOT transfer, so
this REBUILDS the 265 floor_all rows from the PARQUET and confirms they match the sha-locked
floor (95abb88…) exactly. If they do not -> the parquet does not faithfully carry what decide
consumes; FLOOR_REPRODUCTION_FROM_PARQUET_OK=False (HALT).

The 665 MB YAML is slow to write/reload; parquet (zstd, nullable-int stratum columns) is the
columnar form the wired scored eval should consume. Run on Leonardo (the YAML lives on $WORK).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402

from cfm.eval.conditioning_floor import (  # noqa: E402
    build_floor_artifact_payload,
    load_verified_floor,
)

logger = logging.getLogger(__name__)
_RELEASE = "2026-04-15.0"
_YAML = _REPO / "reports/phase-2-bakeoff" / f"real-features-{_RELEASE}.yaml"
_PARQUET = _REPO / "reports/phase-2-bakeoff" / f"real-features-{_RELEASE}.parquet"
_LOCKED = _REPO / "reports/conditioning_floor" / _RELEASE / "conditioning-floor.yaml"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("reading YAML %s (large; ~minutes)", _YAML)
    data = yaml.safe_load(_YAML.read_text(encoding="utf-8"))

    roles: list[str] = []
    cities: list[str] = []
    metrics: list[str] = []
    cols: list[list] = [[], [], [], []]  # s0..s3 (zoning, road_skeleton, density, coastal)
    samples: list[float] = []

    def _add(role: str, by_city: dict) -> None:
        for city, recs in by_city.items():
            for rec in recs:
                st = rec["stratum"]
                if len(st) != 4:
                    raise ValueError(f"{role}/{city}: stratum {st!r} is not a 4-tuple")
                for v in rec["samples"]:
                    roles.append(role)
                    cities.append(city)
                    metrics.append(rec["metric"])
                    for k in range(4):
                        cols[k].append(st[k])  # int or None -> nullable int column
                    samples.append(float(v))

    _add("held_out", data["real_by_city"])
    _add("train", data["real_train_by_city"])
    logger.info("flattened %d sample rows", len(samples))

    table = pa.table(
        {
            "role": roles,
            "city": cities,
            "metric": metrics,
            "s0": pa.array(cols[0], pa.int64()),
            "s1": pa.array(cols[1], pa.int64()),
            "s2": pa.array(cols[2], pa.int64()),
            "s3": pa.array(cols[3], pa.int64()),
            "sample": pa.array(samples, pa.float64()),
        }
    )
    pq.write_table(table, _PARQUET, compression="zstd")
    logger.info(
        "wrote parquet %s bytes=%d (vs YAML %d)",
        _PARQUET,
        _PARQUET.stat().st_size,
        _YAML.stat().st_size,
    )

    # ---- MANDATORY: floor reproduction FROM THE PARQUET ----
    t = pq.read_table(_PARQUET).to_pydict()
    feats: dict[tuple, list[float]] = {}
    for i in range(len(t["sample"])):
        stratum = tuple(t[f"s{k}"][i] for k in range(4))
        feats.setdefault((t["city"][i], stratum, t["metric"][i]), []).append(t["sample"][i])
    logger.info("reconstructed %d (city,stratum,metric) cells from parquet", len(feats))

    locked = load_verified_floor(_LOCKED)
    held = list(locked.payload["held_out_cities"])
    train = list(locked.payload["train_cities"])
    rebuilt = build_floor_artifact_payload(
        feats, release=_RELEASE, held_out_cities=held, train_cities=train,
        min_n=50, alpha=0.05, delta=0.15,
    )

    def _fm(floors: list[dict]) -> dict:
        return {(r["city"], r["metric"], tuple(r["stratum"])): r["floor_all"] for r in floors}

    fn, fl = _fm(rebuilt["floors"]), _fm(locked.payload["floors"])
    shared = set(fn) & set(fl)
    mism = [k for k in shared if fn[k] != fl[k]]
    exact = sum(1 for k in shared if fn[k] == fl[k])
    ok = set(fn) == set(fl) and not mism
    logger.info(
        "FLOOR REPRO from PARQUET: rebuilt=%d locked=%d exact=%d mismatch=%d "
        "only_rebuilt=%d only_locked=%d",
        len(fn), len(fl), exact, len(mism), len(set(fn) - set(fl)), len(set(fl) - set(fn)),
    )
    for k in mism[:8]:
        logger.info("  MISMATCH %s rebuilt=%r locked=%r", k, fn[k], fl[k])
    logger.info(
        "FLOOR_REPRODUCTION_FROM_PARQUET_OK=%s locked_sha=%s",
        ok,
        locked.payload["floor_sha256"],
    )
    print(f"PARQUET_VERIFY_DONE ok={ok}")


if __name__ == "__main__":
    main()
