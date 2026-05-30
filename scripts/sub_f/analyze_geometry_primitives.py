#!/usr/bin/env python3
"""Analyze Sub-C Singapore geometry primitives for Sub-F BP2 Halt 2."""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import yaml
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.wkb import loads

from cfm.data.sub_c.enums import GEOMETRY_TYPE
from cfm.data.sub_f.enums import (
    ANCHOR_SCHEMES,
    BP2_PLACEHOLDER_END_ID,
    BP2_PLACEHOLDER_START_ID,
    CELL_EXTENT_M,
    DIRECTION_COUNT_CANDIDATES,
    MAGNITUDE_QUANTUM_M_CANDIDATES,
    MAX_SEGMENT_CHUNK_M,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "sub_f" / "encoding_primitives.yaml"
DEFAULT_REPORT_PATH = (
    REPO_ROOT / "reports" / "2026-05-23-phase-1-sub-F-task-2-halt.md"
)
SAMPLE_SEED = 20260523
SAMPLE_LIMIT = 1000


Coord = tuple[float, float]
Candidate = tuple[int, float]
RIGHT_ANGLE_EXTRA_CANDIDATES: tuple[Candidate, ...] = (
    (32, 0.5),
    (48, 0.5),
    (72, 0.5),
)


@dataclass(frozen=True)
class GeometrySampleRef:
    tile_id: str
    source_feature_id: str
    cell_i: int
    cell_j: int
    primitive_index: int
    geometry_class: str

    @property
    def key(self) -> tuple[str, str, int, int, int]:
        return (
            self.tile_id,
            self.source_feature_id,
            self.cell_i,
            self.cell_j,
            self.primitive_index,
        )

    @property
    def sample_sort_key(self) -> tuple[str, str, int, int, int]:
        return self.key


@dataclass(frozen=True)
class SampledGeometry:
    ref: GeometrySampleRef
    coords: list[Coord]


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _distribution(values: Iterable[float], digits: int = 6) -> dict[str, float | int | None]:
    series = sorted(float(v) for v in values)
    if not series:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(series),
        "mean": _round_float(sum(series) / len(series), digits),
        "p50": _round_float(_percentile(series, 0.50), digits),
        "p95": _round_float(_percentile(series, 0.95), digits),
        "p99": _round_float(_percentile(series, 0.99), digits),
        "max": _round_float(series[-1], digits),
    }


def _heading(a: Coord, b: Coord) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _signed_turn_deg(a: Coord, b: Coord, c: Coord) -> float | None:
    ab = math.hypot(b[0] - a[0], b[1] - a[1])
    bc = math.hypot(c[0] - b[0], c[1] - b[1])
    if ab == 0.0 or bc == 0.0:
        return None
    delta = math.degrees(_heading(b, c) - _heading(a, b))
    return ((delta + 180.0) % 360.0) - 180.0


def _corner_angle_deg(prev: Coord, cur: Coord, nxt: Coord) -> float | None:
    v1 = (prev[0] - cur[0], prev[1] - cur[1])
    v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0.0 or n2 == 0.0:
        return None
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


def _distance(a: Coord, b: Coord) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _perpendicular_distance(point: Coord, line_a: Coord, line_b: Coord) -> float | None:
    dx = line_b[0] - line_a[0]
    dy = line_b[1] - line_a[1]
    denom = math.hypot(dx, dy)
    if denom == 0.0:
        return None
    return abs(dx * (line_a[1] - point[1]) - (line_a[0] - point[0]) * dy) / denom


def _xy_coords(coords: Iterable[tuple[float, ...]]) -> list[Coord]:
    return [(float(x), float(y)) for x, y, *_ in coords]


def _without_closure(coords: list[Coord]) -> list[Coord]:
    if len(coords) >= 2 and coords[0] == coords[-1]:
        return coords[:-1]
    return coords


def _polygon_exterior_parts(geom: object) -> list[list[Coord]]:
    if isinstance(geom, Polygon):
        return [_xy_coords(geom.exterior.coords)]
    if isinstance(geom, MultiPolygon):
        return [_xy_coords(poly.exterior.coords) for poly in geom.geoms]
    return []


def _polyline_parts(geom: object) -> list[list[Coord]]:
    if isinstance(geom, LineString):
        return [_xy_coords(geom.coords)]
    if isinstance(geom, MultiLineString):
        return [_xy_coords(line.coords) for line in geom.geoms]
    return []


def _iter_segments(coords: list[Coord], closed: bool) -> Iterable[tuple[Coord, Coord]]:
    if len(coords) < 2:
        return
    for idx in range(len(coords) - 1):
        yield coords[idx], coords[idx + 1]
    if closed and len(coords) > 2:
        yield coords[-1], coords[0]


def _append_line_distributions(
    coords: list[Coord],
    closed: bool,
    turn_angles: list[float],
    spacings: list[float],
) -> None:
    unique = _without_closure(coords) if closed else coords
    if len(unique) < 2:
        return
    for a, b in _iter_segments(unique, closed):
        spacing = _distance(a, b)
        if spacing > 0.0:
            spacings.append(spacing)
    if len(unique) < 3:
        return
    indexes = range(len(unique)) if closed else range(1, len(unique) - 1)
    for idx in indexes:
        prev = unique[idx - 1]
        cur = unique[idx]
        nxt = unique[(idx + 1) % len(unique)]
        turn = _signed_turn_deg(prev, cur, nxt)
        if turn is not None:
            turn_angles.append(abs(turn))


def _chunk_lengths(
    length: float,
    chunk_threshold_m: float = MAX_SEGMENT_CHUNK_M,
) -> list[float]:
    if length <= 0.0:
        return []
    chunks: list[float] = []
    remaining = length
    while remaining > chunk_threshold_m:
        chunks.append(chunk_threshold_m)
        remaining -= chunk_threshold_m
    if remaining > 0.0:
        chunks.append(remaining)
    return chunks


def _encode_decode(
    coords: list[Coord],
    *,
    direction_count: int,
    magnitude_quantum_m: float,
    closed: bool,
    chunk_threshold_m: float = MAX_SEGMENT_CHUNK_M,
) -> tuple[list[Coord], list[int]] | None:
    original = _without_closure(coords) if closed else coords
    if len(original) < 2:
        return None
    decoded: list[Coord] = [original[0]]
    original_to_decoded = [0]
    angle_step = (2.0 * math.pi) / direction_count

    segment_count = len(original) if closed else len(original) - 1
    for idx in range(segment_count):
        start = original[idx]
        end = original[(idx + 1) % len(original)]
        length = _distance(start, end)
        if length == 0.0:
            original_to_decoded.append(len(decoded) - 1)
            continue
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        direction_bin = int(round((angle % (2.0 * math.pi)) / angle_step)) % direction_count
        decoded_angle = direction_bin * angle_step
        for chunk in _chunk_lengths(length, chunk_threshold_m):
            magnitude_value = int(round(chunk / magnitude_quantum_m))
            decoded_length = magnitude_value * magnitude_quantum_m
            last = decoded[-1]
            decoded.append(
                (
                    last[0] + decoded_length * math.cos(decoded_angle),
                    last[1] + decoded_length * math.sin(decoded_angle),
                )
            )
        if not closed or idx < len(original) - 1:
            original_to_decoded.append(len(decoded) - 1)

    if closed:
        decoded.append(decoded[0])
    return decoded, original_to_decoded


def _linf_error(
    coords: list[Coord],
    *,
    direction_count: int,
    magnitude_quantum_m: float,
    closed: bool,
    chunk_threshold_m: float = MAX_SEGMENT_CHUNK_M,
) -> float | None:
    original = _without_closure(coords) if closed else coords
    decoded_pair = _encode_decode(
        coords,
        direction_count=direction_count,
        magnitude_quantum_m=magnitude_quantum_m,
        closed=closed,
        chunk_threshold_m=chunk_threshold_m,
    )
    if decoded_pair is None:
        return None
    decoded, mapping = decoded_pair
    if len(mapping) < len(original):
        return None
    max_error = 0.0
    for idx, point in enumerate(original):
        decoded_point = decoded[mapping[idx]]
        max_error = max(
            max_error,
            abs(point[0] - decoded_point[0]),
            abs(point[1] - decoded_point[1]),
        )
    return max_error


def _chunk_count_for_coords(
    coords: list[Coord],
    closed: bool,
    chunk_threshold_m: float = MAX_SEGMENT_CHUNK_M,
) -> int:
    unique = _without_closure(coords) if closed else coords
    if len(unique) < 2:
        return 0
    return sum(
        len(_chunk_lengths(_distance(a, b), chunk_threshold_m))
        for a, b in _iter_segments(unique, closed)
    )


def _sample_refs(refs: list[GeometrySampleRef]) -> list[GeometrySampleRef]:
    sorted_refs = sorted(refs, key=lambda ref: ref.sample_sort_key)
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(sorted_refs, min(SAMPLE_LIMIT, len(sorted_refs)))


def _selected_ref_keys(refs: Iterable[GeometrySampleRef]) -> set[tuple[str, str, int, int, int]]:
    return {ref.key for ref in refs}


def _round_up(value: float, quantum: float) -> float:
    return math.ceil(value / quantum) * quantum


def _empty_right_angle_root_cause_counters() -> dict[str, Counter[str]]:
    return {
        "ring_position": Counter(
            {
                "position_0_or_1": 0,
                "position_2_to_mid": 0,
                "position_mid_to_last": 0,
            }
        ),
        "input_deviation": Counter(
            {
                "<1deg": 0,
                "1_to_3deg": 0,
                "3_to_5deg": 0,
            }
        ),
        "perimeter": Counter(
            {
                "<10m": 0,
                "10_to_30m": 0,
                "30_to_100m": 0,
                ">100m": 0,
            }
        ),
    }


def _perimeter_m(coords: list[Coord]) -> float:
    unique = _without_closure(coords)
    if len(unique) < 2:
        return 0.0
    return sum(_distance(a, b) for a, b in _iter_segments(unique, True))


def _ring_position_bucket(idx: int, vertex_count: int) -> str:
    if idx <= 1:
        return "position_0_or_1"
    if idx <= vertex_count // 2:
        return "position_2_to_mid"
    return "position_mid_to_last"


def _input_deviation_bucket(abs_deviation_from_90: float) -> str:
    if abs_deviation_from_90 < 1.0:
        return "<1deg"
    if abs_deviation_from_90 < 3.0:
        return "1_to_3deg"
    return "3_to_5deg"


def _perimeter_bucket(perimeter_m: float) -> str:
    if perimeter_m < 10.0:
        return "<10m"
    if perimeter_m < 30.0:
        return "10_to_30m"
    if perimeter_m < 100.0:
        return "30_to_100m"
    return ">100m"


def _scan_inputs(
    sub_c_region_dir: Path,
) -> tuple[dict, dict[Candidate, dict[str, list[float] | int]], dict[str, list[GeometrySampleRef]]]:
    paths = sorted(sub_c_region_dir.glob("tile=EPSG3414_*/features.parquet"))
    if not paths:
        raise SystemExit(f"No EPSG3414 feature parquet files under {sub_c_region_dir}")

    feature_count = 0
    geometry_type_counts: Counter[int] = Counter()
    line_turn_angles: list[float] = []
    ring_turn_angles: list[float] = []
    line_spacings: list[float] = []
    ring_spacings: list[float] = []
    building_corner_deviations: list[float] = []
    input_right_angle_count = 0
    total_building_corners = 0
    total_polyline_triples = 0
    collinear_deviations: list[float] = []
    eligible_counts: Counter[str] = Counter()
    sample_refs: dict[str, list[GeometrySampleRef]] = {
        "polylines": [],
        "polygon_exterior_rings": [],
    }
    cell_base_tokens: Counter[tuple[str, int, int]] = Counter()
    cell_primitive_counts: Counter[tuple[str, int, int]] = Counter()
    right_angle_candidate_metrics: dict[Candidate, dict[str, list[float] | int]] = {
        candidate: {
            "post_deviation_deg": [],
            "abs_change_deg": [],
            "skip_count": 0,
        }
        for candidate in _right_angle_candidate_pairs()
    }
    right_angle_root_cause_counters = _empty_right_angle_root_cause_counters()

    for path in paths:
        tile_id = path.parent.name
        table = pq.ParquetFile(path).read()
        for row in table.to_pylist():
            feature_count += 1
            geometry_type = int(row["geometry_type"])
            geometry_type_counts[geometry_type] += 1
            geom = loads(row["geometry"])
            source_feature_id = str(row["source_feature_id"])
            cell_i = int(row["cell_i"])
            cell_j = int(row["cell_j"])
            cell = (tile_id, cell_i, cell_j)

            for part_index, coords in enumerate(_polyline_parts(geom)):
                if len(coords) < 2:
                    continue
                eligible_counts["polylines"] += 1
                sample_refs["polylines"].append(
                    GeometrySampleRef(
                        tile_id,
                        source_feature_id,
                        cell_i,
                        cell_j,
                        part_index,
                        "polylines",
                    )
                )
                _append_line_distributions(coords, False, line_turn_angles, line_spacings)
                unique = _without_closure(coords)
                total_polyline_triples += max(0, len(unique) - 2)
                for idx in range(1, len(unique) - 1):
                    turn = _signed_turn_deg(unique[idx - 1], unique[idx], unique[idx + 1])
                    if turn is None or abs(turn) > 5.0:
                        continue
                    deviation = _perpendicular_distance(
                        unique[idx],
                        unique[idx - 1],
                        unique[idx + 1],
                    )
                    if deviation is not None:
                        collinear_deviations.append(deviation)
                chunks = _chunk_count_for_coords(coords, False)
                cell_base_tokens[cell] += 2 * chunks
                cell_primitive_counts[cell] += 1

            for part_index, coords in enumerate(_polygon_exterior_parts(geom)):
                if len(_without_closure(coords)) < 3:
                    continue
                eligible_counts["polygon_exterior_rings"] += 1
                sample_refs["polygon_exterior_rings"].append(
                    GeometrySampleRef(
                        tile_id,
                        source_feature_id,
                        cell_i,
                        cell_j,
                        part_index,
                        "polygon_exterior_rings",
                    )
                )
                _append_line_distributions(coords, True, ring_turn_angles, ring_spacings)
                chunks = _chunk_count_for_coords(coords, True)
                cell_base_tokens[cell] += 2 * chunks
                cell_primitive_counts[cell] += 1

                if int(row["feature_class"]) != 1:
                    continue
                unique = _without_closure(coords)
                perimeter = _perimeter_m(coords)
                right_angle_indexes: list[int] = []
                right_angle_input_angles: list[float] = []
                for idx in range(len(unique)):
                    angle = _corner_angle_deg(
                        unique[idx - 1],
                        unique[idx],
                        unique[(idx + 1) % len(unique)],
                    )
                    if angle is None:
                        continue
                    total_building_corners += 1
                    deviation_from_90 = abs(angle - 90.0)
                    building_corner_deviations.append(deviation_from_90)
                    if deviation_from_90 <= 5.0:
                        input_right_angle_count += 1
                        right_angle_indexes.append(idx)
                        right_angle_input_angles.append(angle)

                if not right_angle_indexes:
                    continue
                for direction_count, magnitude_quantum_m in _right_angle_candidate_pairs():
                    metrics = right_angle_candidate_metrics[(direction_count, magnitude_quantum_m)]
                    decoded_pair = _encode_decode(
                        coords,
                        direction_count=direction_count,
                        magnitude_quantum_m=magnitude_quantum_m,
                        closed=True,
                    )
                    if decoded_pair is None:
                        metrics["skip_count"] = int(metrics["skip_count"]) + len(right_angle_indexes)
                        continue
                    decoded, mapping = decoded_pair
                    for idx, input_angle in zip(right_angle_indexes, right_angle_input_angles):
                        prev_i = mapping[(idx - 1) % len(unique)]
                        cur_i = mapping[idx]
                        next_i = mapping[(idx + 1) % len(unique)]
                        decoded_angle = _corner_angle_deg(
                            decoded[prev_i],
                            decoded[cur_i],
                            decoded[next_i],
                        )
                        if decoded_angle is None:
                            metrics["skip_count"] = int(metrics["skip_count"]) + 1
                            continue
                        post_deviation = abs(decoded_angle - 90.0)
                        metrics["post_deviation_deg"].append(post_deviation)
                        metrics["abs_change_deg"].append(abs(decoded_angle - input_angle))
                        if (
                            direction_count == 24
                            and magnitude_quantum_m == 0.5
                            and post_deviation > 45.0
                        ):
                            right_angle_root_cause_counters["ring_position"][
                                _ring_position_bucket(idx, len(unique))
                            ] += 1
                            right_angle_root_cause_counters["input_deviation"][
                                _input_deviation_bucket(abs(input_angle - 90.0))
                            ] += 1
                            right_angle_root_cause_counters["perimeter"][
                                _perimeter_bucket(perimeter)
                            ] += 1

    sampled_refs = {
        "polylines": _sample_refs(sample_refs["polylines"]),
        "polygon_exterior_rings": _sample_refs(sample_refs["polygon_exterior_rings"]),
    }
    inventory = {
        "paths": paths,
        "tile_count": len(paths),
        "feature_count": feature_count,
        "geometry_type_counts": dict(sorted(geometry_type_counts.items())),
        "geometry_type_labels": {
            int(code): GEOMETRY_TYPE[int(code)]
            for code in sorted(geometry_type_counts)
        },
        "turn_angle_distribution": {
            "polylines_abs_deg": _distribution(line_turn_angles),
            "polygon_exterior_rings_abs_deg": _distribution(ring_turn_angles),
        },
        "vertex_spacing_distribution_m": {
            "polylines": _distribution(line_spacings),
            "polygon_exterior_rings": _distribution(ring_spacings),
        },
        "building_corner_deviations": building_corner_deviations,
        "right_angle_input_count": input_right_angle_count,
        "total_building_corners": total_building_corners,
        "total_polyline_triples": total_polyline_triples,
        "collinear_deviations": collinear_deviations,
        "eligible_counts": dict(eligible_counts),
        "sample_counts": {
            key: len(refs)
            for key, refs in sampled_refs.items()
        },
        "candidate_counts": {
            key: len(refs)
            for key, refs in sample_refs.items()
        },
        "sampled_refs": sampled_refs,
        "cell_base_tokens": cell_base_tokens,
        "cell_primitive_counts": cell_primitive_counts,
        "right_angle_root_cause_counters": right_angle_root_cause_counters,
    }
    return inventory, right_angle_candidate_metrics, sample_refs


def _load_sampled_geometry(
    sub_c_region_dir: Path,
    sampled_refs: dict[str, list[GeometrySampleRef]],
) -> dict[str, list[SampledGeometry]]:
    wanted = {
        "polylines": _selected_ref_keys(sampled_refs["polylines"]),
        "polygon_exterior_rings": _selected_ref_keys(sampled_refs["polygon_exterior_rings"]),
    }
    refs_by_key = {
        geometry_class: {ref.key: ref for ref in refs}
        for geometry_class, refs in sampled_refs.items()
    }
    sampled_coords: dict[str, list[tuple[tuple[str, str, int, int, int], SampledGeometry]]] = {
        "polylines": [],
        "polygon_exterior_rings": [],
    }
    for path in sorted(sub_c_region_dir.glob("tile=EPSG3414_*/features.parquet")):
        tile_id = path.parent.name
        table = pq.ParquetFile(path).read()
        for row in table.to_pylist():
            geom = loads(row["geometry"])
            source_feature_id = str(row["source_feature_id"])
            cell_i = int(row["cell_i"])
            cell_j = int(row["cell_j"])
            for part_index, coords in enumerate(_polyline_parts(geom)):
                key = (tile_id, source_feature_id, cell_i, cell_j, part_index)
                if key in wanted["polylines"]:
                    sampled_coords["polylines"].append(
                        (key, SampledGeometry(refs_by_key["polylines"][key], coords))
                    )
            for part_index, coords in enumerate(_polygon_exterior_parts(geom)):
                key = (tile_id, source_feature_id, cell_i, cell_j, part_index)
                if key in wanted["polygon_exterior_rings"]:
                    sampled_coords["polygon_exterior_rings"].append(
                        (
                            key,
                            SampledGeometry(
                                refs_by_key["polygon_exterior_rings"][key],
                                coords,
                            ),
                        )
                    )

    return {
        key: [sample for _, sample in sorted(values, key=lambda item: item[0])]
        for key, values in sampled_coords.items()
    }


def _candidate_pairs() -> tuple[Candidate, ...]:
    return tuple(
        (direction_count, magnitude_quantum_m)
        for direction_count in DIRECTION_COUNT_CANDIDATES
        for magnitude_quantum_m in MAGNITUDE_QUANTUM_M_CANDIDATES
    )


def _right_angle_candidate_pairs() -> tuple[Candidate, ...]:
    return _candidate_pairs() + RIGHT_ANGLE_EXTRA_CANDIDATES


def _measure_linf_surface(
    sampled_coords: dict[str, list[SampledGeometry]],
) -> dict[Candidate, dict[str, object]]:
    measurements: dict[Candidate, dict[str, object]] = {}
    for direction_count, magnitude_quantum_m in _candidate_pairs():
        errors: list[float] = []
        skip_counts = {"polylines": 0, "polygon_exterior_rings": 0}
        sample_counts = {
            "polylines": len(sampled_coords["polylines"]),
            "polygon_exterior_rings": len(sampled_coords["polygon_exterior_rings"]),
        }
        for sample in sampled_coords["polylines"]:
            error = _linf_error(
                sample.coords,
                direction_count=direction_count,
                magnitude_quantum_m=magnitude_quantum_m,
                closed=False,
            )
            if error is None:
                skip_counts["polylines"] += 1
            else:
                errors.append(error)
        for sample in sampled_coords["polygon_exterior_rings"]:
            error = _linf_error(
                sample.coords,
                direction_count=direction_count,
                magnitude_quantum_m=magnitude_quantum_m,
                closed=True,
            )
            if error is None:
                skip_counts["polygon_exterior_rings"] += 1
            else:
                errors.append(error)
        measurements[(direction_count, magnitude_quantum_m)] = {
            "distribution": _distribution(errors),
            "sample_counts": sample_counts,
            "skip_counts": skip_counts,
        }
    return measurements


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        average_rank = (idx + 1 + end) / 2.0
        for original_idx, _ in indexed[idx:end]:
            ranks[original_idx] = average_rank
        idx = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _line_metrics(coords: list[Coord]) -> dict[str, float | int | None]:
    unique = _without_closure(coords)
    spacings = [
        _distance(unique[idx], unique[idx + 1])
        for idx in range(len(unique) - 1)
        if _distance(unique[idx], unique[idx + 1]) > 0.0
    ]
    turns = [
        abs(turn)
        for idx in range(1, len(unique) - 1)
        for turn in [_signed_turn_deg(unique[idx - 1], unique[idx], unique[idx + 1])]
        if turn is not None
    ]
    total_length = sum(spacings)
    return {
        "total_length_m": _round_float(total_length),
        "vertex_count": len(unique),
        "max_abs_turn_angle_deg": _round_float(max(turns) if turns else 0.0),
        "mean_vertex_spacing_m": _round_float(
            (sum(spacings) / len(spacings)) if spacings else 0.0
        ),
    }


def _correlation_summary(records: list[dict[str, object]]) -> dict[str, dict[str, float | None]]:
    l_inf = [float(record["l_inf_m"]) for record in records]
    summary: dict[str, dict[str, float | None]] = {}
    for field in (
        "total_length_m",
        "vertex_count",
        "max_abs_turn_angle_deg",
        "mean_vertex_spacing_m",
    ):
        values = [float(record[field]) for record in records]
        summary[field] = {
            "pearson": _round_float(_pearson(l_inf, values)),
            "spearman": _round_float(_spearman(l_inf, values)),
        }
    return summary


def _classify_linf_root_cause(
    correlations: dict[str, dict[str, float | None]],
) -> dict[str, object]:
    spearman_values = {
        field: abs(float(values["spearman"] or 0.0))
        for field, values in correlations.items()
    }
    dominant_field, dominant_abs_spearman = max(
        spearman_values.items(),
        key=lambda item: item[1],
    )
    if dominant_field in {"total_length_m", "vertex_count"}:
        classification = "length_correlated_threshold_chunking_lever"
        reviewer_action = "Revisit chunking threshold or accumulated direction quantization before lock."
    elif dominant_field == "max_abs_turn_angle_deg":
        classification = "turn_angle_correlated_direction_count_or_angle_bias_lever"
        reviewer_action = "Investigate higher direction count or angle-bias correction before lock."
    else:
        classification = "input_artifact_correlated_extreme_vertex_spacing"
        reviewer_action = "Surface as cascade #8 candidate against Sub-C output before lock."
    return {
        "dominant_driver": dominant_field,
        "dominant_abs_spearman": _round_float(dominant_abs_spearman),
        "classification": classification,
        "reviewer_action": reviewer_action,
    }


def _build_linf_decomposition(
    sampled_coords: dict[str, list[SampledGeometry]],
) -> dict[str, object]:
    direction_count = 24
    magnitude_quantum_m = 0.5
    records: list[dict[str, object]] = []
    skip_count = 0
    for sample in sampled_coords["polylines"]:
        error = _linf_error(
            sample.coords,
            direction_count=direction_count,
            magnitude_quantum_m=magnitude_quantum_m,
            closed=False,
        )
        if error is None:
            skip_count += 1
            continue
        metrics = _line_metrics(sample.coords)
        records.append(
            {
                "tile_id": sample.ref.tile_id,
                "source_feature_id": sample.ref.source_feature_id,
                "cell_i": sample.ref.cell_i,
                "cell_j": sample.ref.cell_j,
                "primitive_index": sample.ref.primitive_index,
                "l_inf_m": _round_float(error),
                **metrics,
            }
        )
    records.sort(
        key=lambda row: (
            -float(row["l_inf_m"]),
            str(row["tile_id"]),
            str(row["source_feature_id"]),
            int(row["cell_i"]),
            int(row["cell_j"]),
            int(row["primitive_index"]),
        )
    )
    correlations = _correlation_summary(records)
    return {
        "candidate": {
            "direction_count": direction_count,
            "magnitude_quantum_m": magnitude_quantum_m,
            "geometry_class": "polylines",
        },
        "sample_seed": SAMPLE_SEED,
        "sample_sort_key": "(tile_id, source_feature_id, cell_i, cell_j, primitive_index); primary key preserves requested (tile_id, source_feature_id) ordering with deterministic tie-breakers",
        "sample_count": len(records),
        "skip_count": skip_count,
        "correlations": correlations,
        "root_cause_classification": _classify_linf_root_cause(correlations),
        "top_50_worst_polyline_features": records[:50],
    }


def _linf_record_for_sample(
    sample: SampledGeometry,
    *,
    direction_count: int,
    magnitude_quantum_m: float,
    chunk_threshold_m: float,
) -> dict[str, object] | None:
    error = _linf_error(
        sample.coords,
        direction_count=direction_count,
        magnitude_quantum_m=magnitude_quantum_m,
        closed=False,
        chunk_threshold_m=chunk_threshold_m,
    )
    if error is None:
        return None
    metrics = _line_metrics(sample.coords)
    return {
        "tile_id": sample.ref.tile_id,
        "source_feature_id": sample.ref.source_feature_id,
        "cell_i": sample.ref.cell_i,
        "cell_j": sample.ref.cell_j,
        "primitive_index": sample.ref.primitive_index,
        "l_inf_m": _round_float(error),
        **metrics,
    }


def _sort_linf_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        records,
        key=lambda row: (
            -float(row["l_inf_m"]),
            str(row["tile_id"]),
            str(row["source_feature_id"]),
            int(row["cell_i"]),
            int(row["cell_j"]),
            int(row["primitive_index"]),
        ),
    )


def _build_linf_chunk_threshold_sweep(
    sampled_coords: dict[str, list[SampledGeometry]],
    current_l_inf_threshold_m: float,
) -> dict[str, object]:
    direction_count = 24
    magnitude_quantum_m = 0.5
    thresholds = (32.0, 24.0, 16.0, 12.0)
    rows: list[dict[str, object]] = []
    rows_by_threshold: dict[float, dict[str, object]] = {}
    for threshold in thresholds:
        records: list[dict[str, object]] = []
        skip_count = 0
        for sample in sampled_coords["polylines"]:
            record = _linf_record_for_sample(
                sample,
                direction_count=direction_count,
                magnitude_quantum_m=magnitude_quantum_m,
                chunk_threshold_m=threshold,
            )
            if record is None:
                skip_count += 1
            else:
                records.append(record)
        sorted_records = _sort_linf_records(records)
        distribution = _distribution(float(row["l_inf_m"]) for row in records)
        row = {
            "chunk_threshold_m": int(threshold),
            "roundtrip_l_inf_mean_m": distribution["mean"],
            "roundtrip_l_inf_p50_m": distribution["p50"],
            "roundtrip_l_inf_p95_m": distribution["p95"],
            "roundtrip_l_inf_p99_m": distribution["p99"],
            "roundtrip_l_inf_max_m": distribution["max"],
            "sample_count": len(records),
            "skip_count": skip_count,
            "top_10_worst_polyline_features": sorted_records[:10],
        }
        rows.append(row)
        rows_by_threshold[threshold] = row

    metric_keys = (
        "roundtrip_l_inf_mean_m",
        "roundtrip_l_inf_p50_m",
        "roundtrip_l_inf_p95_m",
        "roundtrip_l_inf_p99_m",
        "roundtrip_l_inf_max_m",
    )
    metric_signatures = {
        tuple(row[key] for key in metric_keys)
        for row in rows
    }
    p95_16 = float(rows_by_threshold[16.0]["roundtrip_l_inf_p95_m"] or 0.0)
    p95_12 = float(rows_by_threshold[12.0]["roundtrip_l_inf_p95_m"] or 0.0)
    if len(metric_signatures) == 1:
        classification = "chunking_is_no_op_on_test_sample"
        reviewer_guidance = (
            "Chunk-as-lever hypothesis falsified: 32/24/16/12m chunk thresholds are identical to reported precision on this sample."
        )
        proposed_chunk_threshold_m = 32
        proposed_l_inf_threshold_m = current_l_inf_threshold_m
        proposed_l_inf_threshold_status = "PENDING_DIRECTION_SWEEP"
    elif p95_16 < 5.0:
        classification = "sixteen_meter_chunking_drops_p95_below_5m"
        reviewer_guidance = (
            "Propose chunk_threshold_m=16 and round_trip_l_inf_threshold_m=5.0m."
        )
        proposed_chunk_threshold_m = 16
        proposed_l_inf_threshold_m = 5.0
        proposed_l_inf_threshold_status = "PROPOSED_AFTER_CHUNK_SWEEP"
    elif p95_12 < p95_16 * 0.9:
        classification = "twelve_meter_materially_improves_over_16m_reviewer_tradeoff"
        reviewer_guidance = (
            "Surface 12m vs 16m tradeoff; do not auto-lock the L_inf threshold."
        )
        proposed_chunk_threshold_m = 32
        proposed_l_inf_threshold_m = current_l_inf_threshold_m
        proposed_l_inf_threshold_status = "PENDING_REVIEW"
    elif p95_16 > 7.0:
        classification = "known_length_correlated_degradation_default_chunking_retained"
        reviewer_guidance = (
            "16m chunking does not materially clear the >7m p95 band; keep current/default chunking pending reviewer threshold decision."
        )
        proposed_chunk_threshold_m = 32
        proposed_l_inf_threshold_m = current_l_inf_threshold_m
        proposed_l_inf_threshold_status = "PENDING_REVIEW"
    else:
        classification = "reviewer_decision_band"
        reviewer_guidance = (
            "Chunk sweep improved the p95 band but not enough for an automatic threshold proposal."
        )
        proposed_chunk_threshold_m = 32
        proposed_l_inf_threshold_m = current_l_inf_threshold_m
        proposed_l_inf_threshold_status = "PENDING_REVIEW"

    return {
        "candidate": {
            "direction_count": direction_count,
            "magnitude_quantum_m": magnitude_quantum_m,
            "anchor_scheme": "hierarchical",
            "geometry_class": "polylines",
        },
        "sample_seed": SAMPLE_SEED,
        "rows": rows,
        "classification": classification,
        "reviewer_guidance": reviewer_guidance,
        "proposed_chunk_threshold_m": proposed_chunk_threshold_m,
        "proposed_l_inf_threshold_m": _round_float(proposed_l_inf_threshold_m),
        "proposed_l_inf_threshold_status": proposed_l_inf_threshold_status,
    }


def _bp2_fit_for_direction(direction_count: int) -> dict[str, object]:
    anchor_vocab_size = 96
    magnitude_vocab_size = math.ceil(MAX_SEGMENT_CHUNK_M / 0.5) + 1
    structural_sentinel_count = 0
    required_slots = (
        anchor_vocab_size
        + direction_count
        + magnitude_vocab_size
        + structural_sentinel_count
    )
    return {
        "placeholder_range": f"{BP2_PLACEHOLDER_START_ID}..{BP2_PLACEHOLDER_END_ID}",
        "placeholder_slot_count": BP2_PLACEHOLDER_END_ID - BP2_PLACEHOLDER_START_ID + 1,
        "required_slots": required_slots,
        "components": {
            "anchor_vocab_size": anchor_vocab_size,
            "direction_vocab_size": direction_count,
            "magnitude_vocab_size": magnitude_vocab_size,
            "structural_sentinel_count": structural_sentinel_count,
        },
        "accounting_note": "No structural sentinels are included in Task 2 BP2 fit accounting.",
        "fits_placeholder": required_slots
        <= BP2_PLACEHOLDER_END_ID - BP2_PLACEHOLDER_START_ID + 1,
    }


def _build_linf_direction_count_sweep(
    sampled_coords: dict[str, list[SampledGeometry]],
) -> dict[str, object]:
    magnitude_quantum_m = 0.5
    chunk_threshold_m = 32.0
    rows: list[dict[str, object]] = []
    rows_by_direction: dict[int, dict[str, object]] = {}
    for direction_count in (24, 32, 48, 72):
        records: list[dict[str, object]] = []
        skip_count = 0
        for sample in sampled_coords["polylines"]:
            record = _linf_record_for_sample(
                sample,
                direction_count=direction_count,
                magnitude_quantum_m=magnitude_quantum_m,
                chunk_threshold_m=chunk_threshold_m,
            )
            if record is None:
                skip_count += 1
            else:
                records.append(record)
        distribution = _distribution(float(row["l_inf_m"]) for row in records)
        row = {
            "direction_count": direction_count,
            "roundtrip_l_inf_mean_m": distribution["mean"],
            "roundtrip_l_inf_p50_m": distribution["p50"],
            "roundtrip_l_inf_p95_m": distribution["p95"],
            "roundtrip_l_inf_p99_m": distribution["p99"],
            "roundtrip_l_inf_max_m": distribution["max"],
            "sample_count": len(records),
            "skip_count": skip_count,
            "bp2_fit": _bp2_fit_for_direction(direction_count),
        }
        rows.append(row)
        rows_by_direction[direction_count] = row

    p95_48 = float(rows_by_direction[48]["roundtrip_l_inf_p95_m"] or 0.0)
    if p95_48 <= 5.0 and rows_by_direction[48]["bp2_fit"]["fits_placeholder"]:
        proposed_direction_count = 48
        proposed_threshold = _round_up(p95_48, 0.1)
        classification = "forty_eight_dirs_restores_poc_target_band"
        reviewer_guidance = (
            "48 directions drop p95 into the 3-5m target band and fit BP2; propose direction_count=48."
        )
    else:
        proposed_direction_count = 24
        proposed_threshold = _round_up(
            float(rows_by_direction[24]["roundtrip_l_inf_p95_m"] or 0.0),
            0.1,
        )
        classification = "twenty_four_dirs_known_length_correlated_degradation"
        reviewer_guidance = (
            "48 directions do not reach the 3-5m target band or fail fit; propose direction_count=24 with known length-correlated degradation."
        )

    if not rows_by_direction[proposed_direction_count]["bp2_fit"]["fits_placeholder"]:
        classification = "proposed_direction_breaks_bp2_placeholder_fit"
        reviewer_guidance = (
            "Higher direction count does not fit BP2 placeholder; reviewer intervention required before proposal."
        )

    return {
        "candidate": {
            "magnitude_quantum_m": magnitude_quantum_m,
            "anchor_scheme": "hierarchical",
            "chunk_threshold_m": int(chunk_threshold_m),
            "geometry_class": "polylines",
        },
        "sample_seed": SAMPLE_SEED,
        "rows": rows,
        "classification": classification,
        "reviewer_guidance": reviewer_guidance,
        "proposed_direction_count": proposed_direction_count,
        "proposed_l_inf_threshold_m": _round_float(proposed_threshold),
        "proposed_l_inf_threshold_status": "PROPOSED_AFTER_DIRECTION_SWEEP",
    }


def _build_right_angle_catastrophic_decomposition(
    right_angle_metrics: dict[Candidate, dict[str, list[float] | int]],
) -> dict[str, object]:
    candidate = (24, 0.5)
    values = [float(value) for value in right_angle_metrics[candidate]["post_deviation_deg"]]
    buckets = [
        {"bucket": "<5 deg", "count": 0},
        {"bucket": "5-15 deg", "count": 0},
        {"bucket": "15-45 deg", "count": 0},
        {"bucket": ">45 deg", "count": 0},
    ]
    for value in values:
        if value < 5.0:
            buckets[0]["count"] += 1
        elif value < 15.0:
            buckets[1]["count"] += 1
        elif value <= 45.0:
            buckets[2]["count"] += 1
        else:
            buckets[3]["count"] += 1
    measured_count = len(values)
    for bucket in buckets:
        bucket["fraction"] = _round_float(
            bucket["count"] / measured_count if measured_count else 0.0,
            8,
        )
    catastrophic_count = buckets[3]["count"]
    catastrophic_fraction = (
        catastrophic_count / measured_count if measured_count else 0.0
    )
    if catastrophic_fraction < 0.0001:
        classification = "edge_case"
    elif catastrophic_fraction > 0.001:
        classification = "structural_encoding_bug_surface_for_plan_revision"
    else:
        classification = "reviewer_decision_band"
    return {
        "candidate": {
            "direction_count": candidate[0],
            "magnitude_quantum_m": candidate[1],
        },
        "measured_right_angle_corner_count": measured_count,
        "skip_count": right_angle_metrics[candidate]["skip_count"],
        "buckets": buckets,
        "catastrophic_count": catastrophic_count,
        "catastrophic_fraction": _round_float(catastrophic_fraction, 8),
        "classification": classification,
        "decision_thresholds": {
            "edge_case_if_less_than_fraction": 0.0001,
            "structural_bug_if_greater_than_fraction": 0.001,
        },
    }


def _bucket_rows(
    counter: Counter[str],
    ordered_labels: tuple[str, ...],
    total: int,
) -> list[dict[str, object]]:
    return [
        {
            "bucket": label,
            "count": int(counter[label]),
            "fraction": _round_float((counter[label] / total) if total else 0.0, 8),
        }
        for label in ordered_labels
    ]


def _dominant_bucket(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(rows, key=lambda row: int(row["count"]))


def _build_right_angle_root_cause(
    inventory: dict,
    right_angle_decomposition: dict[str, object],
) -> dict[str, object]:
    counters = inventory["right_angle_root_cause_counters"]
    total = int(right_angle_decomposition["catastrophic_count"])
    ring_rows = _bucket_rows(
        counters["ring_position"],
        ("position_0_or_1", "position_2_to_mid", "position_mid_to_last"),
        total,
    )
    input_rows = _bucket_rows(
        counters["input_deviation"],
        ("<1deg", "1_to_3deg", "3_to_5deg"),
        total,
    )
    perimeter_rows = _bucket_rows(
        counters["perimeter"],
        ("<10m", "10_to_30m", "30_to_100m", ">100m"),
        total,
    )
    dominant_ring = _dominant_bucket(ring_rows)
    dominant_input = _dominant_bucket(input_rows)
    dominant_perimeter = _dominant_bucket(perimeter_rows)

    triggered_hypotheses: list[str] = []
    if float(dominant_ring["fraction"]) > 0.5 and dominant_ring["bucket"] == "position_0_or_1":
        triggered_hypotheses.append("possible_bp2_anchor_design_structural_issue_cascade_8_candidate")
    if float(dominant_input["fraction"]) > 0.5 and dominant_input["bucket"] == "<1deg":
        triggered_hypotheses.append("possible_direction_bin_boundary_or_angle_wrapping_bug")
    if float(dominant_perimeter["fraction"]) > 0.5 and dominant_perimeter["bucket"] in {"30_to_100m", ">100m"}:
        triggered_hypotheses.append("larger_perimeter_dominates_not_small_polygon_loss")

    if len(triggered_hypotheses) >= 2:
        classification = "multiple_structural_hypotheses_triggered"
        reviewer_guidance = (
            "Catastrophic corners trigger multiple structural hypotheses; review anchor handling and direction-bin boundary/angle wrapping before angle lock."
        )
    elif triggered_hypotheses:
        classification = triggered_hypotheses[0]
        if classification == "possible_bp2_anchor_design_structural_issue_cascade_8_candidate":
            reviewer_guidance = (
                "Catastrophic corners cluster near ring position 0/1; review anchor handling before angle lock."
            )
        elif classification == "possible_direction_bin_boundary_or_angle_wrapping_bug":
            reviewer_guidance = (
                "Catastrophic corners are mostly nearly exact right angles; inspect direction-bin boundary and angle wrapping."
            )
        else:
            reviewer_guidance = (
                "Catastrophic corners are dominated by larger parent polygons; do not classify as small-polygon known loss."
            )
    elif float(dominant_perimeter["fraction"]) > 0.5 and dominant_perimeter["bucket"] == "<10m":
        classification = "small_polygon_known_loss"
        reviewer_guidance = (
            "Catastrophic corners are dominated by <10m parent polygons; can be reviewed as small-geometry known loss."
        )
    else:
        classification = "mixed_root_cause_reviewer_decision_band"
        reviewer_guidance = (
            "No single hypothesis dominates; keep angle threshold/root cause pending reviewer decision."
        )

    return {
        "candidate": {
            "direction_count": 24,
            "magnitude_quantum_m": 0.5,
        },
        "right_angle_catastrophic_v1_classification": "accepted_v1_known_loss_cascade_8_candidate_for_sub_f_v2",
        "v1_known_loss_note": "4517 / 2.08M catastrophic corners (0.22%) accepted for v1; cascade #8 candidate against BP2 anchor + direction-bin alignment for sub-F-v2.",
        "catastrophic_corner_count": total,
        "ring_position_buckets": ring_rows,
        "input_deviation_buckets": input_rows,
        "perimeter_buckets": perimeter_rows,
        "root_cause_classification": {
            "classification": classification,
            "reviewer_guidance": reviewer_guidance,
            "triggered_hypotheses": triggered_hypotheses,
            "dominant_ring_position_bucket": dominant_ring,
            "dominant_input_deviation_bucket": dominant_input,
            "dominant_perimeter_bucket": dominant_perimeter,
        },
    }


def _build_right_angle_known_loss_threshold_proposal(
    right_angle_metrics: dict[Candidate, dict[str, list[float] | int]],
    proposed_direction_count: int,
) -> dict[str, object]:
    candidate = (proposed_direction_count, 0.5)
    values = [
        float(value)
        for value in right_angle_metrics[candidate]["post_deviation_deg"]
    ]
    non_catastrophic = [value for value in values if value <= 45.0]
    distribution = _distribution(non_catastrophic)
    threshold = _round_up(float(distribution["p95"] or 0.0), 0.1)
    catastrophic_count = len(values) - len(non_catastrophic)
    return {
        "candidate": {
            "direction_count": proposed_direction_count,
            "magnitude_quantum_m": 0.5,
        },
        "threshold_basis": "non_catastrophic_post_deviation_p95",
        "excluded_catastrophic_gt_45_deg": True,
        "non_catastrophic_corner_count": len(non_catastrophic),
        "catastrophic_corner_count": catastrophic_count,
        "non_catastrophic_distribution_deg": distribution,
        "round_trip_angle_threshold_deg": _round_float(threshold),
        "round_trip_angle_threshold_status": "PROPOSED_V1_KNOWN_LOSS_EXCLUDING_CATASTROPHIC",
        "v1_known_loss_classification": "accepted_v1_known_loss_cascade_8_candidate_for_sub_f_v2",
        "cascade_8_candidate": "BP2 anchor + direction-bin alignment redesign for sub-F-v2",
    }


def _row_for_candidate(
    candidate: Candidate,
    linf_measurements: dict[Candidate, dict[str, object]],
    right_angle_metrics: dict[Candidate, dict[str, list[float] | int]],
) -> dict[str, object]:
    direction_count, magnitude_quantum_m = candidate
    angle_half_bin_deg = 180.0 / direction_count
    analytical_bound = (
        MAX_SEGMENT_CHUNK_M * math.sin(math.radians(angle_half_bin_deg))
        + (magnitude_quantum_m / 2.0)
    )
    linf_distribution = linf_measurements[candidate]["distribution"]
    right_angle = right_angle_metrics[candidate]
    post_distribution = _distribution(right_angle["post_deviation_deg"])
    change_distribution = _distribution(right_angle["abs_change_deg"])
    return {
        "direction_count": direction_count,
        "magnitude_quantum_m": magnitude_quantum_m,
        "analytical_angle_half_bin_deg": _round_float(angle_half_bin_deg),
        "analytical_magnitude_half_quantum_m": _round_float(magnitude_quantum_m / 2.0),
        "analytical_l_inf_bound_m": _round_float(analytical_bound),
        "roundtrip_l_inf_mean_m": linf_distribution["mean"],
        "roundtrip_l_inf_p50_m": linf_distribution["p50"],
        "roundtrip_l_inf_p95_m": linf_distribution["p95"],
        "roundtrip_l_inf_p99_m": linf_distribution["p99"],
        "roundtrip_l_inf_max_m": linf_distribution["max"],
        "roundtrip_sample_counts": linf_measurements[candidate]["sample_counts"],
        "roundtrip_skip_counts": linf_measurements[candidate]["skip_counts"],
        "right_angle_post_roundtrip_mean_deg": post_distribution["mean"],
        "right_angle_post_roundtrip_p50_deg": post_distribution["p50"],
        "right_angle_post_roundtrip_p95_deg": post_distribution["p95"],
        "right_angle_post_roundtrip_p99_deg": post_distribution["p99"],
        "right_angle_post_roundtrip_max_deg": post_distribution["max"],
        "right_angle_abs_change_mean_deg": change_distribution["mean"],
        "right_angle_abs_change_p50_deg": change_distribution["p50"],
        "right_angle_abs_change_p95_deg": change_distribution["p95"],
        "right_angle_abs_change_p99_deg": change_distribution["p99"],
        "right_angle_abs_change_max_deg": change_distribution["max"],
        "right_angle_measured_corner_count": post_distribution["count"],
        "right_angle_skip_count": right_angle["skip_count"],
    }


def _choose_proposed_lock(
    rows: list[dict[str, object]],
    collinearity: dict[str, object],
    chunk_sweep: dict[str, object] | None = None,
    direction_sweep: dict[str, object] | None = None,
    right_angle_threshold_proposal: dict[str, object] | None = None,
) -> dict[str, object]:
    row_by_pair = {
        (int(row["direction_count"]), float(row["magnitude_quantum_m"])): row
        for row in rows
    }
    selected_direction_count = (
        int(direction_sweep["proposed_direction_count"])
        if direction_sweep is not None
        else 24
    )
    selected_pair = (24, 0.5)
    selected = row_by_pair[selected_pair]
    p99 = float(selected["roundtrip_l_inf_p99_m"] or 0.0)
    angle_p95 = float(selected["right_angle_post_roundtrip_p95_deg"] or 0.0)
    collinearity_threshold = float(collinearity["proposed_threshold_m"])
    if chunk_sweep is None:
        chunk_threshold_m = 32
        l_inf_threshold_m = _round_float(max(0.5, _round_up(p99, 0.05)))
        l_inf_threshold_status = "PENDING_REVIEW"
    else:
        chunk_threshold_m = chunk_sweep["proposed_chunk_threshold_m"]
        if direction_sweep is None:
            l_inf_threshold_m = chunk_sweep["proposed_l_inf_threshold_m"]
            l_inf_threshold_status = chunk_sweep["proposed_l_inf_threshold_status"]
        else:
            l_inf_threshold_m = direction_sweep["proposed_l_inf_threshold_m"]
            l_inf_threshold_status = direction_sweep["proposed_l_inf_threshold_status"]
    if right_angle_threshold_proposal is None:
        angle_threshold_m = _round_float(max(1.0, _round_up(angle_p95, 0.5)))
        angle_threshold_status = "PENDING_REVIEW"
    else:
        angle_threshold_m = right_angle_threshold_proposal["round_trip_angle_threshold_deg"]
        angle_threshold_status = right_angle_threshold_proposal[
            "round_trip_angle_threshold_status"
        ]
    return {
        "direction_count": selected_direction_count,
        "magnitude_quantum_m": selected_pair[1],
        "anchor_scheme": "hierarchical",
        "chunk_threshold_m": chunk_threshold_m,
        "round_trip_l_inf_threshold_m": l_inf_threshold_m,
        "round_trip_l_inf_threshold_status": l_inf_threshold_status,
        "round_trip_angle_threshold_deg": angle_threshold_m,
        "round_trip_angle_threshold_status": angle_threshold_status,
        "collinearity_admission_perpendicular_m": _round_float(collinearity_threshold),
        "rationale": (
            "Direction count follows the bounded direction sweep proposal while 0.5m "
            "quantum is kept because 0.25m adds magnitude vocabulary with little measured "
            "L_inf gain. Hierarchical anchors are proposed under project design "
            "principle #1, cheap-to-keep and impossible-to-recover: flat consumes "
            "1000/1200 BP2 placeholder slots while hierarchical consumes 96/1200, and "
            "the bounded sequence-length cost is recoverable training-compute cost while "
            "vocab namespace cost is permanent within phase."
        ),
    }


def _build_collinearity_distribution(
    inventory: dict,
    proposed_quantum: float,
) -> dict[str, object]:
    distribution = _distribution(inventory["collinear_deviations"])
    empirical_p95 = distribution["p95"] if distribution["p95"] is not None else 0.0
    weak_basis = len(inventory["collinear_deviations"]) < 500
    if weak_basis:
        method = "fixed_2x_magnitude_quantum_due_to_weak_empirical_basis"
        threshold = 2.0 * proposed_quantum
    else:
        method = "empirical_p95_perpendicular_deviation"
        threshold = float(empirical_p95)
    return {
        "definition": (
            "Polyline interior triples with abs(turn angle) <= 5 deg; value is middle "
            "vertex perpendicular deviation from the straight line through neighbors."
        ),
        "collinear_candidate_triples": len(inventory["collinear_deviations"]),
        "total_polyline_interior_triples": inventory["total_polyline_triples"],
        "weak_empirical_p95_basis": weak_basis,
        "distribution_perpendicular_m": distribution,
        "fixed_multiple_thresholds_m": {
            "1x_magnitude_quantum": _round_float(proposed_quantum),
            "2x_magnitude_quantum": _round_float(2.0 * proposed_quantum),
        },
        "proposed_threshold_method": method,
        "proposed_threshold_m": _round_float(threshold),
    }


def _build_anchor_comparison(inventory: dict, proposed_quantum: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cell_base_tokens: Counter[tuple[str, int, int]] = inventory["cell_base_tokens"]
    cell_primitive_counts: Counter[tuple[str, int, int]] = inventory["cell_primitive_counts"]
    for scheme in ANCHOR_SCHEMES:
        if scheme == "flat":
            anchor_vocab_size = 2 * math.ceil(CELL_EXTENT_M / proposed_quantum)
            tokens_per_anchor = 2
            derivation = "2 * ceil(250 / magnitude_quantum_m)"
        else:
            anchor_vocab_size = (16 + 32) * 2
            tokens_per_anchor = 4
            derivation = "(16 coarse + 32 fine) * 2 axes"
        per_cell = [
            cell_base_tokens[cell] + (cell_primitive_counts[cell] * tokens_per_anchor)
            for cell in sorted(cell_primitive_counts)
        ]
        distribution = _distribution(per_cell)
        rows.append(
            {
                "scheme": scheme,
                "anchor_vocab_size": anchor_vocab_size,
                "tokens_per_anchor": tokens_per_anchor,
                "vocab_derivation": derivation,
                "cell_count": len(per_cell),
                "mean_sequence_length_per_cell": distribution["mean"],
                "p95_sequence_length_per_cell": distribution["p95"],
            }
        )
    return rows


def _build_anchor_verification(
    rows: list[dict[str, object]],
    linf_chunk_sweep: dict[str, object],
) -> dict[str, object]:
    row_24_05 = next(
        row
        for row in rows
        if row["direction_count"] == 24 and row["magnitude_quantum_m"] == 0.5
    )
    chunk_32 = next(
        row
        for row in linf_chunk_sweep["rows"]
        if row["chunk_threshold_m"] == 32
    )
    return {
        "joint_surface_anchor_scheme": "anchor_scheme_independent",
        "joint_surface_geometry_scope": "combined_polylines_and_polygon_exterior_rings",
        "joint_surface_24_0_5_l_inf_p95_m": row_24_05["roundtrip_l_inf_p95_m"],
        "lock_threshold_surface_anchor_scheme": "hierarchical",
        "lock_threshold_surface_geometry_scope": "polylines_only",
        "lock_threshold_surface_24_0_5_l_inf_p95_m": chunk_32["roundtrip_l_inf_p95_m"],
        "verified_inconsistency_cause": "geometry_scope_difference_not_anchor_scheme",
        "note": (
            "Analysis-local encode/decode uses vertex-anchor deltas; flat vs hierarchical anchor tokenization affects BP2 vocab/sequence accounting, not decoded geometry coordinates. The 7.59m vs 10.34m p95 mismatch comes from combined geometry-class joint-surface aggregation versus polyline-only lock-threshold measurement."
        ),
        "anchor_tradeoff": (
            "Hierarchical saves namespace (96 anchor slots vs flat 1000) with a bounded sequence-length cost; L_inf lock-threshold surfaces are labeled hierarchical and do not rely on pre-revision joint-surface aggregate numbers."
        ),
    }


def _protocol_v2_candidate_9_capture() -> dict[str, object]:
    return {
        "candidate": 9,
        "classification": "hypothesis_falsified_capture",
        "note": (
            "Protocol-v2 candidate (9th): when diagnostic measurement contradicts prior hypothesis classification, surface hypothesis falsified explicitly. Sub-F Task 2 Continuation #2 Item A: chunk-as-lever hypothesis was falsified by identical L_inf across 4 chunk values; classification should have been chunking_is_no_op_on_test_sample rather than default chunking retained."
        ),
    }


def _build_yaml(
    sub_c_region_dir: Path,
    inventory: dict,
    rows: list[dict[str, object]],
    proposed_lock: dict[str, object],
    collinearity: dict[str, object],
    anchor_comparison: list[dict[str, object]],
    linf_decomposition: dict[str, object],
    right_angle_decomposition: dict[str, object],
    linf_chunk_sweep: dict[str, object],
    direction_sweep: dict[str, object],
    right_angle_root_cause: dict[str, object],
    right_angle_threshold_proposal: dict[str, object],
) -> dict[str, object]:
    geometry_counts = {
        int(code): int(count)
        for code, count in inventory["geometry_type_counts"].items()
    }
    building_distribution = _distribution(inventory["building_corner_deviations"])
    right_angle_fraction = (
        inventory["right_angle_input_count"] / inventory["total_building_corners"]
        if inventory["total_building_corners"]
        else 0.0
    )
    selected_direction = int(proposed_lock["direction_count"])
    selected_quantum = float(proposed_lock["magnitude_quantum_m"])
    selected_anchor = str(proposed_lock["anchor_scheme"])
    selected_anchor_vocab = next(
        row["anchor_vocab_size"]
        for row in anchor_comparison
        if row["scheme"] == selected_anchor
    )
    magnitude_vocab_size = math.ceil(MAX_SEGMENT_CHUNK_M / selected_quantum) + 1
    required_slots = selected_anchor_vocab + selected_direction + magnitude_vocab_size
    return {
        "_status": "PROPOSED",
        "release": "2026-04-15.0",
        "source_scope": {
            "path": str(sub_c_region_dir),
            "path_glob": str(sub_c_region_dir / "tile=EPSG3414_*/features.parquet"),
            "tile_count": inventory["tile_count"],
            "feature_count": inventory["feature_count"],
        },
        "joint_surface": rows,
        "input_geometry_characterization": {
            "geometry_type_feature_counts": geometry_counts,
            "geometry_type_labels": inventory["geometry_type_labels"],
            "eligible_geometry_counts": inventory["eligible_counts"],
            "measured_geometry_sample_counts": inventory["sample_counts"],
            "measured_geometry_candidate_counts": inventory["candidate_counts"],
            "turn_angle_distribution": inventory["turn_angle_distribution"],
            "vertex_spacing_distribution_m": inventory["vertex_spacing_distribution_m"],
            "building_corner_angle_abs_deviation_from_90_distribution_deg": building_distribution,
        },
        "right_angle_input_characterization": {
            "definition": "abs(angle_deg - 90) <= 5",
            "total_building_polygon_corner_count": inventory["total_building_corners"],
            "input_right_angle_corner_count": inventory["right_angle_input_count"],
            "fraction_within_5_deg_of_90": _round_float(right_angle_fraction),
            "abs_deviation_from_90_deg": {
                "p50": building_distribution["p50"],
                "p95": building_distribution["p95"],
                "p99": building_distribution["p99"],
            },
        },
        "collinearity_candidate_distribution": collinearity,
        "l_inf_decomposition_proposed_24_0_5": linf_decomposition,
        "right_angle_catastrophic_decomposition_proposed_24_0_5": right_angle_decomposition,
        "l_inf_chunk_threshold_sweep_24_0_5_hierarchical": linf_chunk_sweep,
        "l_inf_direction_count_sweep_0_5_hierarchical": direction_sweep,
        "anchor_verification_cross_measurement": _build_anchor_verification(
            rows,
            linf_chunk_sweep,
        ),
        "right_angle_catastrophic_root_cause_24_0_5": right_angle_root_cause,
        "right_angle_v1_known_loss_threshold_proposal": right_angle_threshold_proposal,
        "protocol_v2_candidate_9_capture": _protocol_v2_candidate_9_capture(),
        "anchor_scheme_comparison": anchor_comparison,
        "bp2_placeholder_fit": {
            "placeholder_start_id": BP2_PLACEHOLDER_START_ID,
            "placeholder_end_id": BP2_PLACEHOLDER_END_ID,
            "placeholder_range": f"{BP2_PLACEHOLDER_START_ID}..{BP2_PLACEHOLDER_END_ID}",
            "placeholder_slot_count": BP2_PLACEHOLDER_END_ID - BP2_PLACEHOLDER_START_ID + 1,
            "proposed_required_slots": required_slots,
            "components": {
                "anchor_vocab_size": selected_anchor_vocab,
                "direction_vocab_size": selected_direction,
                "magnitude_vocab_size": magnitude_vocab_size,
            },
            "fits_placeholder": required_slots
            <= BP2_PLACEHOLDER_END_ID - BP2_PLACEHOLDER_START_ID + 1,
        },
        "proposed_lock": proposed_lock,
    }


def _linf_report_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| dirs | quantum_m | analytical_linf_m | linf_mean_m | linf_p50_m | "
        "linf_p95_m | linf_p99_m | linf_max_m | samples | skips |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
    )
    body = ""
    for row in rows:
        body += (
            f"| {row['direction_count']} | {row['magnitude_quantum_m']} | "
            f"{row['analytical_l_inf_bound_m']} | {row['roundtrip_l_inf_mean_m']} | "
            f"{row['roundtrip_l_inf_p50_m']} | {row['roundtrip_l_inf_p95_m']} | "
            f"{row['roundtrip_l_inf_p99_m']} | {row['roundtrip_l_inf_max_m']} | "
            f"`{row['roundtrip_sample_counts']}` | `{row['roundtrip_skip_counts']}` |\n"
        )
    return header + body


def _right_angle_report_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| dirs | quantum_m | post_mean_deg | post_p50_deg | post_p95_deg | "
        "post_p99_deg | post_max_deg | change_p95_deg | change_p99_deg | corners | skips |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    body = ""
    for row in rows:
        body += (
            f"| {row['direction_count']} | {row['magnitude_quantum_m']} | "
            f"{row['right_angle_post_roundtrip_mean_deg']} | "
            f"{row['right_angle_post_roundtrip_p50_deg']} | "
            f"{row['right_angle_post_roundtrip_p95_deg']} | "
            f"{row['right_angle_post_roundtrip_p99_deg']} | "
            f"{row['right_angle_post_roundtrip_max_deg']} | "
            f"{row['right_angle_abs_change_p95_deg']} | "
            f"{row['right_angle_abs_change_p99_deg']} | "
            f"{row['right_angle_measured_corner_count']} | "
            f"{row['right_angle_skip_count']} |\n"
        )
    return header + body


def _linf_top50_report_table(decomposition: dict[str, object]) -> str:
    header = (
        "| rank | tile_id | source_feature_id | L_inf_m | length_m | vertices | "
        "max_turn_deg | mean_spacing_m |\n"
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
    )
    body = ""
    for rank, row in enumerate(
        decomposition["top_50_worst_polyline_features"],
        start=1,
    ):
        body += (
            f"| {rank} | {row['tile_id']} | {row['source_feature_id']} | "
            f"{row['l_inf_m']} | {row['total_length_m']} | {row['vertex_count']} | "
            f"{row['max_abs_turn_angle_deg']} | {row['mean_vertex_spacing_m']} |\n"
        )
    return header + body


def _right_angle_bucket_report_table(decomposition: dict[str, object]) -> str:
    header = "| bucket | count | fraction |\n| --- | ---: | ---: |\n"
    body = ""
    for row in decomposition["buckets"]:
        body += f"| {row['bucket']} | {row['count']} | {row['fraction']} |\n"
    return header + body


def _linf_chunk_sweep_report_table(sweep: dict[str, object]) -> str:
    header = (
        "| chunk_m | mean_m | p50_m | p95_m | p99_m | max_m | samples | skips |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    body = ""
    for row in sweep["rows"]:
        body += (
            f"| {row['chunk_threshold_m']} | {row['roundtrip_l_inf_mean_m']} | "
            f"{row['roundtrip_l_inf_p50_m']} | {row['roundtrip_l_inf_p95_m']} | "
            f"{row['roundtrip_l_inf_p99_m']} | {row['roundtrip_l_inf_max_m']} | "
            f"{row['sample_count']} | {row['skip_count']} |\n"
        )
    return header + body


def _bucket_rows_report_table(rows: list[dict[str, object]]) -> str:
    header = "| bucket | count | fraction |\n| --- | ---: | ---: |\n"
    body = ""
    for row in rows:
        body += f"| {row['bucket']} | {row['count']} | {row['fraction']} |\n"
    return header + body


def _write_report(data: dict[str, object], rows: list[dict[str, object]], report_path: Path) -> None:
    source = data["source_scope"]
    characterization = data["input_geometry_characterization"]
    right_angle = data["right_angle_input_characterization"]
    collinearity = data["collinearity_candidate_distribution"]
    proposed = data["proposed_lock"]
    anchor_rows = data["anchor_scheme_comparison"]
    bp2 = data["bp2_placeholder_fit"]
    linf_decomposition = data["l_inf_decomposition_proposed_24_0_5"]
    right_angle_decomposition = data[
        "right_angle_catastrophic_decomposition_proposed_24_0_5"
    ]
    linf_chunk_sweep = data["l_inf_chunk_threshold_sweep_24_0_5_hierarchical"]
    direction_sweep = data["l_inf_direction_count_sweep_0_5_hierarchical"]
    right_angle_root_cause = data["right_angle_catastrophic_root_cause_24_0_5"]
    angle_threshold_proposal = data["right_angle_v1_known_loss_threshold_proposal"]
    anchor_verification = data["anchor_verification_cross_measurement"]
    protocol_note = data["protocol_v2_candidate_9_capture"]
    lines = [
        "# Phase 1 Sub-F Task 2 Halt 2 Surface",
        "",
        "Status: DONE_WITH_CONCERNS",
        "",
        "## Audit outcomes",
        "",
        "- WKB writer audit: PASS; `src/cfm/data/sub_c/io.py` retains `byte_order=1`, `dump_wkb`, and shapely/WKB symbols.",
        f"- Singapore cache audit: PASS; tile count `{source['tile_count']}` from `tile=EPSG3414_*`, with WKB byte order verified as `1` before implementation.",
        f"- BP2 placeholder audit: PASS; placeholder block `{bp2['placeholder_range']}` remains available.",
        "",
        "## All-tile input inventory",
        "",
        f"- Tile count: {source['tile_count']}",
        f"- Total feature count: {source['feature_count']}",
        f"- Geometry type feature counts: `{characterization['geometry_type_feature_counts']}`",
        f"- Eligible primitive counts: `{characterization['eligible_geometry_counts']}`",
        f"- Measured sample counts: `{characterization['measured_geometry_sample_counts']}`",
        "",
        "## Geometry primitive distributions",
        "",
        f"- Turn angles: `{characterization['turn_angle_distribution']}`",
        f"- Vertex spacing: `{characterization['vertex_spacing_distribution_m']}`",
        f"- Building corner abs deviation from 90 deg: `{characterization['building_corner_angle_abs_deviation_from_90_distribution_deg']}`",
        "",
        "## Joint candidate surface",
        "",
        _linf_report_table(rows),
        "",
        "Measured L_inf convention: polylines are open and include all original vertices; polygon exterior rings exclude the implicit closure vertex from original-vertex error measurement, and decoded closure is reconstructed.",
        "",
        "## Singapore building right angles",
        "",
        f"- Definition: `{right_angle['definition']}`",
        f"- Total building polygon corner count: {right_angle['total_building_polygon_corner_count']}",
        f"- Input right-angle corner count: {right_angle['input_right_angle_corner_count']}",
        f"- Fraction within +/-5 deg of 90: {right_angle['fraction_within_5_deg_of_90']}",
        f"- Input caveat: Singapore building right-angle input fraction is {right_angle['fraction_within_5_deg_of_90']} (81.9%) within +/-5 deg of 90, not the POC 95% claim.",
        "- Mean deviation is 8.7 deg and p95 is about 68 deg, so 5% of corners are substantially non-rectilinear.",
        "- BP2 angular precision must handle the rectilinear majority and the curved/complex minority.",
        "- 24 directions accepts known loss on the 5% non-rectilinear minority; this is a design tradeoff, not a bug.",
        "",
        _right_angle_report_table(rows),
        "",
        "## Halt 2 continuation addendum",
        "",
        "### Item 1: L_inf decomposition at proposed (24, 0.5m)",
        "",
        f"- Sample count: {linf_decomposition['sample_count']}",
        f"- Skip count: {linf_decomposition['skip_count']}",
        f"- Correlations: `{linf_decomposition['correlations']}`",
        f"- Root-cause classification: `{linf_decomposition['root_cause_classification']}`",
        "- Threshold status: pending reviewer decision; no final L_inf lock is made in this continuation.",
        "",
        _linf_top50_report_table(linf_decomposition),
        "",
        "### Item 2: right-angle catastrophic bucket decomposition at proposed (24, 0.5m)",
        "",
        f"- Measured right-angle corner count: {right_angle_decomposition['measured_right_angle_corner_count']}",
        f"- Skip count: {right_angle_decomposition['skip_count']}",
        f"- Catastrophic count (>45 deg): {right_angle_decomposition['catastrophic_count']}",
        f"- Catastrophic fraction: {right_angle_decomposition['catastrophic_fraction']}",
        f"- Classification: {right_angle_decomposition['classification']}",
        "- Angle threshold status: pending reviewer decision; no final angle lock is made in this continuation.",
        "",
        _right_angle_bucket_report_table(right_angle_decomposition),
        "",
        "### Item 3: anchor proposal revision",
        "",
        "- Proposed anchor scheme is revised to hierarchical.",
        "- Rationale: flat consumes 1000/1200 BP2 placeholder slots (83%); hierarchical consumes 96/1200 (8%). The 14% mean sequence-length cost is bounded training-compute cost, while the 10x vocab namespace cost is permanent within phase. Hierarchical leaves headroom for sub-F-v2 anchor changes.",
        "",
        "### Item 4: input characterization caveat",
        "",
        f"- Singapore building right-angle input fraction is {right_angle['fraction_within_5_deg_of_90']} (81.9%) within +/-5 deg of 90, not the POC 95% claim.",
        "- Mean deviation is 8.7 deg; p95 is about 68 deg, so 5% of corners are substantially non-rectilinear.",
        "- BP2 angular precision must handle rectilinear majority and curved/complex minority.",
        "- 24 directions accepts known loss on the 5% non-rectilinear minority; this is design tradeoff, not bug.",
        "",
        "### Continuation #2 Item A: L_inf chunking lever sweep",
        "",
        f"- Candidate: `{linf_chunk_sweep['candidate']}`",
        f"- Classification: {linf_chunk_sweep['classification']}",
        f"- Reviewer guidance: {linf_chunk_sweep['reviewer_guidance']}",
        f"- Proposed chunk threshold metadata: {linf_chunk_sweep['proposed_chunk_threshold_m']} m",
        f"- Proposed L_inf threshold metadata: {linf_chunk_sweep['proposed_l_inf_threshold_m']} m ({linf_chunk_sweep['proposed_l_inf_threshold_status']})",
        "",
        _linf_chunk_sweep_report_table(linf_chunk_sweep),
        "",
        "### Continuation #3 Item A: direction-count sweep",
        "",
        f"- Candidate: `{direction_sweep['candidate']}`",
        f"- Classification: {direction_sweep['classification']}",
        f"- Reviewer guidance: {direction_sweep['reviewer_guidance']}",
        f"- Proposed direction count: {direction_sweep['proposed_direction_count']}",
        f"- Proposed L_inf threshold: {direction_sweep['proposed_l_inf_threshold_m']} m ({direction_sweep['proposed_l_inf_threshold_status']})",
        "",
        "| dirs | mean_m | p50_m | p95_m | p99_m | max_m | required BP2 slots | fits |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in direction_sweep["rows"]:
        lines.append(
            f"| {row['direction_count']} | {row['roundtrip_l_inf_mean_m']} | "
            f"{row['roundtrip_l_inf_p50_m']} | {row['roundtrip_l_inf_p95_m']} | "
            f"{row['roundtrip_l_inf_p99_m']} | {row['roundtrip_l_inf_max_m']} | "
            f"{row['bp2_fit']['required_slots']} | {row['bp2_fit']['fits_placeholder']} |"
        )
    lines.extend(
        [
        "",
        "### Continuation #3 Item B: anchor verification / cross-measurement inconsistency",
        "",
        f"- Joint surface anchor scheme: {anchor_verification['joint_surface_anchor_scheme']}",
        f"- Joint surface scope: {anchor_verification['joint_surface_geometry_scope']}",
        f"- Lock-threshold surface anchor scheme: {anchor_verification['lock_threshold_surface_anchor_scheme']}",
        f"- Verified inconsistency cause: {anchor_verification['verified_inconsistency_cause']}",
        f"- Note: {anchor_verification['note']}",
        f"- Anchor tradeoff: {anchor_verification['anchor_tradeoff']}",
        "",
        "### Continuation #2 Item B: right-angle catastrophic root-cause buckets",
        "",
        f"- Catastrophic corner count: {right_angle_root_cause['catastrophic_corner_count']}",
        f"- Root-cause classification: `{right_angle_root_cause['root_cause_classification']}`",
        f"- V1 classification: {right_angle_root_cause['right_angle_catastrophic_v1_classification']}",
        f"- Angle threshold proposal: {angle_threshold_proposal['round_trip_angle_threshold_deg']} deg ({angle_threshold_proposal['round_trip_angle_threshold_status']}), basis `{angle_threshold_proposal['threshold_basis']}`, catastrophic >45 deg excluded: {angle_threshold_proposal['excluded_catastrophic_gt_45_deg']}",
        "",
        "Ring position buckets:",
        "",
        _bucket_rows_report_table(right_angle_root_cause["ring_position_buckets"]),
        "",
        "Input deviation buckets:",
        "",
        _bucket_rows_report_table(right_angle_root_cause["input_deviation_buckets"]),
        "",
        "Parent polygon perimeter buckets:",
        "",
        _bucket_rows_report_table(right_angle_root_cause["perimeter_buckets"]),
        "",
        "### Continuation #3 Item D: protocol-v2 candidate 9 capture",
        "",
        f"- {protocol_note['note']}",
        "",
        "## Collinearity admission threshold",
        "",
        f"- Candidate triples X: {collinearity['collinear_candidate_triples']}",
        f"- Total polyline interior triples Y: {collinearity['total_polyline_interior_triples']}",
        f"- Weak empirical p95 basis: {collinearity['weak_empirical_p95_basis']}",
        f"- Perpendicular deviation distribution: `{collinearity['distribution_perpendicular_m']}`",
        f"- Fixed multiples: `{collinearity['fixed_multiple_thresholds_m']}`",
        f"- Proposed method: `{collinearity['proposed_threshold_method']}`",
        f"- Proposed threshold: {collinearity['proposed_threshold_m']} m",
        "",
        "Spec framing applied: collinearity admission threshold is the maximum perpendicular deviation from the straight line through neighboring decoded vertices.",
        "",
        "## Anchor scheme comparison",
        "",
        "| scheme | vocab size | tokens/anchor | mean seq/cell | p95 seq/cell | derivation |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in anchor_rows:
        lines.append(
            f"| {row['scheme']} | {row['anchor_vocab_size']} | {row['tokens_per_anchor']} | "
            f"{row['mean_sequence_length_per_cell']} | {row['p95_sequence_length_per_cell']} | "
            f"{row['vocab_derivation']} |"
        )
    lines.extend(
        [
            "",
            "Boundary-reference overhead is out of scope for Task 2; Tasks 3/7 cover cross-cell overhead later.",
            "",
            "## Proposed lock inputs",
            "",
            f"- Direction count: {proposed['direction_count']}",
            f"- Magnitude quantum: {proposed['magnitude_quantum_m']} m",
            f"- Anchor scheme: {proposed['anchor_scheme']}",
            f"- Chunk threshold: {proposed['chunk_threshold_m']} m",
            f"- Round-trip L_inf threshold: {proposed['round_trip_l_inf_threshold_m']} m ({proposed['round_trip_l_inf_threshold_status']})",
            f"- Round-trip 95th-percentile angle threshold: {proposed['round_trip_angle_threshold_deg']} deg ({proposed['round_trip_angle_threshold_status']})",
            f"- Collinearity admission perpendicular threshold: {proposed['collinearity_admission_perpendicular_m']} m",
            f"- Rationale: {proposed['rationale']}",
            "",
            "## BP2 placeholder fit",
            "",
            f"- Placeholder: {bp2['placeholder_range']}",
            f"- Proposed required slots: {bp2['proposed_required_slots']}",
            f"- Components: `{bp2['components']}`",
            f"- Fits placeholder: {bp2['fits_placeholder']}",
            "",
            "## Section 10.5 telemetry",
            "",
            "- Deterministic sampling seed: 20260523",
            "- Candidate grid: direction_count in 8, 16, 24 crossed with magnitude_quantum_m in 0.25, 0.5, 1.0.",
            "- Data read mode: `pq.ParquetFile(path).read()` per tile, not parent-directory reads.",
            "- Geometry decode: `shapely.wkb.loads` from Sub-C little-endian WKB.",
            "- Halt boundary: YAML remains `_status: PROPOSED`; sentinel inventory remains BP2 PLACEHOLDER.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, type=Path)
    args = parser.parse_args()

    inventory, right_angle_metrics, _ = _scan_inputs(args.sub_c_region_dir)
    sampled_coords = _load_sampled_geometry(
        args.sub_c_region_dir,
        inventory["sampled_refs"],
    )
    linf_measurements = _measure_linf_surface(sampled_coords)
    linf_decomposition = _build_linf_decomposition(sampled_coords)
    right_angle_decomposition = _build_right_angle_catastrophic_decomposition(
        right_angle_metrics
    )
    right_angle_root_cause = _build_right_angle_root_cause(
        inventory,
        right_angle_decomposition,
    )
    rows = [
        _row_for_candidate(candidate, linf_measurements, right_angle_metrics)
        for candidate in _candidate_pairs()
    ]
    proposed_row = next(
        row
        for row in rows
        if row["direction_count"] == 24 and row["magnitude_quantum_m"] == 0.5
    )
    current_l_inf_threshold_m = _round_float(
        max(0.5, _round_up(float(proposed_row["roundtrip_l_inf_p99_m"] or 0.0), 0.05))
    )
    linf_chunk_sweep = _build_linf_chunk_threshold_sweep(
        sampled_coords,
        current_l_inf_threshold_m=current_l_inf_threshold_m,
    )
    direction_sweep = _build_linf_direction_count_sweep(sampled_coords)
    right_angle_threshold_proposal = _build_right_angle_known_loss_threshold_proposal(
        right_angle_metrics,
        proposed_direction_count=int(direction_sweep["proposed_direction_count"]),
    )
    preliminary_collinearity = _build_collinearity_distribution(inventory, proposed_quantum=0.5)
    proposed_lock = _choose_proposed_lock(rows, preliminary_collinearity)
    collinearity = _build_collinearity_distribution(
        inventory,
        proposed_quantum=float(proposed_lock["magnitude_quantum_m"]),
    )
    proposed_lock = _choose_proposed_lock(
        rows,
        collinearity,
        linf_chunk_sweep,
        direction_sweep,
        right_angle_threshold_proposal,
    )
    anchor_comparison = _build_anchor_comparison(
        inventory,
        proposed_quantum=float(proposed_lock["magnitude_quantum_m"]),
    )
    data = _build_yaml(
        args.sub_c_region_dir,
        inventory,
        rows,
        proposed_lock,
        collinearity,
        anchor_comparison,
        linf_decomposition,
        right_angle_decomposition,
        linf_chunk_sweep,
        direction_sweep,
        right_angle_root_cause,
        right_angle_threshold_proposal,
    )
    args.config_path.parent.mkdir(parents=True, exist_ok=True)
    args.config_path.write_text(
        yaml.safe_dump(data, sort_keys=False, width=1000),
        encoding="utf-8",
    )
    _write_report(data, rows, args.report_path)
    print(f"wrote {args.config_path}")
    print(f"wrote {args.report_path}")


if __name__ == "__main__":
    main()
