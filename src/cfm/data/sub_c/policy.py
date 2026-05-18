"""apply_missing_value_policy: applies the four-case (missing_value, not_in_vocab)
rule from configs/data/missing_value_policy.yaml + configs/tokenizer/vocab_phase1.yaml
to raw Overture themes.

Per spec §10.1: pure function; signature enforces non-mutation (returns NEW
themes dict). Closed handler-map registry; unknown policy types raise PolicyError.

Per spec §10.2 four-case table:
  emit_unknown_token + scalar (buildings.class, places.categories.primary):
    NULL → <prefix>__UNK__; not-in-vocab → store raw (tokenizer handles at encode)
  drop_row + scalar (transportation.class):
    NULL OR not-in-vocab → drop row
  n_a + scalar (base.class):
    NULL doesn't occur; not-in-vocab → drop row (Strict floor explicit decision)
  n_a + list (places.categories.alternate):
    Store full list raw; tokenizer filters not-in-vocab elements at encode
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import yaml

from cfm.data.sub_c.errors import PolicyError

# ---------------------------------------------------------------------------
# Vocab → kept-token set (loaded once per call)
# ---------------------------------------------------------------------------


def _load_vocab_kept_sets(vocab_yaml_path: Path) -> dict[str, set[str]]:
    """Returns map: field_name → set of kept token values (prefix stripped).

    e.g., "buildings.class" → {"residential", "house", ...} (no B_ prefix).
    Placeholders containing __UNK__ are excluded — they are not data values.
    """
    with open(vocab_yaml_path) as f:
        vocab = yaml.safe_load(f)
    sections = vocab["feature_class"]
    field_map = {
        "buildings.class": ("building", "B_"),
        "transportation.class": ("road", "R_"),
        "places.categories.primary": ("poi", "POI_"),
        "base.class": ("base", "BASE_"),
    }
    result: dict[str, set[str]] = {}
    for field, (section_name, prefix) in field_map.items():
        tokens = sections[section_name]["tokens"]
        kept: set[str] = set()
        for tok in tokens:
            if "__UNK__" in tok:
                continue  # placeholders are not data values
            assert tok.startswith(prefix), f"token {tok} missing expected prefix {prefix}"
            kept.add(tok[len(prefix) :])
        result[field] = kept
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_missing_value_policy(
    themes: dict[str, pa.Table],
    policy_yaml_path: Path,
    *,
    vocab_yaml_path: Path | None = None,
) -> dict[str, pa.Table]:
    """Returns a NEW themes dict; signature enforces non-mutation.
    Sub-A's Region object is untouched.

    The base.class not-in-vocab drop_row step removes ocean/strait/bay rows
    from the returned policied_themes — correct, because sea polygons are
    masks (not features). Sea-masking sources its sea-polygon set from the
    pre-policy derive_sea_polygons view (spec §6 + §9.1).

    Per spec §10.1 closed-set handler-map; unknown policy types raise PolicyError.
    """
    with open(policy_yaml_path) as f:
        policy = yaml.safe_load(f)

    vocab_kept = _load_vocab_kept_sets(vocab_yaml_path) if vocab_yaml_path else {}

    # Start with a shallow copy so unmentioned themes pass through unchanged
    new_themes: dict[str, pa.Table] = dict(themes)

    # Process each field per its policy
    for field_path, entry in policy["fields"].items():
        theme_name, _, _rest = field_path.partition(".")

        if theme_name == "places":
            if "places" not in new_themes:
                continue
            new_themes["places"] = _apply_places_policy(
                new_themes["places"],
                field_path,
                entry,
                vocab_kept,
            )
        elif theme_name == "buildings":
            if "buildings" not in new_themes:
                continue
            new_themes["buildings"] = _apply_scalar_policy(
                new_themes["buildings"],
                "class",
                entry,
                vocab_kept.get(field_path),
                unknown_token="B__UNK__",
            )
        elif theme_name == "transportation":
            if "transportation" not in new_themes:
                continue
            new_themes["transportation"] = _apply_scalar_policy(
                new_themes["transportation"],
                "class",
                entry,
                vocab_kept.get(field_path),
                unknown_token=None,  # drop_row policy; no unknown token
            )
        elif theme_name == "base":
            if "base" not in new_themes:
                continue
            new_themes["base"] = _apply_scalar_policy(
                new_themes["base"],
                "class",
                entry,
                vocab_kept.get(field_path),
                unknown_token=None,  # n_a + not-in-vocab drop_row; no unknown token
            )
        else:
            raise PolicyError(f"unknown theme {theme_name!r} in policy YAML")

    return new_themes


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_scalar_policy(
    table: pa.Table,
    column: str,
    entry: dict,
    kept_values: set[str] | None,
    *,
    unknown_token: str | None,
) -> pa.Table:
    """Apply (missing_value, not_in_vocab) policies for a scalar string column.

    Step 1 — missing_value:
      emit_unknown_token → coalesce(col, unknown_token)
      drop_row           → filter(is_valid(col))
      n_a                → pass through (field is 100% non-null)
      else               → PolicyError

    Step 2 — not_in_vocab:
      n_a / emit_unknown_token → pass through (tokenizer handles at encode)
      drop_row                 → keep rows whose value is in kept_values | {unknown_token}
      drop_element             → PolicyError (list-only; not applicable to scalar)
      else                     → PolicyError
    """
    mv_type = entry["policies"]["missing_value"]["type"]
    niv_entry = entry["policies"].get("not_in_vocab", {})
    niv_type = niv_entry.get("type") if niv_entry else None

    col = table.column(column)

    # ------------------------------------------------------------------
    # Step 1: missing_value handling
    # ------------------------------------------------------------------
    if mv_type == "emit_unknown_token":
        if unknown_token is None:
            raise PolicyError(
                f"emit_unknown_token policy requires an unknown_token; column={column}"
            )
        new_col = pc.coalesce(col, pa.scalar(unknown_token))
        table = table.set_column(
            table.schema.get_field_index(column),
            column,
            pa.chunked_array([new_col]),
        )
        col = table.column(column)
    elif mv_type == "drop_row":
        table = table.filter(pc.is_valid(col))
        col = table.column(column)
    elif mv_type == "n_a":
        pass  # 100% non-null; nothing to do
    else:
        raise PolicyError(f"unknown missing_value type {mv_type!r}")

    # ------------------------------------------------------------------
    # Step 2: not_in_vocab handling
    # ------------------------------------------------------------------
    if niv_type is None or niv_type == "n_a":
        pass
    elif niv_type == "emit_unknown_token":
        # Sub-C stores raw value; tokenizer maps to __UNK__ at encode time
        # (spec §10.2 Option A — cost-asymmetry benefit for Phase 1.1 expansion)
        pass
    elif niv_type == "drop_row":
        if kept_values is None:
            raise PolicyError(f"drop_row not_in_vocab requires vocab_yaml_path; column={column}")
        # Keep rows whose value is in kept_values OR is the unknown_token placeholder
        # Critical: unknown_token must survive the not_in_vocab filter so that
        # the just-coalesced placeholder is not then incorrectly dropped.
        valid_set: set[str] = set(kept_values)
        if unknown_token is not None:
            valid_set.add(unknown_token)
        if not valid_set:
            # Edge case: empty kept set → drop all rows
            table = table.filter(pa.array([False] * table.num_rows, type=pa.bool_()))
        else:
            is_kept = pc.is_in(
                table.column(column),
                value_set=pa.array(list(valid_set), type=pa.string()),
            )
            table = table.filter(is_kept)
    elif niv_type == "drop_element":
        raise PolicyError(
            f"drop_element policy is for list fields only; not applicable to scalar "
            f"column={column!r}. Use the places handler for list fields."
        )
    else:
        raise PolicyError(f"unknown not_in_vocab type {niv_type!r}")

    return table


def _apply_places_policy(
    table: pa.Table,
    field_path: str,
    entry: dict,
    vocab_kept: dict[str, set[str]],
) -> pa.Table:
    """Apply the places.categories.{primary,alternate} policies.

    primary is a scalar string field (struct sub-field); alternate is a list
    field preserved fully at sub-C (storage_policy=preserve_all per B2).

    For primary:
      missing_value=emit_unknown_token → NULL primary → "POI__UNK__"
      not_in_vocab=emit_unknown_token  → pass through (tokenizer handles)

    For alternate:
      No mutation at sub-C. Tokenizer filters not-in-vocab elements at encode.
    """
    if field_path == "places.categories.primary":
        mv_type = entry["policies"]["missing_value"]["type"]
        if mv_type == "emit_unknown_token":
            cats = table.column("categories")
            primary = pc.struct_field(cats, "primary")
            primary_filled = pc.coalesce(primary, pa.scalar("POI__UNK__"))
            alternate = pc.struct_field(cats, "alternate")
            new_categories = pa.StructArray.from_arrays(
                [primary_filled.combine_chunks(), alternate.combine_chunks()],
                names=["primary", "alternate"],
            )
            return table.set_column(
                table.schema.get_field_index("categories"),
                "categories",
                pa.chunked_array([new_categories]),
            )
        elif mv_type == "n_a":
            pass  # no nulls in primary
        elif mv_type == "drop_row":
            # Drop rows where primary is null
            cats = table.column("categories")
            primary = pc.struct_field(cats, "primary")
            table = table.filter(pc.is_valid(primary))
        else:
            raise PolicyError(f"unknown missing_value type {mv_type!r}")

    elif field_path == "places.categories.alternate":
        # Sub-C: preserve full alternate list raw per storage_policy=preserve_all
        # Tokenizer filters not-in-vocab elements at encode time (spec §10.2)
        pass

    else:
        raise PolicyError(f"unrecognised places field path {field_path!r}")

    return table
