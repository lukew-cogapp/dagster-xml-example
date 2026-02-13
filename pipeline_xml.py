"""Dagster XML example — simulated Dagster asset graph.

Each top-level function represents a Dagster asset. Parameters simulate
upstream dependencies (what the IO manager would inject). Returns simulate
what the IO manager would persist to Parquet.

Asset graph:
    harvest_terminology ─┐
                         ├─→ objects_transform ──→ objects_output
    harvest_objects ─────┘

    check_objects_transform (asset check, blocking)

Data quality issues planted in the source XML:
    - OBJ-003: missing date_made, empty constituents
    - OBJ-005: dimension value of 0.5 (should be > 1)

Usage:
    uv run python pipeline_xml.py
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandera.polars as pa
import polars as pl
import xmltodict

from schema import object_transform_schema

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


# =========================================================================
# HARVEST ASSETS — XML → Parquet (close to source format)
# =========================================================================


def harvest_terminology() -> pl.DataFrame:
    """@dg.asset(kinds={"xml", "polars"}, group_name="harvest")

    Parse terminology.xml into a flat lookup DataFrame.
    """
    term_path = DATA_DIR / "xml" / "terminology.xml"
    _validate_xml([term_path])

    with open(term_path) as f:
        record = _harvest_xml_record(f.read(), root="terminology")

    return pl.DataFrame(record["term"]).rename(
        {"id": "term_id", "type": "term_type", "text": "label"}
    )


def harvest_objects() -> pl.DataFrame:
    """@dg.asset(kinds={"xml", "polars"}, group_name="harvest")

    Parse all object XML files into a DataFrame with native nested types.
    Nested XML elements become List(Struct(...)) columns — no flattening.
    Harvest stays close to source: term_ids are NOT resolved here.
    """
    xml_dir = DATA_DIR / "xml" / "objects"
    paths = sorted(xml_dir.glob("*.xml"))
    _validate_xml(paths)

    records = []
    for p in paths:
        with open(p) as f:
            records.append(_harvest_xml_record(f.read(), root="object"))

    return pl.DataFrame(records)


# =========================================================================
# TRANSFORM ASSET — Enrich with terminology joins (lazy API)
# =========================================================================


def objects_transform(
    harvest_terminology: pl.DataFrame,
    harvest_objects: pl.DataFrame,
) -> pl.DataFrame:
    """@dg.asset(kinds={"polars"}, group_name="transform")

    Enrich objects with terminology labels via Polars joins.
    Uses lazy API throughout — single collect(engine="streaming") at the end.

    Selectively flattens only the columns that need enrichment:
    - classifications: explode → join term_id → re-nest
    - constituents:    explode → join nationality_id → re-nest
    - dimensions, media: pass-through (never flattened)
    """
    terminology = harvest_terminology.lazy()
    objects = harvest_objects.lazy()

    # --- Normalize column names across source formats (XML vs JSON) ---
    col_renames = {"id": "object_id", "classifications": "classification_ids"}
    objects = objects.rename(
        {old: new for old, new in col_renames.items() if old in harvest_objects.columns}
    )

    # --- Enrich classifications: explode → join → re-nest ---
    type_labels = terminology.select(
        pl.col("term_id").alias("type_id"),
        pl.col("label").alias("type_label"),
    )
    term_labels = terminology.select(
        "term_id",
        pl.col("label").alias("term_label"),
    )

    classifications_flat = (
        objects.select("object_id", "classification_ids")
        .filter(pl.col("classification_ids").list.len() > 0)
        .explode("classification_ids")
        .unnest("classification_ids")
    )

    classifications_nested = (
        classifications_flat.join(type_labels, on="type_id", how="left")
        .join(term_labels, on="term_id", how="left")
        .select("object_id", "type_label", "term_label")
        .group_by("object_id")
        .agg(pl.struct("type_label", "term_label").alias("classifications"))
    )

    # --- Enrich constituents: explode → join nationality → re-nest ---
    nationality_lookup = terminology.filter(
        pl.col("term_type") == "nationality"
    ).select(
        pl.col("term_id").alias("nationality_id"),
        pl.col("label").alias("nationality"),
    )

    constituents_flat = (
        objects.select("object_id", "constituents")
        .filter(pl.col("constituents").list.len() > 0)
        .explode("constituents")
        .unnest("constituents")
    )

    constituents_nested = (
        constituents_flat.join(nationality_lookup, on="nationality_id", how="left")
        .select("object_id", "name", "role", "birth_year", "nationality")
        .group_by("object_id")
        .agg(
            pl.struct("name", "role", "birth_year", "nationality").alias("constituents")
        )
    )

    # --- Assemble: flat fields + pass-through nested + enriched nested ---
    enriched = (
        objects.select(
            "object_id",
            "title",
            "date_made",
            "credit_line",
            "department",
            "dimensions",
            "media",
        )
        .join(classifications_nested, on="object_id", how="left")
        .join(constituents_nested, on="object_id", how="left")
        .with_columns(
            pl.col("classifications").fill_null([]),
            pl.col("constituents").fill_null([]),
        )
    )

    result = enriched.collect(engine="streaming")
    assert isinstance(result, pl.DataFrame)
    return result


# =========================================================================
# ASSET CHECK — Pandera validation (blocking)
# =========================================================================


def check_objects_transform(
    objects_transform: pl.DataFrame,
) -> tuple[bool, list[dict]]:
    """@dg.asset_check(asset=objects_transform, blocking=True)

    Validate transform output with Pandera. Blocks objects_output if failed.
    """
    errors = []
    for col_name, column in object_transform_schema.columns.items():
        try:
            pa.DataFrameSchema({col_name: column}).validate(
                objects_transform, lazy=False
            )
        except pa.errors.SchemaError as e:
            errors.append({"column": col_name, "check": str(e.check)})
    return (True, []) if not errors else (False, errors)


# =========================================================================
# OUTPUT ASSET — Nested JSON for Elasticsearch
# =========================================================================


def objects_output(
    objects_transform: pl.DataFrame,
) -> Path:
    """@dg.asset(kinds={"json"}, group_name="output")

    Write enriched objects as nested JSON for Elasticsearch bulk indexing.
    """
    output_path = OUTPUT_DIR / "xml" / "objects.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = objects_transform.to_dicts()
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    return output_path


# =========================================================================
# XML HELPERS
# =========================================================================


def _validate_xml(paths: list[Path]) -> None:
    """Check that all XML files are well-formed. Raises ValueError listing every bad file."""
    errors = []
    for p in paths:
        try:
            ET.parse(p)
        except ET.ParseError as e:
            errors.append(f"  {p.name}: {e}")
    if errors:
        raise ValueError(
            f"Malformed XML ({len(errors)} file(s)):\n" + "\n".join(errors)
        )


def _harvest_xml_record(xml_text: str, root: str) -> dict:  # type: ignore[type-arg]
    """Parse a single XML document into a clean dict with auto-unwrapped lists.

    Handles the full xmltodict cleanup pipeline:
    1. Parse with force_list=True so every element is a list
    2. Strip @ prefixes, rename #text → text, auto-cast numbers
    3. Auto-unwrap wrapper elements (e.g. {constituents: {constituent: [...]}} → {constituents: [...]})
    4. Collapse single-element lists for scalar fields (e.g. ["text"] → "text")
    """
    raw: dict = xmltodict.parse(xml_text, force_list=True)  # type: ignore[assignment]
    cleaned: dict = _clean_xmltodict(raw)  # type: ignore[assignment]
    # force_list wraps root in a list too — unwrap it
    record = cleaned[root]
    if isinstance(record, list) and len(record) == 1:
        record = record[0]
    result = _auto_unwrap(record)
    assert isinstance(result, dict)
    return result


def _auto_unwrap(obj: object) -> object:
    """Recursively unwrap xmltodict's wrapper patterns.

    - Single-key dicts whose value is a list → just the list
      ({constituent: [{...}, {...}]} → [{...}, {...}])
    - Single-element lists containing a scalar → the scalar
      (["hello"] → "hello", [42] → 42)
    - None / empty string from empty elements → [] for wrapper positions
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v = _auto_unwrap(v)
            # Unwrap wrapper elements: if value is a dict with a single key
            # whose value is a list, it's the xmltodict wrapper pattern
            if isinstance(v, dict) and len(v) == 1:
                inner_val = next(iter(v.values()))
                if isinstance(inner_val, list):
                    v = inner_val
            # Empty elements (e.g. <constituents/>) parsed as None
            if v is None:
                v = []
            cleaned[k] = v
        return cleaned
    if isinstance(obj, list):
        unwrapped = [_auto_unwrap(i) for i in obj]
        if len(unwrapped) == 1:
            item = unwrapped[0]
            # Collapse wrapper: [{"child_key": [...]}] → [...]
            # This is the xmltodict pattern where force_list wraps a parent
            # element that itself contains a single repeating child element.
            if isinstance(item, dict) and len(item) == 1:
                inner_val = next(iter(item.values()))
                if isinstance(inner_val, list):
                    return inner_val
            # Collapse scalar: ["hello"] → "hello"
            if not isinstance(item, (dict, list)):
                return item
        return unwrapped
    return obj


def _clean_xmltodict(obj: object) -> object:
    """Recursively strip @ from keys, rename #text → text, auto-cast numeric strings."""
    if isinstance(obj, dict):
        return {
            ("text" if k == "#text" else k.lstrip("@")): _clean_xmltodict(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_clean_xmltodict(i) for i in obj]
    if isinstance(obj, str):
        try:
            return int(obj)
        except ValueError:
            try:
                return float(obj)
            except ValueError:
                return obj
    return obj


# =========================================================================
# MAIN — Simulate Dagster materialisation sequence
# =========================================================================


def main() -> None:
    print("=" * 70)
    print("DAGSTER XML EXAMPLE")
    print("Simulated Dagster asset graph: XML → Polars → Parquet → JSON")
    print("=" * 70)

    # --- Harvest ---
    print("\n--- harvest_terminology ---")
    terminology_df = harvest_terminology()
    print(f"  {len(terminology_df)} terms → Parquet")

    print("\n--- harvest_objects ---")
    objects_df = harvest_objects()
    print(
        f"  {len(objects_df)} objects from {len(list((DATA_DIR / 'xml' / 'objects').glob('*.xml')))} XML files"
    )
    print(
        f"  Nested columns: {[n for n, t in objects_df.schema.items() if 'List' in str(t)]}"
    )

    # Simulate IO manager
    harvest_dir = OUTPUT_DIR / "xml" / "harvest"
    harvest_dir.mkdir(parents=True, exist_ok=True)
    terminology_df.write_parquet(harvest_dir / "terminology.parquet")
    objects_df.write_parquet(harvest_dir / "objects.parquet")

    # --- Transform ---
    print("\n--- objects_transform (lazy → streaming collect) ---")
    transform_df = objects_transform(terminology_df, objects_df)
    print(f"  {len(transform_df)} enriched objects → Parquet")

    # Show one enriched record
    row = transform_df.filter(pl.col("object_id") == "OBJ-001")
    classifications = (
        row.select("classifications")
        .explode("classifications")
        .unnest("classifications")
    )
    print(f"  OBJ-001 classifications: {classifications['term_label'].to_list()}")

    constituents = (
        row.select("constituents").explode("constituents").unnest("constituents")
    )
    print(f"  OBJ-001 constituents: {constituents['name'].to_list()}")

    transform_dir = OUTPUT_DIR / "xml" / "transform"
    transform_dir.mkdir(parents=True, exist_ok=True)
    transform_df.write_parquet(transform_dir / "objects_enriched.parquet")

    # Verify Parquet round-trip
    reloaded = pl.read_parquet(transform_dir / "objects_enriched.parquet")
    assert transform_df.schema == reloaded.schema, "Parquet round-trip schema mismatch!"
    print("  Parquet round-trip: schema preserved")

    # --- Validate ---
    print("\n--- check_objects_transform (schema validation) ---")
    passed, errors = check_objects_transform(transform_df)
    if passed:
        print("  All checks PASSED")
    else:
        # Deduplicate: Pandera reports cascading errors for list columns
        seen = set()
        unique_errors = []
        for err in errors:
            key = (err["column"], err["check"])
            if key not in seen:
                seen.add(key)
                unique_errors.append(err)

        print(f"  FAILED — {len(unique_errors)} check(s):")
        for err in unique_errors:
            print(f"    [{err['column']}] {err['check']}")

        # Show the actual bad records
        missing_constituents = transform_df.filter(
            pl.col("constituents").list.len() == 0
        )
        if len(missing_constituents) > 0:
            ids = missing_constituents["object_id"].to_list()
            print(f"\n  Missing constituents: {ids}")

        bad_dimensions = transform_df.filter(
            pl.col("dimensions")
            .list.eval(pl.element().struct.field("value"))
            .list.min()
            <= 1
        )
        if len(bad_dimensions) > 0:
            for row_dict in bad_dimensions.to_dicts():
                dims = row_dict["dimensions"]
                bad_vals = [d["value"] for d in dims if d["value"] <= 1]
                print(f"  Bad dimension in {row_dict['object_id']}: {bad_vals}")

        print("\n  (In Dagster, blocking=True would prevent objects_output)")

    # --- Output ---
    print("\n--- objects_output ---")
    output_path = objects_output(transform_df)
    print(f"  {len(transform_df)} records → {output_path.name}")

    # Show one complete record
    sample = transform_df.filter(pl.col("object_id") == "OBJ-004").to_dicts()[0]
    print("\n  Sample ES document (OBJ-004):")
    print(json.dumps(sample, indent=2, default=str))

    print("\n  Output files:")
    print(f"    {harvest_dir}/terminology.parquet")
    print(f"    {harvest_dir}/objects.parquet")
    print(f"    {transform_dir}/objects_enriched.parquet")
    print(f"    {output_path}")


if __name__ == "__main__":
    main()
