"""Dagster JSON example — same pipeline, different source format.

Demonstrates source-format independence: only the harvest layer changes.
Transform, validate, and output are identical to the XML pipeline.

Asset graph:
    harvest_terminology ─┐
                         ├─→ objects_transform ──→ objects_output
    harvest_objects ─────┘

    check_objects_transform (asset check, blocking)

The JSON harvest is trivially simple compared to XML parsing —
pl.read_json() replaces all the ElementTree code.

Usage:
    uv run python pipeline_json.py
"""

import json
from pathlib import Path

import polars as pl

from pipeline_xml import check_objects_transform, objects_transform

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


# =========================================================================
# HARVEST ASSETS — JSON → Parquet (close to source format)
# =========================================================================


def harvest_terminology() -> pl.DataFrame:
    """@dg.asset(kinds={"json", "polars"}, group_name="harvest")

    Read terminology.json into a flat lookup DataFrame.
    """
    return pl.read_json(DATA_DIR / "json" / "terminology.json")


def harvest_objects() -> pl.DataFrame:
    """@dg.asset(kinds={"json", "polars"}, group_name="harvest")

    Read objects.json into a DataFrame with native nested types.
    pl.read_json() produces List(Struct(...)) columns automatically
    from nested JSON arrays — no manual parsing needed.
    """
    return pl.read_json(DATA_DIR / "json" / "objects.json")


# =========================================================================
# OUTPUT ASSET — Nested JSON for Elasticsearch
# =========================================================================


def objects_output(
    objects_transform_df: pl.DataFrame,
) -> Path:
    """@dg.asset(kinds={"json"}, group_name="output")

    Write enriched objects as nested JSON for Elasticsearch bulk indexing.
    Output goes to output/json/ to keep it separate from the XML pipeline.
    """
    output_path = OUTPUT_DIR / "json" / "objects.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = objects_transform_df.to_dicts()
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    return output_path


# =========================================================================
# MAIN — Simulate Dagster materialisation sequence
# =========================================================================


def main() -> None:
    print("=" * 70)
    print("DAGSTER JSON EXAMPLE")
    print("Simulated Dagster asset graph: JSON → Polars → Parquet → JSON")
    print("=" * 70)

    # --- Harvest ---
    print("\n--- harvest_terminology ---")
    terminology_df = harvest_terminology()
    print(f"  {len(terminology_df)} terms → Parquet")

    print("\n--- harvest_objects ---")
    objects_df = harvest_objects()
    print(f"  {len(objects_df)} objects from data/json/objects.json")
    print(
        f"  Nested columns: {[n for n, t in objects_df.schema.items() if 'List' in str(t)]}"
    )

    # Simulate IO manager
    harvest_dir = OUTPUT_DIR / "json" / "harvest"
    harvest_dir.mkdir(parents=True, exist_ok=True)
    terminology_df.write_parquet(harvest_dir / "terminology.parquet")
    objects_df.write_parquet(harvest_dir / "objects.parquet")

    # --- Transform (reuses pipeline.objects_transform) ---
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

    transform_dir = OUTPUT_DIR / "json" / "transform"
    transform_dir.mkdir(parents=True, exist_ok=True)
    transform_df.write_parquet(transform_dir / "objects_enriched.parquet")

    # Verify Parquet round-trip
    reloaded = pl.read_parquet(transform_dir / "objects_enriched.parquet")
    assert transform_df.schema == reloaded.schema, "Parquet round-trip schema mismatch!"
    print("  Parquet round-trip: schema preserved")

    # --- Validate (reuses pipeline.check_objects_transform) ---
    print("\n--- check_objects_transform (14 Pandera checks) ---")
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
    print(f"  {len(transform_df)} records → {output_path}")

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
