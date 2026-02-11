"""CollectionFlow pattern demo — simulated Dagster asset graph.

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
    uv run python pipeline.py
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandera.polars as pa
import polars as pl

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
    tree = ET.parse(DATA_DIR / "terminology.xml")
    rows = []
    for term in tree.findall(".//term"):
        rows.append(
            {
                "term_id": term.get("id"),
                "term_type": term.get("type"),
                "label": term.text.strip() if term.text else None,
            }
        )
    return pl.DataFrame(rows)


def harvest_objects() -> pl.DataFrame:
    """@dg.asset(kinds={"xml", "polars"}, group_name="harvest")

    Parse all object XML files into a DataFrame with native nested types.
    Nested XML elements become List(Struct(...)) columns — no flattening.
    Harvest stays close to source: term_ids are NOT resolved here.
    """
    records = [
        _parse_object_xml(p)
        for p in sorted((DATA_DIR / "objects").glob("*.xml"))
    ]
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
        classifications_flat
        .join(type_labels, on="type_id", how="left")
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
        constituents_flat
        .join(nationality_lookup, on="nationality_id", how="left")
        .select("object_id", "name", "role", "birth_year", "nationality")
        .group_by("object_id")
        .agg(pl.struct("name", "role", "birth_year", "nationality").alias("constituents"))
    )

    # --- Assemble: flat fields + pass-through nested + enriched nested ---
    enriched = (
        objects.select(
            "object_id", "title", "date_made", "credit_line", "department",
            "dimensions", "media",
        )
        .join(classifications_nested, on="object_id", how="left")
        .join(constituents_nested, on="object_id", how="left")
        .with_columns(
            pl.col("classifications").fill_null([]),
            pl.col("constituents").fill_null([]),
        )
    )

    return enriched.collect(engine="streaming")


# =========================================================================
# ASSET CHECK — Pandera validation (blocking)
# =========================================================================


class ObjectTransformSchema(pa.DataFrameModel):
    """Pandera schema for objects_transform output.

    Demonstrates validation at multiple levels without flattening:
    - Flat column checks (unique, range, nullable)
    - Nested list checks (list.len, list.any, list.all)
    - Nested struct field checks via list.eval(pl.element().struct.field(...))
    - String pattern checks inside nested structs
    - Cross-column checks comparing fields across columns
    """

    object_id: str = pa.Field(unique=True)
    title: str = pa.Field(str_length={"min_value": 1})
    date_made: int = pa.Field(nullable=True, ge=0, le=2100)
    credit_line: str
    department: str
    dimensions: list
    media: list
    classifications: list
    constituents: list

    # --- Basic nested list checks ---

    @pa.check("constituents", name="has_at_least_one_constituent")
    @classmethod
    def has_constituents(cls, col: pl.Series) -> pl.Series:
        return col.list.len() > 0

    @pa.check("media", name="has_at_least_one_image")
    @classmethod
    def has_media(cls, col: pl.Series) -> pl.Series:
        return col.list.len() > 0

    # --- Nested struct field value checks ---

    @pa.check("dimensions", name="all_dimension_values_gt_1")
    @classmethod
    def dimension_values_positive(cls, col: pl.Series) -> pl.Series:
        """Every dimension measurement must be > 1."""
        return col.list.eval(pl.element().struct.field("value")).list.min() > 1

    @pa.check("dimensions", name="has_height_dimension")
    @classmethod
    def has_height(cls, col: pl.Series) -> pl.Series:
        """Every object must have a height dimension."""
        return col.list.eval(
            pl.element().struct.field("type") == "height"
        ).list.any()

    @pa.check("dimensions", name="all_units_are_cm")
    @classmethod
    def units_consistent(cls, col: pl.Series) -> pl.Series:
        """All dimensions must use the same unit (cm)."""
        return col.list.eval(
            pl.element().struct.field("unit") == "cm"
        ).list.all()

    # --- String pattern checks inside nested structs ---

    @pa.check("media", name="all_urls_are_https")
    @classmethod
    def urls_are_https(cls, col: pl.Series) -> pl.Series:
        """All media URLs must use HTTPS."""
        return col.list.eval(
            pl.element().struct.field("url").str.starts_with("https://")
        ).list.all()

    @pa.check("media", name="has_primary_image")
    @classmethod
    def has_primary_image(cls, col: pl.Series) -> pl.Series:
        """Every object must have exactly one primary image."""
        return col.list.eval(
            pl.element().struct.field("type") == "primary"
        ).list.sum() == 1

    @pa.check("constituents", name="no_empty_constituent_names")
    @classmethod
    def constituent_names_not_empty(cls, col: pl.Series) -> pl.Series:
        """Constituent names must not be empty strings."""
        has_items = col.list.len() > 0
        names_ok = col.list.eval(
            pl.element().struct.field("name").str.len_chars() > 0
        ).list.all()
        return ~has_items | names_ok

    @pa.check("constituents", name="has_at_least_one_artist")
    @classmethod
    def has_artist(cls, col: pl.Series) -> pl.Series:
        """If an object has constituents, at least one must have role 'artist'."""
        has_items = col.list.len() > 0
        has_artist_role = col.list.eval(
            pl.element().struct.field("role") == "artist"
        ).list.any()
        return ~has_items | has_artist_role

    @pa.check("classifications", name="no_null_labels_in_classifications")
    @classmethod
    def classification_labels_resolved(cls, col: pl.Series) -> pl.Series:
        """All classification labels must be resolved (no nulls from failed joins)."""
        has_items = col.list.len() > 0
        no_nulls = col.list.eval(
            pl.element().struct.field("term_label").is_not_null()
        ).list.all()
        return ~has_items | no_nulls

    # --- Cross-column checks (use @pa.dataframe_check) ---

    @pa.dataframe_check(name="sculpture_must_have_depth")
    @classmethod
    def sculpture_has_depth(cls, df: pl.DataFrame) -> pl.Series:
        """Objects in the Sculpture department must have a depth dimension."""
        is_sculpture = df["department"] == "Sculpture"
        has_depth = df["dimensions"].list.eval(
            pl.element().struct.field("type") == "depth"
        ).list.any()
        return ~is_sculpture | has_depth

    @pa.dataframe_check(name="constituent_born_before_artwork")
    @classmethod
    def birth_before_creation(cls, df: pl.DataFrame) -> pl.Series:
        """Constituent birth years must be before the artwork's date_made."""
        date_made = df["date_made"]
        max_birth = df["constituents"].list.eval(
            pl.element().struct.field("birth_year")
        ).list.max()
        both_known = date_made.is_not_null() & max_birth.is_not_null()
        valid_where_known = max_birth < date_made
        return ~both_known | valid_where_known

    class Config:
        coerce = True
        strict = False


def check_objects_transform(
    objects_transform: pl.DataFrame,
) -> tuple[bool, list[dict]]:
    """@dg.asset_check(asset=objects_transform, blocking=True)

    Validate transform output with Pandera. Blocks objects_output if failed.
    """
    try:
        ObjectTransformSchema.validate(objects_transform, lazy=True)
        return True, []
    except pa.errors.SchemaErrors as e:
        errors = []
        for row in e.failure_cases.to_dicts():
            errors.append(
                {
                    "column": row.get("column"),
                    "check": row.get("check"),
                    "failure_case": str(row.get("failure_case", ""))[:80],
                }
            )
        return False, errors


# =========================================================================
# OUTPUT ASSET — Nested JSON for Elasticsearch
# =========================================================================


def objects_output(
    objects_transform: pl.DataFrame,
) -> Path:
    """@dg.asset(kinds={"json"}, group_name="output")

    Write enriched objects as nested JSON for Elasticsearch bulk indexing.
    """
    output_path = OUTPUT_DIR / "objects.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = objects_transform.to_dicts()
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    return output_path


# =========================================================================
# XML PARSING HELPERS (would live in a shared module / harvester library)
# =========================================================================


def _parse_object_xml(path: Path) -> dict:
    """Parse a single object XML file into a dict with nested structures."""
    tree = ET.parse(path)
    obj = tree.getroot()

    record: dict = {
        "object_id": obj.get("id"),
        "title": _text(obj, "title"),
        "date_made": _int_or_none(obj, "date_made"),
        "credit_line": _text(obj, "credit_line"),
        "department": _text(obj, "department"),
    }

    record["constituents"] = [
        {
            "name": _text(c, "name"),
            "role": c.get("role"),
            "birth_year": _int_or_none(c, "birth_year"),
            "nationality_id": _text(c, "nationality_id"),
        }
        for c in obj.findall(".//constituent")
    ]

    record["classification_ids"] = [
        {"type_id": c.get("type_id"), "term_id": c.get("term_id")}
        for c in obj.findall(".//classification")
    ]

    record["dimensions"] = [
        {
            "type": d.get("type"),
            "value": float(d.get("value", 0)),
            "unit": d.get("unit"),
        }
        for d in obj.findall(".//dimension")
    ]

    record["media"] = [
        {
            "type": img.get("type"),
            "url": _text(img, "url"),
            "caption": _text(img, "caption"),
        }
        for img in obj.findall(".//image")
    ]

    return record


def _text(el: ET.Element, tag: str) -> str | None:
    child = el.find(tag)
    return child.text.strip() if child is not None and child.text else None


def _int_or_none(el: ET.Element, tag: str) -> int | None:
    val = _text(el, tag)
    return int(val) if val else None


# =========================================================================
# MAIN — Simulate Dagster materialisation sequence
# =========================================================================


def main() -> None:
    print("=" * 70)
    print("COLLECTIONFLOW PATTERN DEMO")
    print("Simulated Dagster asset graph: XML → Polars → Parquet → JSON")
    print("=" * 70)

    # --- Harvest ---
    print("\n--- harvest_terminology ---")
    terminology_df = harvest_terminology()
    print(f"  {len(terminology_df)} terms → Parquet")

    print("\n--- harvest_objects ---")
    objects_df = harvest_objects()
    print(f"  {len(objects_df)} objects from {len(list((DATA_DIR / 'objects').glob('*.xml')))} XML files")
    print(f"  Nested columns: {[n for n, t in objects_df.schema.items() if 'List' in str(t)]}")

    # Simulate IO manager
    harvest_dir = OUTPUT_DIR / "harvest"
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
        row.select("constituents")
        .explode("constituents")
        .unnest("constituents")
    )
    print(f"  OBJ-001 constituents: {constituents['name'].to_list()}")

    transform_dir = OUTPUT_DIR / "transform"
    transform_dir.mkdir(parents=True, exist_ok=True)
    transform_df.write_parquet(transform_dir / "objects_enriched.parquet")

    # Verify Parquet round-trip
    reloaded = pl.read_parquet(transform_dir / "objects_enriched.parquet")
    assert transform_df.schema == reloaded.schema, "Parquet round-trip schema mismatch!"
    print("  Parquet round-trip: schema preserved")

    # --- Validate ---
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
            .list.min() <= 1
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
    print(f"\n  Sample ES document (OBJ-004):")
    print(json.dumps(sample, indent=2, default=str))

    print(f"\n  Output files:")
    print(f"    {harvest_dir}/terminology.parquet")
    print(f"    {harvest_dir}/objects.parquet")
    print(f"    {transform_dir}/objects_enriched.parquet")
    print(f"    {output_path}")


if __name__ == "__main__":
    main()
