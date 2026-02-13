"""Benchmark schema validation on 1M rows from parquet.

Prereq: run gen_bench_data.py first to create bench.parquet.

Usage:
    uv run python scripts/gen_bench_data.py   # once
    uv run python scripts/bench_schema.py     # benchmark
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandera.polars as pa
import polars as pl

PARQUET = Path(__file__).resolve().parent.parent / "output" / "bench.parquet"

if not PARQUET.exists():
    print(f"Missing {PARQUET}")
    print("Run: uv run python scripts/gen_bench_data.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Read from parquet (lazy scan, stream collect)
# ---------------------------------------------------------------------------
print("Reading parquet (scan + stream collect)...")
t0 = time.perf_counter()
df = pl.scan_parquet(PARQUET).collect(engine="streaming")
read_time = time.perf_counter() - t0
print(
    f"  {len(df):,} rows, {len(df.columns)} cols, {df.estimated_size('mb'):.0f} MB in {read_time:.3f}s"
)

# ---------------------------------------------------------------------------
# Full schema: dtype + value checks on nested columns
# ---------------------------------------------------------------------------
full_schema = pa.DataFrameSchema(
    {
        "object_id": pa.Column(pl.String, unique=True),
        "title": pa.Column(pl.String, pa.Check.str_length(min_value=1)),
        "date_made": pa.Column(pl.Int64, nullable=True),
        "credit_line": pa.Column(pl.String),
        "department": pa.Column(pl.String),
        "accession_number": pa.Column(pl.String),
        "is_public_domain": pa.Column(pl.Boolean),
        "gallery_number": pa.Column(pl.String),
        "dimensions": pa.Column(
            pl.List(
                pl.Struct({"type": pl.String, "value": pl.Float64, "unit": pl.String})
            ),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("value"))
                        .list.min()
                        .gt(1)
                        .alias(data.key)
                    ),
                    name="all_dimension_values_gt_1",
                ),
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("type") == "height")
                        .list.any()
                        .alias(data.key)
                    ),
                    name="has_height_dimension",
                ),
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("unit") == "cm")
                        .list.all()
                        .alias(data.key)
                    ),
                    name="all_units_are_cm",
                ),
            ],
        ),
        "media": pa.Column(
            pl.List(
                pl.Struct({"type": pl.String, "url": pl.String, "caption": pl.String})
            ),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(
                            pl.element().struct.field("url").str.starts_with("https://")
                        )
                        .list.all()
                        .alias(data.key)
                    ),
                    name="all_urls_are_https",
                ),
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("type") == "primary")
                        .list.sum()
                        .eq(1)
                        .alias(data.key)
                    ),
                    name="has_primary_image",
                ),
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key).list.len().gt(0).alias(data.key)
                    ),
                    name="has_at_least_one_image",
                ),
            ],
        ),
        "classifications": pa.Column(
            pl.List(pl.Struct({"type_label": pl.String, "term_label": pl.String})),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(
                            pl.element().struct.field("term_label").is_not_null()
                        )
                        .list.all()
                        .alias(data.key)
                    ),
                    name="no_null_labels",
                ),
            ],
        ),
        "constituents": pa.Column(
            pl.List(
                pl.Struct(
                    {
                        "name": pl.String,
                        "role": pl.String,
                        "birth_year": pl.Int64,
                        "nationality": pl.String,
                    }
                )
            ),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key).list.len().gt(0).alias(data.key)
                    ),
                    name="has_at_least_one_constituent",
                ),
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(
                            pl.element().struct.field("name").str.len_chars() > 0
                        )
                        .list.all()
                        .alias(data.key)
                    ),
                    name="no_empty_names",
                ),
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("role") == "artist")
                        .list.any()
                        .alias(data.key)
                    ),
                    name="has_at_least_one_artist",
                ),
            ],
        ),
        "exhibitions": pa.Column(
            pl.List(
                pl.Struct({"title": pl.String, "year": pl.Int64, "venue": pl.String})
            ),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("year").gt(1800))
                        .list.all()
                        .alias(data.key)
                    ),
                    name="exhibition_year_after_1800",
                ),
            ],
        ),
        "provenance": pa.Column(
            pl.List(
                pl.Struct(
                    {"owner": pl.String, "acquired": pl.Int64, "method": pl.String}
                )
            ),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key).list.len().gt(0).alias(data.key)
                    ),
                    name="has_provenance",
                ),
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(
                            pl.element().struct.field("owner").str.len_chars() > 0
                        )
                        .list.all()
                        .alias(data.key)
                    ),
                    name="no_empty_owners",
                ),
            ],
        ),
        "bibliography": pa.Column(
            pl.List(
                pl.Struct(
                    {
                        "author": pl.String,
                        "title": pl.String,
                        "year": pl.Int64,
                        "pages": pl.String,
                    }
                )
            ),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("year").gt(1400))
                        .list.all()
                        .alias(data.key)
                    ),
                    name="bib_year_after_1400",
                ),
            ],
        ),
        "inscriptions": pa.Column(
            pl.List(
                pl.Struct({"text": pl.String, "location": pl.String, "type": pl.String})
            ),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(
                            pl.element().struct.field("text").str.len_chars() > 0
                        )
                        .list.all()
                        .alias(data.key)
                    ),
                    name="no_empty_inscription_text",
                ),
            ],
        ),
        "conservation": pa.Column(
            pl.List(
                pl.Struct(
                    {
                        "date": pl.String,
                        "treatment": pl.String,
                        "conservator": pl.String,
                    }
                )
            ),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(
                            pl.element().struct.field("conservator").str.len_chars() > 0
                        )
                        .list.all()
                        .alias(data.key)
                    ),
                    name="no_empty_conservator",
                ),
            ],
        ),
        "related_objects": pa.Column(
            pl.List(pl.Struct({"object_id": pl.String, "relationship": pl.String})),
        ),
        "alt_titles": pa.Column(
            pl.List(pl.Struct({"language": pl.String, "title": pl.String})),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(
                            pl.element().struct.field("language").str.len_chars() == 2
                        )
                        .list.all()
                        .alias(data.key)
                    ),
                    name="language_code_is_2_chars",
                ),
            ],
        ),
        "locations": pa.Column(
            pl.List(
                pl.Struct({"gallery": pl.String, "floor": pl.Int64, "wing": pl.String})
            ),
            checks=[
                pa.Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("floor").ge(0))
                        .list.all()
                        .alias(data.key)
                    ),
                    name="floor_non_negative",
                ),
            ],
        ),
    },
    coerce=True,
    strict=False,
)

n_checks = sum(len(c.checks) for c in full_schema.columns.values())

# Dtype-only schema (derived from full)
dtype_only_schema = pa.DataFrameSchema(
    {name: pa.Column(col.dtype) for name, col in full_schema.columns.items()},
    coerce=True,
    strict=False,
)

# Hybrid schema: dtype all + value checks flat only
hybrid_schema = pa.DataFrameSchema(
    {
        "object_id": pa.Column(pl.String, unique=True),
        "title": pa.Column(pl.String, pa.Check.str_length(min_value=1)),
        "date_made": pa.Column(pl.Int64, nullable=True),
        "credit_line": pa.Column(pl.String),
        "department": pa.Column(pl.String),
        "accession_number": pa.Column(pl.String),
        "is_public_domain": pa.Column(pl.Boolean),
        "gallery_number": pa.Column(pl.String),
        "dimensions": pa.Column(
            pl.List(
                pl.Struct({"type": pl.String, "value": pl.Float64, "unit": pl.String})
            )
        ),
        "media": pa.Column(
            pl.List(
                pl.Struct({"type": pl.String, "url": pl.String, "caption": pl.String})
            )
        ),
        "classifications": pa.Column(
            pl.List(pl.Struct({"type_label": pl.String, "term_label": pl.String}))
        ),
        "constituents": pa.Column(
            pl.List(
                pl.Struct(
                    {
                        "name": pl.String,
                        "role": pl.String,
                        "birth_year": pl.Int64,
                        "nationality": pl.String,
                    }
                )
            )
        ),
        "exhibitions": pa.Column(
            pl.List(
                pl.Struct({"title": pl.String, "year": pl.Int64, "venue": pl.String})
            )
        ),
        "provenance": pa.Column(
            pl.List(
                pl.Struct(
                    {"owner": pl.String, "acquired": pl.Int64, "method": pl.String}
                )
            )
        ),
        "bibliography": pa.Column(
            pl.List(
                pl.Struct(
                    {
                        "author": pl.String,
                        "title": pl.String,
                        "year": pl.Int64,
                        "pages": pl.String,
                    }
                )
            )
        ),
        "inscriptions": pa.Column(
            pl.List(
                pl.Struct({"text": pl.String, "location": pl.String, "type": pl.String})
            )
        ),
        "conservation": pa.Column(
            pl.List(
                pl.Struct(
                    {
                        "date": pl.String,
                        "treatment": pl.String,
                        "conservator": pl.String,
                    }
                )
            )
        ),
        "related_objects": pa.Column(
            pl.List(pl.Struct({"object_id": pl.String, "relationship": pl.String}))
        ),
        "alt_titles": pa.Column(
            pl.List(pl.Struct({"language": pl.String, "title": pl.String}))
        ),
        "locations": pa.Column(
            pl.List(
                pl.Struct({"gallery": pl.String, "floor": pl.Int64, "wing": pl.String})
            )
        ),
    },
    coerce=True,
    strict=False,
)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def run_full_schema(schema, data):
    """Run per-column validation, collecting errors."""
    errors = []
    for col_name, column in schema.columns.items():
        try:
            pa.DataFrameSchema({col_name: column}).validate(data, lazy=False)
        except pa.errors.SchemaError as e:
            errors.append(f"    [{col_name}] {e.check}")
    return errors


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
print(f"\n--- Full schema (dtype + {n_checks} nested checks) ---")
t0 = time.perf_counter()
full_errors = run_full_schema(full_schema, df)
full_time = time.perf_counter() - t0
print(f"  {full_time:.3f}s — {len(full_errors)} failures")
for e in full_errors:
    print(e)

print("\n--- Hybrid schema (dtype all + value checks flat only) ---")
t0 = time.perf_counter()
try:
    hybrid_schema.validate(df, lazy=False)
    hybrid_errors = 0
except pa.errors.SchemaError:
    hybrid_errors = 1
hybrid_time = time.perf_counter() - t0
print(f"  {hybrid_time:.3f}s — {hybrid_errors} failures")

print("\n--- Dtype-only schema (no checks) ---")
t0 = time.perf_counter()
try:
    dtype_only_schema.validate(df, lazy=False)
    dtype_errors = 0
except pa.errors.SchemaError:
    dtype_errors = 1
dtype_time = time.perf_counter() - t0
print(f"  {dtype_time:.3f}s — {dtype_errors} failures")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(
    f"\n--- Summary ({len(df):,} rows, {len(df.columns)} cols, {n_checks} nested checks) ---"
)
print(f"  Parquet read:       {read_time:.3f}s")
print(f"  Dtype only:         {dtype_time:.3f}s")
print(f"  Hybrid (flat vals): {hybrid_time:.3f}s")
print(f"  Full (all checks):  {full_time:.3f}s")
print("")
print(f"  Hybrid vs dtype:    +{hybrid_time - dtype_time:.3f}s")
print(f"  Full vs hybrid:     +{full_time - hybrid_time:.3f}s")
