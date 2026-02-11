# Dagster XML/JSON Example

Demonstrates a Dagster ETL pattern for museum/collection data:
**nested source data** (XML or JSON) through **Polars DataFrames with native nested types**
to **nested JSON for Elasticsearch**.

Two pipelines share identical transform, validate, and output code — only the
**harvest layer** changes. This demonstrates that the pipeline is source-format agnostic.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HARVEST (XML or JSON → Parquet)                       │
│                                                                         │
│  XML:  data/xml/*.xml        ──→ harvest_* ──→ Parquet (nested)         │
│  JSON: data/json/*.json      ──→ harvest_* ──→ Parquet (nested)         │
│                                                                         │
│  Only the harvest layer differs — everything below is shared.           │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORM (Polars lazy + streaming)                   │
│                                                                         │
│  objects_transform:                                                      │
│    classifications: explode → join terminology → re-nest as List(Struct) │
│    constituents:    explode → join terminology → re-nest as List(Struct) │
│    dimensions:      pass-through (never flattened)                       │
│    media:           pass-through (never flattened)                       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   VALIDATE (Pandera — 14 checks, blocking)              │
│                                                                         │
│  check_objects_transform:                                                │
│    nested struct field checks via list.eval(element().struct.field(...)) │
│    cross-column checks via @pa.dataframe_check                          │
│    NO flattening required                                                │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       OUTPUT (Parquet → nested JSON)                     │
│                                                                         │
│  objects_output ──→ objects.json (ready for Elasticsearch bulk index)    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync

# Run the XML pipeline
uv run python pipeline.py

# Run the JSON pipeline (same transform/validate/output, different harvest)
uv run python pipeline_json.py
```

To inspect the generated Parquet files (run after either pipeline):

```bash
uv run python show_parquet.py
```

## What It Does

Simulates a Dagster asset graph without requiring Dagster as a dependency.
Each function represents a Dagster asset, with parameters representing upstream
dependencies that the IO manager would inject.

```
pipeline.py (XML)                    pipeline_json.py (JSON)
─────────────────                    ───────────────────────
harvest_terminology (XML parsing)    harvest_terminology (pl.read_json)
harvest_objects     (XML parsing)    harvest_objects     (pl.read_json)
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
              objects_transform        ← shared from pipeline.py
              check_objects_transform  ← shared from pipeline.py
              objects_output           ← local (different output dir)
```

## Asset Graph

### harvest_terminology

**XML pipeline**: Parses `data/xml/terminology.xml` into a flat lookup DataFrame (term_id, term_type, label).

**JSON pipeline**: Reads `data/json/terminology.json` with `pl.read_json()` — one line of code.

### harvest_objects

**XML pipeline**: Parses 5 XML files from `data/xml/objects/` into a single DataFrame.

**JSON pipeline**: Reads `data/json/objects.json` with `pl.read_json()` — nested JSON arrays
become `List(Struct(...))` columns automatically.

In both cases, nested elements
(constituents, classifications, dimensions, media) are stored as Polars
`List(Struct(...))` columns — not flattened, not serialised to JSON strings.

### objects_transform

Enriches objects with terminology labels using Polars lazy API with streaming collect.

- **Classifications**: explode → join term_id to label → group_by → re-nest as `List(Struct)`
- **Constituents**: explode → join nationality_id to label → group_by → re-nest
- **Dimensions, media**: pass-through (never flattened — no enrichment needed)

Uses `.lazy()` throughout with a single `.collect(engine="streaming")` at the end.

### check_objects_transform

Pandera validation on the transform output — 14 checks across 5 categories, all
operating on nested struct fields **without flattening**.

**Basic list checks:**
- Every object has at least one constituent
- Every object has at least one image

**Nested struct field value checks:**
- All dimension values > 1
- Every object has a height dimension
- All dimension units are cm

**String pattern checks inside nested structs:**
- All media URLs use HTTPS
- Every object has exactly one primary image
- No empty constituent names
- If constituents exist, at least one must have role "artist"

**Referential integrity:**
- All classification labels resolved (no nulls from failed joins)

**Cross-column checks:**
- Sculpture department objects must have a depth dimension
- Constituent birth years must be before the artwork's date_made

All nested checks use `list.eval(pl.element().struct.field(...))` — no flattening needed.

### objects_output

Writes the enriched DataFrame as nested JSON, ready for Elasticsearch bulk indexing.

## Key Patterns Demonstrated

### Native nested types in Parquet

Polars `List(Struct(...))` columns survive Parquet round-trips with full type fidelity.
No JSON serialisation needed — the nested schema is stored in Parquet's native format.

### Selective flattening

Only columns that need enrichment (joins with terminology) are exploded.
Pass-through nested data stays nested throughout the entire pipeline.

### Pandera validation of nested data

Custom Pandera checks can reach into struct fields without exploding:

```python
@pa.check("dimensions", name="all_values_gt_1")
@classmethod
def dimension_values_positive(cls, col: pl.Series) -> pl.Series:
    return col.list.eval(pl.element().struct.field("value")).list.min() > 1
```

Cross-column checks use `@pa.dataframe_check` with the vacuously-true pattern
(`~condition | check_result`) for checks on optional nested lists:

```python
@pa.dataframe_check(name="sculpture_must_have_depth")
@classmethod
def sculpture_has_depth(cls, df: pl.DataFrame) -> pl.Series:
    is_sculpture = df["department"] == "Sculpture"
    has_depth = df["dimensions"].list.eval(
        pl.element().struct.field("type") == "depth"
    ).list.any()
    return ~is_sculpture | has_depth
```

### Source-format independence

The XML and JSON pipelines produce identical data content — schemas match at every
stage (harvest, transform, output) and all values are equal. The only differences
are row ordering and list element ordering within `group_by` results, which are
non-deterministic in Polars. The transform, validate, and output code doesn't
know or care whether the source was XML or JSON.

### Terminology enrichment as joins

Term ID resolution happens via Polars `.join()`, not Python dict lookups.
This handles nulls, duplicates, and many-to-many relationships correctly.

## Data Quality Issues (Planted)

Two objects have intentional data quality issues to demonstrate validation:

- **OBJ-003**: missing `date_made`, empty `constituents` list
- **OBJ-005**: dimension height value of 0.5 cm (fails the > 1 check)

## Project Structure

```
collectionflow-demo/
├── pyproject.toml          # uv project config (polars + pandera)
├── pipeline.py             # XML pipeline (harvest + shared transform/validate/output)
├── pipeline_json.py        # JSON pipeline (harvest only, imports shared code)
├── show_parquet.py         # Inspect generated Parquet files from both pipelines
├── data/
│   ├── xml/
│   │   ├── terminology.xml     # 16 terms (object types, media, subjects, nationalities)
│   │   └── objects/
│   │       ├── OBJ-001.xml     # Harbor painting, 2 constituents, 4 classifications
│   │       ├── OBJ-002.xml     # Still life, 1 constituent
│   │       ├── OBJ-003.xml     # Portrait, missing data (no date, no constituents)
│   │       ├── OBJ-004.xml     # Bronze sculpture, 3 dimensions
│   │       └── OBJ-005.xml     # Print, bad dimension value (0.5 cm)
│   └── json/
│       ├── terminology.json    # Same 16 terms as flat JSON array
│       └── objects.json        # Same 5 objects as JSON array (with nested arrays)
└── output/                     # Generated at runtime (gitignored)
    ├── xml/                    # XML pipeline outputs
    │   ├── harvest/            # Parquet close to source
    │   ├── transform/          # Enriched Parquet
    │   └── objects.json        # Final nested JSON
    └── json/                   # JSON pipeline outputs
        ├── harvest/            # Parquet close to source
        ├── transform/          # Enriched Parquet
        └── objects.json        # Final nested JSON
```
