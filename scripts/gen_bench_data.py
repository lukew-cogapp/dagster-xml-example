"""Generate 1M row benchmark dataset and write to parquet.

Run once, then use bench_schema.py to benchmark validation.
Builds in batches to avoid memory blowup from Python dicts.
"""

from pathlib import Path

import polars as pl

N = 1_000_000
BATCH = 100_000
ERROR_RATE = 50_000  # 5% of rows get bad data
OUT = Path(__file__).resolve().parent.parent / "output" / "bench.parquet"


def make_batch(offset: int, size: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "object_id": [f"OBJ-{offset + i:07d}" for i in range(size)],
            "title": [f"Artwork {offset + i}" for i in range(size)],
            "date_made": [1800 + ((offset + i) % 200) for i in range(size)],
            "credit_line": [f"Gift of donor {offset + i}" for i in range(size)],
            "department": ["European Paintings"] * size,
            "accession_number": [f"2024.{offset + i}.1" for i in range(size)],
            "is_public_domain": [(offset + i) % 3 != 0 for i in range(size)],
            "gallery_number": [f"Gallery {(offset + i) % 50}" for i in range(size)],
            "dimensions": [
                [
                    {
                        "type": "height",
                        "value": 50.0 + ((offset + i) % 100),
                        "unit": "cm",
                    },
                    {
                        "type": "width",
                        "value": 30.0 + ((offset + i) % 80),
                        "unit": "cm",
                    },
                ]
                for i in range(size)
            ],
            "media": [
                [
                    {
                        "type": "primary",
                        "url": "https://cdn.example.org/img.jpg",
                        "caption": "Front",
                    },
                    {
                        "type": "detail",
                        "url": "https://cdn.example.org/det.jpg",
                        "caption": "Detail",
                    },
                ]
                for _ in range(size)
            ],
            "classifications": [
                [
                    {"type_label": "Painting", "term_label": "Oil on canvas"},
                    {"type_label": "Subject", "term_label": "Landscape"},
                ]
                for _ in range(size)
            ],
            "constituents": [
                [
                    {
                        "name": f"Artist {offset + i}",
                        "role": "artist",
                        "birth_year": 1750 + ((offset + i) % 150),
                        "nationality": "French",
                    },
                    {
                        "name": f"Maker {offset + i}",
                        "role": "frame_maker",
                        "birth_year": 1760 + ((offset + i) % 140),
                        "nationality": "Italian",
                    },
                ]
                for i in range(size)
            ],
            "exhibitions": [
                [
                    {
                        "title": f"Exhibition {(offset + i) % 20}",
                        "year": 1990 + ((offset + i) % 30),
                        "venue": f"Museum {(offset + i) % 10}",
                    }
                ]
                for i in range(size)
            ],
            "provenance": [
                [
                    {
                        "owner": f"Collector {(offset + i) % 100}",
                        "acquired": 1900 + ((offset + i) % 100),
                        "method": "purchase",
                    },
                    {
                        "owner": f"Dealer {(offset + i) % 50}",
                        "acquired": 1850 + ((offset + i) % 120),
                        "method": "inheritance",
                    },
                ]
                for i in range(size)
            ],
            "bibliography": [
                [
                    {
                        "author": f"Author {(offset + i) % 200}",
                        "title": f"Book {(offset + i) % 500}",
                        "year": 1950 + ((offset + i) % 70),
                        "pages": f"{(offset + i) % 300}-{(offset + i) % 300 + 5}",
                    }
                ]
                for i in range(size)
            ],
            "inscriptions": [
                [
                    {
                        "text": f"Signed {offset + i}",
                        "location": "lower right",
                        "type": "signature",
                    }
                ]
                for i in range(size)
            ],
            "conservation": [
                [
                    {
                        "date": f"20{(offset + i) % 24:02d}-01-15",
                        "treatment": "cleaning",
                        "conservator": f"Conservator {(offset + i) % 30}",
                    }
                ]
                for i in range(size)
            ],
            "related_objects": [
                [
                    {
                        "object_id": f"OBJ-{(offset + i + 1) % N:07d}",
                        "relationship": "companion",
                    }
                ]
                for i in range(size)
            ],
            "alt_titles": [
                [
                    {"language": "fr", "title": f"Oeuvre {offset + i}"},
                    {"language": "de", "title": f"Werk {offset + i}"},
                ]
                for i in range(size)
            ],
            "locations": [
                [
                    {
                        "gallery": f"Gallery {(offset + i) % 50}",
                        "floor": (offset + i) % 4,
                        "wing": "East",
                    }
                ]
                for i in range(size)
            ],
        }
    )


print(f"Generating {N:,} rows in batches of {BATCH:,}...")
batches = []
for offset in range(0, N, BATCH):
    print(f"  batch {offset // BATCH + 1}/{N // BATCH}...", end=" ", flush=True)
    batch = make_batch(offset, min(BATCH, N - offset))
    print(f"{batch.estimated_size('mb'):.0f} MB")
    batches.append(batch)

df = pl.concat(batches)
del batches
print(f"Total: {len(df):,} rows, {df.estimated_size('mb'):.0f} MB in memory")

# Inject errors into first ERROR_RATE * 4 rows
print(
    f"\nInjecting errors into {ERROR_RATE * 4:,} rows ({ERROR_RATE * 4 / N * 100:.0f}%)..."
)

dims = df["dimensions"].to_list()
for i in range(ERROR_RATE):
    dims[i] = [{"type": "height", "value": 0.5, "unit": "cm"}]
df = df.with_columns(pl.Series("dimensions", dims))
del dims

consts = df["constituents"].to_list()
for i in range(ERROR_RATE, ERROR_RATE * 2):
    consts[i] = []
df = df.with_columns(pl.Series("constituents", consts))
del consts

media = df["media"].to_list()
for i in range(ERROR_RATE * 2, ERROR_RATE * 3):
    media[i] = []
df = df.with_columns(pl.Series("media", media))
del media

prov = df["provenance"].to_list()
for i in range(ERROR_RATE * 3, ERROR_RATE * 4):
    prov[i] = []
df = df.with_columns(pl.Series("provenance", prov))
del prov

print("  bad dimensions, empty constituents, empty media, empty provenance")

OUT.parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(OUT)
size_mb = OUT.stat().st_size / 1024 / 1024
print(f"\nWrote {OUT} ({size_mb:.1f} MB)")
