"""Pandera schema for the enriched objects DataFrame.

Validates column types (including nested List(Struct) dtypes)
and value-level checks on nested struct fields.

Note: Use lazy=False when validating, as lazy=True crashes on
List(Struct) columns when building failure_cases.
"""

import pandera.polars as pa
import polars as pl

object_transform_schema = pa.DataFrameSchema(
    {
        "object_id": pa.Column(pl.String, unique=True),
        "title": pa.Column(pl.String, pa.Check.str_length(min_value=1)),
        "date_made": pa.Column(pl.Int64, nullable=True),
        "credit_line": pa.Column(pl.String),
        "department": pa.Column(pl.String),
        "dimensions": pa.Column(
            pl.List(
                pl.Struct(
                    {
                        "type": pl.String,
                        "value": pl.Float64,
                        "unit": pl.String,
                    }
                )
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
                pl.Struct(
                    {
                        "type": pl.String,
                        "url": pl.String,
                        "caption": pl.String,
                    }
                )
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
            pl.List(
                pl.Struct(
                    {
                        "type_label": pl.String,
                        "term_label": pl.String,
                    }
                )
            ),
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
                    name="no_null_labels_in_classifications",
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
                    name="no_empty_constituent_names",
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
    },
    coerce=True,
    strict=False,
)
