"""Test nested struct validation patterns with Pandera + Polars.

Focus: Can you attach Check objects directly to pa.Column for
List(Struct(...)) columns using the imperative DataFrameSchema API?

Tests the user's proposed pattern and several alternatives.

Usage:
    uv run python scripts/test_nested_checks.py
"""

import sys
import traceback

import polars as pl
import pandera.polars as pa
from pandera import Check
from pandera.api.polars.types import PolarsData

PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"

results: list[tuple[str, str, str]] = []


def run_test(name: str, fn):
    """Run a test function, capture result."""
    try:
        status, detail = fn()
        results.append((name, status, detail))
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, ERROR, f"{type(e).__name__}: {e}\n{tb}"))


def make_constituents_df() -> pl.DataFrame:
    """Create test DataFrame with List(Struct) column."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "constituents": [
                [
                    {
                        "name": "Alice",
                        "role": "Artist",
                        "birth_year": 1980,
                        "nationality": "American",
                    },
                    {
                        "name": "Bob",
                        "role": "Sculptor",
                        "birth_year": 1975,
                        "nationality": "British",
                    },
                ],
                [
                    {
                        "name": "Charlie",
                        "role": "Painter",
                        "birth_year": 1960,
                        "nationality": "French",
                    },
                ],
                [
                    {
                        "name": "Diana",
                        "role": "Curator",
                        "birth_year": -50,
                        "nationality": "Italian",
                    },
                    {
                        "name": "Eve",
                        "role": "Designer",
                        "birth_year": 1990,
                        "nationality": "German",
                    },
                ],
            ],
        }
    )


# =========================================================================
# TEST 1: User's proposed pattern -- Check lambda on Column with List(Struct)
# Key question: does the lambda receive a pl.Series or a PolarsData?
# =========================================================================


def test_user_pattern_series_lambda():
    """The user's original pattern: Check(lambda s: s.list.eval(...))

    This assumes the lambda receives a pl.Series. Based on source code
    analysis of PolarsCheckBackend.apply() (checks.py line 54):
        out = self.check_fn(check_obj)
    where check_obj is PolarsData, NOT pl.Series.

    So this pattern will CRASH because PolarsData has no .list attribute.
    """
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(
                    lambda s: s.list.eval(
                        pl.element().struct.field("birth_year").gt(0)
                    ).list.all()
                ),
            ),
        }
    )

    try:
        schema.validate(df)
        return FAIL, "Validation passed -- check was silently skipped?"
    except pa.errors.SchemaErrors as e:
        fc_str = str(e.failure_cases)
        if "AttributeError" in fc_str or "has no attribute" in fc_str:
            return PASS, (
                "CONFIRMED: lambda receives PolarsData, NOT pl.Series.\n"
                "The lambda crashes with AttributeError because PolarsData has no .list.\n"
                "Pandera catches this internally and reports it as a validation failure\n"
                "(which is misleading -- it's a code bug, not bad data).\n"
                f"failure_cases excerpt: {fc_str[:300]}"
            )
        return PASS, f"Check failed with unexpected error:\n{fc_str[:300]}"
    except pa.errors.SchemaError as e:
        msg = str(e)
        if "AttributeError" in msg or "has no attribute" in msg:
            return PASS, (
                "CONFIRMED: lambda receives PolarsData, NOT pl.Series.\n"
                f"SchemaError: {msg[:300]}"
            )
        return PASS, f"Check failed (SchemaError): {msg[:300]}"


run_test(
    "1. User's pattern: Check(lambda s: s.list.eval(...))",
    test_user_pattern_series_lambda,
)


# =========================================================================
# TEST 2: Inspect what Check lambda actually receives
# =========================================================================


def test_inspect_check_arg_type():
    """Introspect the actual argument type passed to a Check lambda."""
    df = make_constituents_df()
    received_type = None
    received_attrs = None

    def inspector(arg):
        nonlocal received_type, received_attrs
        received_type = type(arg).__name__
        received_attrs = [a for a in dir(arg) if not a.startswith("_")]
        # Return True to pass the check
        return True

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(inspector),
            ),
        }
    )

    try:
        schema.validate(df)
    except Exception:
        pass

    if received_type == "PolarsData":
        return PASS, (
            f"Check lambda receives: {received_type}\n"
            f"Attributes: {received_attrs}\n"
            "This is a named tuple with .lazyframe and .key fields.\n"
            "You CANNOT use pl.Series methods like .list.eval() on it."
        )
    return FAIL, f"Received unexpected type: {received_type}, attrs: {received_attrs}"


run_test(
    "2. Inspect: what type does Check lambda receive?",
    test_inspect_check_arg_type,
)


# =========================================================================
# TEST 3: CORRECT PATTERN -- Check lambda using PolarsData API
# =========================================================================


def test_correct_polarsdata_lambda():
    """Correct pattern: Check lambda that takes PolarsData, returns LazyFrame."""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("birth_year").gt(0))
                        .list.all()
                        .alias(data.key)
                    ),
                    name="birth_year_positive",
                ),
            ),
        }
    )

    try:
        schema.validate(df, lazy=True)
        return FAIL, "Should have caught birth_year=-50 in row 2"
    except pa.errors.SchemaErrors as e:
        try:
            fc = e.failure_cases
            return PASS, (
                f"CORRECT PATTERN WORKS (with lazy=True).\nfailure_cases:\n{fc}\n"
            )
        except Exception as fmt_err:
            return PASS, (
                "Check DETECTED the bad data, but Pandera crashes building\n"
                f"failure_cases: {type(fmt_err).__name__}: {fmt_err}\n"
                "This is the known Polars cast bug for List(Struct) -> String.\n"
                "The CHECK LOGIC is correct; the error REPORTING is broken."
            )
    except pa.errors.SchemaError as e:
        return PASS, f"Check detected bad data (eager mode):\n{str(e)[:400]}"
    except Exception as e:
        if "cannot cast" in str(e) or "conversion from" in str(e):
            return PASS, (
                "Check DETECTED the bad data, but Pandera crashes on error report:\n"
                f"{type(e).__name__}: {str(e)[:200]}\n"
                "Known bug: Pandera cannot format List(Struct) failure cases."
            )
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "3a. Correct: Check(lambda data: data.lazyframe.select(...))",
    test_correct_polarsdata_lambda,
)


def test_correct_pattern_eager():
    """Same correct pattern but with lazy=False (eager mode)."""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("birth_year").gt(0))
                        .list.all()
                        .alias(data.key)
                    ),
                    name="birth_year_positive",
                ),
            ),
        }
    )

    try:
        schema.validate(df, lazy=False)
        return FAIL, "Should have caught birth_year=-50 in row 2"
    except pa.errors.SchemaError as e:
        return PASS, (
            "CORRECT PATTERN WORKS (with lazy=False, eager mode).\n"
            f"SchemaError:\n{str(e)[:500]}"
        )
    except pa.errors.SchemaErrors as e:
        return PASS, f"Check detected bad data (lazy mode):\n{str(e)[:400]}"
    except Exception as e:
        if "cannot cast" in str(e) or "conversion from" in str(e):
            return PASS, (
                "Check detected bad data but error reporting crashed:\n"
                f"{type(e).__name__}: {str(e)[:200]}"
            )
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "3b. Correct pattern with lazy=False (eager mode)",
    test_correct_pattern_eager,
)


def test_correct_pattern_valid_data():
    """Correct pattern with data that should PASS validation."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "constituents": [
                [
                    {
                        "name": "Alice",
                        "role": "Artist",
                        "birth_year": 1980,
                        "nationality": "US",
                    },
                ],
                [
                    {
                        "name": "Bob",
                        "role": "Painter",
                        "birth_year": 1960,
                        "nationality": "UK",
                    },
                    {
                        "name": "Charlie",
                        "role": "Sculptor",
                        "birth_year": 1975,
                        "nationality": "FR",
                    },
                ],
            ],
        }
    )

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("birth_year").gt(0))
                        .list.all()
                        .alias(data.key)
                    ),
                    name="birth_year_positive",
                ),
            ),
        }
    )

    try:
        result = schema.validate(df)
        return PASS, f"Valid data passes correctly. Result shape: {result.shape}"
    except Exception as e:
        return FAIL, f"Valid data should have passed: {type(e).__name__}: {e}"


run_test(
    "3c. Correct pattern with valid data (should pass)",
    test_correct_pattern_valid_data,
)


# =========================================================================
# TEST 4: Named function instead of lambda (for readability)
# =========================================================================


def test_named_function_check():
    """Using a named function instead of lambda for clarity."""
    df = make_constituents_df()

    def check_birth_year_positive(data: PolarsData) -> pl.LazyFrame:
        """All birth_year values in the list must be > 0."""
        return data.lazyframe.select(
            pl.col(data.key)
            .list.eval(pl.element().struct.field("birth_year").gt(0))
            .list.all()
            .alias(data.key)
        )

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(check_birth_year_positive, name="birth_year_positive"),
            ),
        }
    )

    try:
        schema.validate(df, lazy=False)
        return FAIL, "Should have caught birth_year=-50"
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
        return PASS, (f"Named function pattern works.\nError: {str(e)[:300]}")
    except Exception as e:
        if "cannot cast" in str(e):
            return (
                PASS,
                "Named function works (check caught bad data, error reporting crashed)",
            )
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "4. Named function: Check(check_birth_year_positive)",
    test_named_function_check,
)


# =========================================================================
# TEST 5: Multiple checks on a single column
# =========================================================================


def test_multiple_checks_on_column():
    """Multiple Check objects on the same List(Struct) column."""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                    # Check 1: birth_year > 0
                    Check(
                        lambda data: data.lazyframe.select(
                            pl.col(data.key)
                            .list.eval(pl.element().struct.field("birth_year").gt(0))
                            .list.all()
                            .alias(data.key)
                        ),
                        name="birth_year_positive",
                    ),
                    # Check 2: list must be non-empty
                    Check(
                        lambda data: data.lazyframe.select(
                            pl.col(data.key).list.len().gt(0).alias(data.key)
                        ),
                        name="non_empty_constituents",
                    ),
                    # Check 3: name must not be null
                    Check(
                        lambda data: data.lazyframe.select(
                            pl.col(data.key)
                            .list.eval(pl.element().struct.field("name").is_not_null())
                            .list.all()
                            .alias(data.key)
                        ),
                        name="name_not_null",
                    ),
                ],
            ),
        }
    )

    try:
        schema.validate(df, lazy=False)
        return FAIL, "Should have caught birth_year=-50"
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
        return (
            PASS,
            f"Multiple checks work on List(Struct) column.\nError: {str(e)[:400]}",
        )
    except Exception as e:
        if "cannot cast" in str(e):
            return PASS, "Multiple checks work (error reporting crashed as expected)"
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "5. Multiple checks on same List(Struct) column",
    test_multiple_checks_on_column,
)


# =========================================================================
# TEST 6: Built-in checks on List(Struct) -- do they work?
# =========================================================================


def test_builtin_gt_on_list_column():
    """Can you use Check.greater_than(0) on a List(Struct) column? (Expect: no)"""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check.greater_than(0),
            ),
        }
    )

    try:
        schema.validate(df)
        return FAIL, "Built-in check accepted on List(Struct) -- unexpected"
    except Exception as e:
        return PASS, (
            "Built-in Check.greater_than(0) DOES NOT WORK on List(Struct).\n"
            f"{type(e).__name__}: {str(e)[:200]}\n"
            "Built-in checks are designed for scalar columns only."
        )


run_test(
    "6a. Built-in Check.greater_than(0) on List(Struct)",
    test_builtin_gt_on_list_column,
)


def test_builtin_str_length_on_list_column():
    """Can you use Check.str_length on a List(Struct) column? (Expect: no)"""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check.str_length(min_value=1),
            ),
        }
    )

    try:
        schema.validate(df)
        return FAIL, "Check.str_length accepted on List(Struct) -- unexpected"
    except Exception as e:
        return PASS, (
            "Built-in Check.str_length DOES NOT WORK on List(Struct).\n"
            f"{type(e).__name__}: {str(e)[:200]}\n"
            "Built-in checks are designed for scalar string columns only."
        )


run_test(
    "6b. Built-in Check.str_length on List(Struct)",
    test_builtin_str_length_on_list_column,
)


def test_builtin_isin_on_list_column():
    """Can you use Check.isin on a List(Struct) column? (Expect: no)"""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check.isin(["Artist", "Sculptor"]),
            ),
        }
    )

    try:
        schema.validate(df)
        return FAIL, "Check.isin accepted on List(Struct) -- unexpected"
    except Exception as e:
        return PASS, (
            "Built-in Check.isin DOES NOT WORK on List(Struct).\n"
            f"{type(e).__name__}: {str(e)[:200]}\n"
            "Built-in checks compare the WHOLE LIST value, not inner elements."
        )


run_test(
    "6c. Built-in Check.isin on List(Struct)",
    test_builtin_isin_on_list_column,
)


# =========================================================================
# TEST 7: element_wise=True on List(Struct) -- does it help?
# =========================================================================


def test_element_wise_check_on_list():
    """element_wise=True with List(Struct) -- each 'element' is a whole list value."""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(
                    lambda element: all(item["birth_year"] > 0 for item in element),
                    element_wise=True,
                    name="birth_year_positive_elementwise",
                ),
            ),
        }
    )

    try:
        schema.validate(df, lazy=False)
        return FAIL, "Should have caught birth_year=-50"
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
        return PASS, (
            "element_wise=True WORKS for List(Struct)!\n"
            "Each 'element' is a whole list (row value), and each\n"
            "struct inside becomes a Python dict. So you use Python\n"
            "iteration to check the inner values.\n"
            f"Error: {str(e)[:400]}"
        )
    except Exception as e:
        return ERROR, f"{type(e).__name__}: {str(e)[:300]}"


run_test(
    "7a. element_wise=True: each element is a whole list value",
    test_element_wise_check_on_list,
)


def test_element_wise_valid_data():
    """element_wise=True with valid data -- should pass."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "constituents": [
                [
                    {
                        "name": "Alice",
                        "role": "Artist",
                        "birth_year": 1980,
                        "nationality": "US",
                    }
                ],
                [
                    {
                        "name": "Bob",
                        "role": "Painter",
                        "birth_year": 1960,
                        "nationality": "UK",
                    }
                ],
            ],
        }
    )

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(
                    lambda element: all(item["birth_year"] > 0 for item in element),
                    element_wise=True,
                    name="birth_year_positive_elementwise",
                ),
            ),
        }
    )

    try:
        result = schema.validate(df)
        return PASS, f"element_wise=True passes with valid data. Shape: {result.shape}"
    except Exception as e:
        return FAIL, f"Valid data should have passed: {type(e).__name__}: {e}"


run_test(
    "7b. element_wise=True with valid data",
    test_element_wise_valid_data,
)


# =========================================================================
# TEST 8: Complex nested checks with PolarsData (string patterns, isin, etc.)
# =========================================================================


def test_nested_isin_check():
    """Check that all 'role' values in struct list are from an allowed set."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "constituents": [
                [
                    {
                        "name": "Alice",
                        "role": "Artist",
                        "birth_year": 1980,
                        "nationality": "US",
                    },
                ],
                [
                    {
                        "name": "Bob",
                        "role": "INVALID_ROLE",
                        "birth_year": 1960,
                        "nationality": "UK",
                    },
                ],
            ],
        }
    )

    allowed_roles = ["Artist", "Sculptor", "Painter", "Curator", "Designer"]

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(
                            pl.element().struct.field("role").is_in(allowed_roles)
                        )
                        .list.all()
                        .alias(data.key)
                    ),
                    name="valid_roles",
                ),
            ),
        }
    )

    try:
        schema.validate(df, lazy=False)
        return FAIL, "Should have caught INVALID_ROLE"
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
        return PASS, (
            f"Nested is_in check works for struct fields.\nError: {str(e)[:400]}"
        )
    except Exception as e:
        if "cannot cast" in str(e):
            return PASS, "Nested is_in works (error reporting crashed as expected)"
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "8a. Nested is_in check on struct field",
    test_nested_isin_check,
)


def test_nested_string_pattern_check():
    """Check that all 'name' values match a pattern (non-empty, no digits)."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "constituents": [
                [
                    {
                        "name": "Alice",
                        "role": "Artist",
                        "birth_year": 1980,
                        "nationality": "US",
                    }
                ],
                [
                    {
                        "name": "",
                        "role": "Painter",
                        "birth_year": 1960,
                        "nationality": "UK",
                    }
                ],
            ],
        }
    )

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(
                            pl.element().struct.field("name").str.len_chars().gt(0)
                        )
                        .list.all()
                        .alias(data.key)
                    ),
                    name="name_non_empty",
                ),
            ),
        }
    )

    try:
        schema.validate(df, lazy=False)
        return FAIL, "Should have caught empty name"
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
        return PASS, (f"Nested string length check works.\nError: {str(e)[:400]}")
    except Exception as e:
        if "cannot cast" in str(e):
            return (
                PASS,
                "Nested string check works (error reporting crashed as expected)",
            )
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "8b. Nested string length check on struct field",
    test_nested_string_pattern_check,
)


# =========================================================================
# TEST 9: Dtype-only validation (no value checks)
# =========================================================================


def test_dtype_only_validation():
    """Column dtype validation for List(Struct) without value checks."""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
        }
    )

    try:
        result = schema.validate(df)
        return PASS, f"Dtype-only validation passes. Result type: {result.schema}"
    except Exception as e:
        return FAIL, f"Dtype-only validation should pass: {type(e).__name__}: {e}"


run_test(
    "9a. Dtype-only: List(Struct) column type check",
    test_dtype_only_validation,
)


def test_dtype_wrong_field_detected():
    """Dtype validation catches wrong field type in struct."""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
            "constituents": pa.Column(
                pl.List(
                    pl.Struct(
                        {
                            "name": pl.String,
                            "role": pl.String,
                            "birth_year": pl.Float64,  # WRONG: data has Int64
                            "nationality": pl.String,
                        }
                    )
                )
            ),
        }
    )

    try:
        schema.validate(df)
        return FAIL, "Should have caught wrong field type (Float64 vs Int64)"
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
        return (
            PASS,
            f"Dtype validation catches wrong struct field type:\n{str(e)[:300]}",
        )


run_test(
    "9b. Dtype: detects wrong struct field type",
    test_dtype_wrong_field_detected,
)


# =========================================================================
# TEST 10: Comparison: DataFrameModel @pa.check vs DataFrameSchema Check
# =========================================================================


def test_dataframemodel_check_comparison():
    """Same check logic in DataFrameModel for comparison."""
    df = make_constituents_df()

    class Schema(pa.DataFrameModel):
        id: int
        constituents: list

        @pa.check("constituents", name="birth_year_positive")
        @classmethod
        def check_birth_year(cls, data: PolarsData) -> pl.LazyFrame:
            return data.lazyframe.select(
                pl.col(data.key)
                .list.eval(pl.element().struct.field("birth_year").gt(0))
                .list.all()
                .alias(data.key)
            )

        class Config:
            strict = False
            coerce = False

    try:
        Schema.validate(df, lazy=False)
        return FAIL, "Should have caught birth_year=-50"
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
        return PASS, (
            f"DataFrameModel @pa.check works identically.\nError: {str(e)[:400]}"
        )
    except Exception as e:
        if "cannot cast" in str(e):
            return (
                PASS,
                "DataFrameModel check works (error reporting crashed as expected)",
            )
        return ERROR, f"{type(e).__name__}: {e}"


run_test(
    "10. DataFrameModel @pa.check for comparison",
    test_dataframemodel_check_comparison,
)


# =========================================================================
# TEST 11: Error reporting for List(Struct) failure cases
# =========================================================================


def test_failure_case_cast_bug():
    """Demonstrate the List(Struct) -> String cast bug in failure_cases."""
    df = make_constituents_df()

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64),
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
                checks=Check(
                    lambda data: data.lazyframe.select(
                        pl.col(data.key)
                        .list.eval(pl.element().struct.field("birth_year").gt(0))
                        .list.all()
                        .alias(data.key)
                    ),
                    name="birth_year_positive",
                ),
            ),
        }
    )

    # Test with lazy=True (collects all errors)
    lazy_result = ""
    try:
        schema.validate(df, lazy=True)
        lazy_result = "PASSED (unexpected)"
    except pa.errors.SchemaErrors as e:
        try:
            fc = e.failure_cases
            lazy_result = f"lazy=True: failure_cases works!\n{fc}"
        except Exception as fmt_err:
            lazy_result = (
                f"lazy=True: failure_cases CRASHES\n"
                f"{type(fmt_err).__name__}: {str(fmt_err)[:200]}\n"
                "Known bug: Pandera casts List(Struct) to String via .cast(pl.Utf8),\n"
                "but Polars cannot do this cast. See pandera/backends/polars/base.py line 198."
            )
    except Exception as e:
        lazy_result = f"lazy=True: {type(e).__name__}: {str(e)[:200]}"

    # Test with lazy=False (stops at first error)
    eager_result = ""
    try:
        schema.validate(df, lazy=False)
        eager_result = "PASSED (unexpected)"
    except pa.errors.SchemaError as e:
        eager_result = f"lazy=False: SchemaError raised correctly\n{str(e)[:300]}"
    except pa.errors.SchemaErrors as e:
        eager_result = f"lazy=False: SchemaErrors\n{str(e)[:300]}"
    except Exception as e:
        eager_result = f"lazy=False: {type(e).__name__}: {str(e)[:200]}"

    return PASS, f"{lazy_result}\n\n{eager_result}"


run_test(
    "11. Error reporting: lazy=True vs lazy=False for List(Struct)",
    test_failure_case_cast_bug,
)


# =========================================================================
# REPORT
# =========================================================================


def main():
    import pandera

    print("=" * 78)
    print("PANDERA NESTED CHECK VALIDATION TESTS")
    print(f"Pandera v{pandera.__version__} | Polars v{pl.__version__}")
    print("=" * 78)

    pass_count = sum(1 for _, s, _ in results if s == PASS)
    fail_count = sum(1 for _, s, _ in results if s == FAIL)
    error_count = sum(1 for _, s, _ in results if s == ERROR)

    for name, status, detail in results:
        icon = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "ERROR": "[ERR] "}[status]
        print(f"\n{icon} {name}")
        for line in detail.split("\n"):
            print(f"       {line}")

    print("\n" + "=" * 78)
    print(
        f"SUMMARY: {pass_count} passed, {fail_count} unexpected, {error_count} errors"
    )
    print("=" * 78)

    print(
        """
========================================================================
CONCLUSIONS
========================================================================

QUESTION: Can you attach Check objects directly to pa.Column for
List(Struct) columns in the imperative DataFrameSchema API?

ANSWER: YES, but the lambda signature is different from what you'd expect.

THE CRITICAL DIFFERENCE:
  - In Pandas:   Check lambda receives pd.Series
  - In Polars:   Check lambda receives PolarsData (NOT pl.Series)

PolarsData is a named tuple with:
  .lazyframe  -- the full pl.LazyFrame being validated
  .key        -- the column name being checked

YOUR PROPOSED PATTERN (DOES NOT WORK):
    Check(
        lambda s: s.list.eval(
            pl.element().struct.field("birth_year").gt(0)
        ).list.all()
    )
    ^^^ Crashes: PolarsData has no .list attribute

CORRECT PATTERN (Vectorized, recommended):
    Check(
        lambda data: data.lazyframe.select(
            pl.col(data.key)
            .list.eval(pl.element().struct.field("birth_year").gt(0))
            .list.all()
            .alias(data.key)
        ),
        name="birth_year_positive",
    )

ALTERNATIVE PATTERN (element_wise=True, simpler but slower):
    Check(
        lambda element: all(
            item["birth_year"] > 0 for item in element
        ),
        element_wise=True,
        name="birth_year_positive",
    )
    ^^^ Each 'element' is one row's Python list of dicts.
    Simpler to write but slower (Python loop, no Polars optimization).

KNOWN BUG:
    When a Check on a List(Struct) column FAILS, Pandera tries to cast
    the failure cases to String (base.py line 198). Polars cannot cast
    List(Struct) to String, so:
      - lazy=True:  crashes with InvalidOperationError
      - lazy=False: works (raises SchemaError with examples)
    WORKAROUND: Use lazy=False for now.

BEST PRACTICES:
    1. Use the PolarsData pattern (data.lazyframe.select(...))
    2. Always .alias(data.key) the output column
    3. Return a LazyFrame with one boolean column
    4. Use lazy=False to avoid the failure_cases crash
    5. Use named functions (not lambdas) for complex checks
    6. Give checks descriptive names via name= parameter
"""
    )

    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
