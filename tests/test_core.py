"""
Automated tests for the non-GUI data-processing functions in the NV4GP main
module.

These tests exercise the pure functions used by the marker-filtering module
(filter_markers, filter_genotypes, impute_matrix, maf_of_marker_numeric) and
the error metric (mean_squared_percentage_error). They deliberately avoid
importing/instantiating the Tkinter GUI classes (MarkerFilterApp,
NestedSvrCvFrame, etc.), which require a display and are not suited to
headless CI.

STATUS: this is a starter scaffold, not a finished, fully verified suite.
Several tests assert structural invariants (e.g. "markers kept + markers
removed == markers before") that must hold regardless of implementation
details, so they should pass as-is against the current code. A few areas are
left as TODOs at the bottom where the exact numeric behavior needs to be
confirmed against the real function bodies before asserting specific
expected values.

Run with:
    pip install pytest
    pytest tests/test_core.py -v
"""
import numpy as np
import pandas as pd
import pytest

# TODO: update this import to match the actual module filename after the
# FNV4GP -> NV4GP rename (e.g. `from NV4GP import ...`).
from NV4GP import (
    mean_squared_percentage_error,
    maf_of_marker_numeric,
    filter_markers,
    filter_genotypes,
    impute_matrix,
)


# ---------------------------------------------------------------------------
# mean_squared_percentage_error
# ---------------------------------------------------------------------------

def test_mspe_zero_for_perfect_prediction():
    y_true = np.array([100.0, 200.0, 50.0, 10.0])
    y_pred = y_true.copy()
    assert mean_squared_percentage_error(y_true, y_pred) == pytest.approx(0.0)


def test_mspe_nonnegative():
    y_true = np.array([100.0, 200.0, 50.0])
    y_pred = np.array([90.0, 210.0, 55.0])
    assert mean_squared_percentage_error(y_true, y_pred) >= 0.0


# ---------------------------------------------------------------------------
# maf_of_marker_numeric
# ---------------------------------------------------------------------------

def test_maf_zero_for_monomorphic_marker():
    # A marker with no variation (all individuals homozygous major) has MAF 0.
    monomorphic = pd.Series([2, 2, 2, 2, 2, 2])
    assert maf_of_marker_numeric(monomorphic) == pytest.approx(0.0)


def test_maf_bounded_between_0_and_half():
    marker = pd.Series([0, 0, 1, 2, 2, 2])
    maf = maf_of_marker_numeric(marker)
    assert 0.0 <= maf <= 0.5


# ---------------------------------------------------------------------------
# filter_markers — structural invariants that must hold for any thresholds
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_marker_df():
    # 6 individuals x 4 markers (+ ID column). M4 is monomorphic (MAF 0) and
    # should always be dropped once maf_thr > 0.
    return pd.DataFrame({
        "ID": [f"ind{i}" for i in range(6)],
        "M1": [0, 0, 1, 2, 2, 2],
        "M2": [0, 1, 1, 1, 2, np.nan],
        "M3": [0, 2, 0, 2, 0, 2],
        "M4": [2, 2, 2, 2, 2, 2],
    })


def test_filter_markers_counts_are_consistent(toy_marker_df):
    out_df, summary, details = filter_markers(
        toy_marker_df, maf_thr=0.05, max_missing_mrk=0.5, max_het_mrk=1.0
    )
    assert summary["n_markers_before"] == 4
    assert summary["n_markers_after"] + summary["removed_total"] == summary["n_markers_before"]
    # ID column + surviving markers only
    assert out_df.shape[1] == summary["n_markers_after"] + 1


def test_filter_markers_drops_monomorphic_marker(toy_marker_df):
    out_df, summary, details = filter_markers(
        toy_marker_df, maf_thr=0.05, max_missing_mrk=1.0, max_het_mrk=1.0
    )
    assert "M4" not in out_df.columns


# ---------------------------------------------------------------------------
# filter_genotypes — structural invariants
# ---------------------------------------------------------------------------

def test_filter_genotypes_never_increases_row_count(toy_marker_df):
    out_df, summary, details = filter_genotypes(
        toy_marker_df, max_missing_ind=0.5, max_het_ind=1.0
    )
    assert out_df.shape[0] == summary["n_genotypes_after"]
    assert summary["n_genotypes_after"] <= summary["n_genotypes_before"]


# ---------------------------------------------------------------------------
# impute_matrix
# ---------------------------------------------------------------------------

def test_impute_matrix_none_is_identity(toy_marker_df):
    out = impute_matrix(toy_marker_df, method="None")
    pd.testing.assert_frame_equal(out, toy_marker_df)


def test_impute_matrix_mean_removes_missing_values(toy_marker_df):
    out = impute_matrix(toy_marker_df, method="Mean")
    marker_cols = out.columns[1:]
    assert not out[marker_cols].isna().any().any()


# ---------------------------------------------------------------------------
# TODO: functions not covered here — add tests once behavior is confirmed
# against the real implementation:
#   - detect_sep                  (delimiter inference from file extension/content)
#   - normalize_call              (genotype string normalization / missing tokens)
#   - read_numeric_marker_matrix  (CSV/TSV -> DataFrame round trip)
#   - hapmap_to_numeric_matrix    (HapMap -> numeric matrix transposition)
# ---------------------------------------------------------------------------
