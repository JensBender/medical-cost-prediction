"""Shared data-loading helpers for analysis and benchmarking."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.constants import ID_COLUMN, MEPS_MISSING_CODES, TARGET_COLUMN, WEIGHT_COLUMN


def load_preprocessor_input_split(
    preprocessor_input_path,
    *,
    audit_columns: Sequence[str] = (),
    raw_data_path=None,
):
    """Load aligned preprocessor inputs, target, weights, and optional audit columns.

    The saved preprocessor-input data contains cleaned feature values before
    scaling, encoding, and imputation. Audit-only columns are loaded from the raw
    MEPS file and aligned by ``DUPERSID`` without adding them to model inputs. In
    this project, this is used to add ``RACETHX`` for the subgroup fairness audit;
    the generic ``audit_columns`` argument leaves room for other audit variables.

    Args:
        preprocessor_input_path: Path to a saved preprocessor-input Parquet split.
        audit_columns: Raw MEPS columns to append for analysis only.
        raw_data_path: Path to the raw MEPS SAS file. Required when audit columns
            are requested.

    Returns:
        A tuple of aligned feature, target, and sample-weight objects.
    """
    split_data = pd.read_parquet(preprocessor_input_path)
    required_columns = {TARGET_COLUMN, WEIGHT_COLUMN}
    missing_columns = required_columns.difference(split_data.columns)
    if missing_columns:
        raise ValueError(
            "Preprocessor-input split is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    split_data.index = split_data.index.astype(str)
    if split_data.index.has_duplicates:
        raise ValueError("Preprocessor-input split contains duplicate row IDs.")

    features = split_data.drop(columns=[TARGET_COLUMN, WEIGHT_COLUMN]).copy()
    target = split_data[TARGET_COLUMN].copy()
    sample_weight = split_data[WEIGHT_COLUMN].copy()

    audit_columns = list(dict.fromkeys(audit_columns))
    if not audit_columns:
        return features, target, sample_weight

    if raw_data_path is None:
        raise ValueError("raw_data_path is required when audit columns are requested.")

    overlapping_columns = set(audit_columns).intersection(features.columns)
    if overlapping_columns:
        raise ValueError(
            "Audit columns already exist in the preprocessor inputs: "
            f"{sorted(overlapping_columns)}"
        )

    raw_data = pd.read_sas(raw_data_path, format="sas7bdat", encoding="latin1")
    required_raw_columns = [ID_COLUMN, *audit_columns]
    missing_raw_columns = set(required_raw_columns).difference(raw_data.columns)
    if missing_raw_columns:
        raise ValueError(
            "Raw MEPS data is missing requested audit columns: "
            f"{sorted(missing_raw_columns)}"
        )

    audit_data = raw_data.loc[:, required_raw_columns].copy()
    audit_data[ID_COLUMN] = audit_data[ID_COLUMN].astype(str)
    audit_data.set_index(ID_COLUMN, inplace=True)
    if audit_data.index.has_duplicates:
        raise ValueError("Raw MEPS data contains duplicate row IDs.")

    audit_data = audit_data.loc[:, audit_columns].replace(MEPS_MISSING_CODES, np.nan)
    missing_ids = features.index.difference(audit_data.index)
    if not missing_ids.empty:
        raise ValueError(
            "Raw MEPS data is missing "
            f"{len(missing_ids):,} rows from the preprocessor-input split."
        )

    features = features.join(
        audit_data.reindex(features.index),
        how="left",
        validate="one_to_one",
    )
    return features, target, sample_weight
