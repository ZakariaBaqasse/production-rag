"""Module for loading and normalizing curated testsets from CSV files.

This module provides utilities to parse and validate testset data, including
handling various list formats and ensuring required fields are present.
"""
import ast
import json
from pathlib import Path
from typing import Any
import pandas as pd


def _safe_parse_list(value: any) -> list[str]:
    """Parse a list-like CSV field robustly.

    The curated CSV may contain Python-list strings, JSON arrays, real lists,
    or plain text. This helper normalizes all of these into ``list[str]``.
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    return [text]


def load_testset(path: Path) -> list[dict[str, Any]]:
    """Load and normalize the curated testset from CSV.

    Returns rows containing only the fields needed for experiment execution.
    """
    if not path.exists():
        raise FileNotFoundError(f"Testset not found: {path}")

    df = pd.read_csv(path)
    required_columns = {
        "user_input",
        "reference_contexts",
        "reference",
        "question_category",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in testset: {missing}")

    # Defensive cleanup: some files include an accidental duplicated header row.
    if "synthesizer_name" in df.columns:
        df = df[df["synthesizer_name"] != "synthesizer_name"]

    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append(
            {
                "user_input": str(row.get("user_input", "")).strip(),
                "reference": str(row.get("reference", "")).strip(),
                "reference_contexts": _safe_parse_list(row.get("reference_contexts")),
                "question_category": str(row.get("question_category", "")).strip(),
            }
        )

    records = [r for r in records if r["user_input"]]
    if not records:
        raise ValueError("No valid rows found in testset after parsing.")

    return records
