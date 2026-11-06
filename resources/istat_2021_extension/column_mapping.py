"""Validated source-to-canonical column mapping for workflow CSV inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def normalize_mapping(
    source_to_canonical: Mapping[str, str] | None,
    label: str,
) -> dict[str, str]:
    """Validate and normalize a ``source name -> canonical name`` mapping."""
    if source_to_canonical is None:
        return {}
    if not isinstance(source_to_canonical, Mapping):
        raise TypeError(f"{label} column mapping must be a JSON object")

    normalized: dict[str, str] = {}
    for source, canonical in source_to_canonical.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{label} mapping contains an empty/non-string source name")
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(
                f"{label} mapping for {source!r} has an empty/non-string target"
            )
        source = source.strip()
        canonical = canonical.strip()
        if source in normalized and normalized[source] != canonical:
            raise ValueError(f"{label} maps source column {source!r} more than once")
        normalized[source] = canonical

    targets: dict[str, list[str]] = {}
    for source, canonical in normalized.items():
        targets.setdefault(canonical, []).append(source)
    duplicates = {
        canonical: sources
        for canonical, sources in targets.items()
        if len(sources) > 1
    }
    if duplicates:
        raise ValueError(
            f"{label} maps multiple source columns to one canonical column: "
            f"{duplicates}"
        )
    return normalized


def read_mapped_csv(
    path: Path,
    required_columns: Sequence[str],
    source_to_canonical: Mapping[str, str] | None,
    label: str,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read required canonical fields without modifying the source dataset.

    Only columns needed by the caller are loaded. A mapped source column takes
    the place of its canonical target. If both names occur in the source file,
    the input is rejected as ambiguous rather than choosing one silently.
    """
    mapping = normalize_mapping(source_to_canonical, label)
    inverse = {canonical: source for source, canonical in mapping.items()}
    header_kwargs = dict(read_csv_kwargs)
    header_kwargs.pop("usecols", None)
    header = pd.read_csv(path, nrows=0, **header_kwargs).columns.tolist()
    header_set = set(header)

    source_columns: list[str] = []
    missing: list[dict[str, str]] = []
    ambiguous: list[dict[str, str]] = []
    for canonical in required_columns:
        source = inverse.get(canonical, canonical)
        if source not in header_set:
            missing.append({"canonical": canonical, "expected_source": source})
            continue
        if source != canonical and canonical in header_set:
            ambiguous.append({"canonical": canonical, "mapped_source": source})
        source_columns.append(source)

    if missing:
        raise ValueError(f"{label} is missing mapped required columns: {missing}")
    if ambiguous:
        raise ValueError(
            f"{label} contains both canonical and mapped aliases: {ambiguous}"
        )
    if len(source_columns) != len(set(source_columns)):
        raise ValueError(f"{label} mapping resolves required fields ambiguously")

    frame = pd.read_csv(path, usecols=source_columns, **read_csv_kwargs)
    applicable = {source: mapping[source] for source in source_columns if source in mapping}
    frame = frame.rename(columns=applicable)
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{label} produced duplicate canonical columns: {duplicates}")
    return frame[list(required_columns)]
