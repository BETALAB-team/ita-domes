"""Regression checks for the 2013/2021 ISTAT workflow."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from archetype_pipeline import (  # noqa: E402
    AGE_RANGES_2021,
    EDUCATION_YEARS_2021,
    build_household_features,
    randomize_age_classes,
)
from istat_harmonizer import (  # noqa: E402
    CORE_COLUMNS,
    ENERGY_2013_COLUMNS,
    sha256_file,
)
from report_pipeline import FIGURE_STEMS  # noqa: E402


class SurveyProfileTests(unittest.TestCase):
    def test_2021_age_draws_stay_inside_published_classes(self) -> None:
        classes = np.arange(1, 12, dtype=int)
        ages = randomize_age_classes(
            classes,
            np.random.RandomState(42),
            AGE_RANGES_2021,
        )
        for age_class, age in zip(classes, ages, strict=True):
            lower, upper = AGE_RANGES_2021[int(age_class)]
            self.assertGreaterEqual(age, lower)
            self.assertLessEqual(age, upper)

    def test_households_beyond_released_member_columns_are_audited(self) -> None:
        row = {"id": 1, "q_1_1_sq1": 6}
        row.update({f"cletad_{index}": 1 for index in range(1, 13)})
        row.update({f"titstu_{index}": 1 for index in range(1, 13)})
        eligible = dict(row)
        eligible["id"] = 2
        eligible["q_1_1_sq1"] = 1
        features, exclusions = build_household_features(
            pd.DataFrame([row, eligible]),
            age_seed=42,
            available_member_columns=5,
            age_ranges=AGE_RANGES_2021,
            education_years=EDUCATION_YEARS_2021,
            unknown_education_codes=(0,),
            education_age_basis="realized_age",
        )
        self.assertEqual(features["id"].tolist(), [2])
        self.assertEqual(
            exclusions.iloc[0].to_dict(),
            {"id": 1, "reason": "member_count_exceeds_available_columns"},
        )


class MultiYearArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_root = WORKSPACE / "multiyear_outputs"

    def test_both_year_reports_pass_with_expected_figure_sets(self) -> None:
        expected = {2013: FIGURE_STEMS, 2021: FIGURE_STEMS[:7]}
        for year, expected_stems in expected.items():
            with self.subTest(year=year):
                report = json.loads(
                    (self.output_root / str(year) / "validation_report.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(report["passed"])
                self.assertEqual(
                    set(report["stages"]["figures"]["actual_stems"]),
                    set(expected_stems),
                )

    def test_year_manifest_hashes_match_outputs(self) -> None:
        for year in (2013, 2021):
            year_dir = self.output_root / str(year)
            manifest = json.loads(
                (year_dir / "workflow_manifest.json").read_text(encoding="utf-8")
            )
            for relative, metadata in manifest["outputs"].items():
                with self.subTest(year=year, output=relative):
                    self.assertEqual(
                        sha256_file(year_dir / relative), metadata["sha256"]
                    )

    def test_raw_2013_harmonization_matches_legacy_conversion(self) -> None:
        columns = [
            column
            for column in CORE_COLUMNS + ENERGY_2013_COLUMNS
            if column != "survey_year"
        ]
        harmonized = pd.read_csv(
            self.output_root / "2013" / "harmonized_istat.csv",
            sep=";",
            usecols=columns,
        )
        legacy = pd.read_csv(
            WORKSPACE / "istat_microdata_2013.csv",
            sep=";",
            usecols=columns,
        )[columns]
        pd.testing.assert_frame_equal(harmonized, legacy, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
