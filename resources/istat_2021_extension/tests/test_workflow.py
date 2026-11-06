"""Automated integrity checks for the regenerated workflow artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from run_workflow import load_config, verify_outputs, workflow_files  # noqa: E402
from column_mapping import read_mapped_csv  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WorkflowArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(WORKSPACE / "workflow_config.json")

    def test_all_cross_stage_checks_pass(self) -> None:
        report = verify_outputs(self.config)
        failures = [name for name, passed in report["checks"].items() if not passed]
        self.assertTrue(report["passed"], f"Failed checks: {failures}")

    def test_manifest_hashes_match_current_outputs(self) -> None:
        manifest_path = (
            self.config["_paths"]["workflow_output_dir"]
            / "workflow_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, path in workflow_files(self.config).items():
            with self.subTest(output=name):
                self.assertEqual(
                    manifest["outputs"][name]["sha256"], sha256_file(path)
                )

    def test_paper_drift_is_reported_not_hidden(self) -> None:
        report = verify_outputs(self.config)
        alignment = report["paper_alignment"]
        self.assertEqual(alignment["anova_terms_compared"], 28)
        self.assertIn("heating_group_count_matches_paper", alignment)
        self.assertIn(
            "electricity_group_count_matches_a_published_count", alignment
        )

    def test_word_report_contains_captioned_tables_and_figures(self) -> None:
        report_path = self.config["_paths"]["report_validation_file"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        counts = report["validation"]["counts"]
        self.assertEqual(counts["figures"], 10)
        self.assertEqual(counts["figure_captions"], 10)
        self.assertEqual(counts["tables"], counts["table_captions"])
        self.assertTrue(
            report["validation"]["checks"][
                "native_svgs_are_embedded_with_png_fallbacks"
            ]
        )


class ColumnMappingTests(unittest.TestCase):
    def test_workflow_config_accepts_separate_raw_and_energy_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_data = json.loads(
                (WORKSPACE / "workflow_config.json").read_text(encoding="utf-8")
            )
            config_data["column_mappings"] = {
                "raw_istat": {
                    "household_key": "id",
                    "region_code": "reg",
                },
                "energy_results": {
                    "annual_appliance_kwh": "appliances_el_kWh"
                },
            }
            path = Path(temporary_directory) / "mapped_config.json"
            path.write_text(json.dumps(config_data), encoding="utf-8")
            loaded = load_config(path)
            self.assertEqual(
                loaded["column_mappings"]["raw_istat"],
                {"household_key": "id", "region_code": "reg"},
            )
            self.assertEqual(
                loaded["column_mappings"]["energy_results"],
                {"annual_appliance_kwh": "appliances_el_kWh"},
            )

    def test_source_names_are_normalized_to_required_canonical_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mapped.csv"
            path.write_text(
                "household_key;household_members;unused\n101;3;x\n",
                encoding="utf-8",
            )
            result = read_mapped_csv(
                path,
                ["id", "q_1_1_sq1"],
                {
                    "household_key": "id",
                    "household_members": "q_1_1_sq1",
                },
                "Synthetic census",
                sep=";",
            )
            self.assertEqual(result.columns.tolist(), ["id", "q_1_1_sq1"])
            self.assertEqual(result.iloc[0].to_dict(), {"id": 101, "q_1_1_sq1": 3})

    def test_canonical_and_alias_columns_are_rejected_as_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ambiguous.csv"
            path.write_text("id;household_key\n101;202\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both canonical and mapped aliases"):
                read_mapped_csv(
                    path,
                    ["id"],
                    {"household_key": "id"},
                    "Synthetic census",
                    sep=";",
                )


if __name__ == "__main__":
    unittest.main()
