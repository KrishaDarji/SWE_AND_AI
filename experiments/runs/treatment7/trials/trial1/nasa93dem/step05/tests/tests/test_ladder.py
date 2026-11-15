"""Treatment 7 ladder tests: every step runs on every domain adapter.

Test names start with test_stepN_ so the trial runner can select one step
with `python3 -m unittest discover -s tests -k step3_`.
"""

import unittest

from src.core.certify import same
from src.core.datasets import features, sample_rows
from src.domains import DATASETS
from src.ladder import SKILLS, run_step

ROWS = {name: sample_rows(spec, 60) for name, spec in DATASETS.items()}


def results(step):
    return [(spec, run_step(spec, ROWS[name], step)["result"]) for name, spec in DATASETS.items()]


class LadderTests(unittest.TestCase):
    def test_step1_adapters_are_complete(self):
        for spec in DATASETS.values():
            self.assertEqual(sorted(spec.requirements), list(range(1, 11)))
            feature, a, b = spec.contrast
            self.assertIn(feature, features(spec, ROWS[spec.name]))
        self.assertEqual(len(SKILLS), 10)

    def test_step1_ignores_x_columns(self):
        names = features(DATASETS["nasa93dem"], ROWS["nasa93dem"])
        self.assertNotIn("MonthsX", names)
        self.assertNotIn("idX", names)
        self.assertIn("Kloc+", names)

    def test_step1_remember_summarizes_and_ranks(self):
        for spec, result in results(1):
            self.assertEqual(set(result["objectives"]), set(spec.objectives))
            scores = [item["score"] for item in result["best"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_step2_guess_reports_model_and_baseline_error(self):
        for spec, result in results(2):
            for objective in spec.objectives:
                self.assertGreaterEqual(result["errors"][objective]["model_mae"], 0)
                self.assertGreaterEqual(result["errors"][objective]["baseline_mae"], 0)

    def test_step3_same_matches_ezr_semantics(self):
        self.assertTrue(same([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
        self.assertFalse(same([1, 2, 3, 4, 5], [11, 12, 13, 14, 15]))

    def test_step3_certify_runs_on_domain_contrast(self):
        for spec, result in results(3):
            for objective in spec.objectives:
                self.assertIn("same", result[objective])
                self.assertLessEqual(result[objective]["p_value"], 1.0)

    def test_step4_flow_stops_within_available_rows(self):
        for spec, result in results(4):
            self.assertLessEqual(result["seen"], result["available"])

    def test_step5_blame_never_blames_objectives(self):
        for spec, result in results(5):
            self.assertTrue(result["findings"])
            for finding in result["findings"]:
                self.assertNotIn(finding["feature"], spec.objectives)

    def test_step6_justify_mentions_the_client(self):
        for spec, result in results(6):
            self.assertIn(spec.client, result["text"])

    def test_step7_choose_stays_within_budget(self):
        for spec, result in results(7):
            self.assertEqual(result["evaluated"], 20)
            self.assertGreaterEqual(result["beats_fraction_of_all_rows"], 0.5)

    def test_step8_fix_changes_at_most_one_setting(self):
        for spec, result in results(8):
            changed = [key for key in features(spec, ROWS[spec.name])
                       if result["candidate"][key] != result["poor_candidate"][key]]
            self.assertLessEqual(len(changed), 1)
            self.assertGreaterEqual(result["improvement"], 0)

    def test_step9_spot_ranks_unusual_rows(self):
        for spec, result in results(9):
            scores = [finding["score"] for finding in result["findings"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_step10_race_uses_equal_budgets(self):
        for spec, result in results(10):
            self.assertEqual(len(result["shared_scores"]), len(result["baseline_scores"]))
            self.assertIn("same", result)


if __name__ == "__main__":
    unittest.main()
