"""Deep semantic audit for the SE-for-AI shrinking-code project.

These checks come from the submission contract, not implementation style. They
verify that the repository can support the claims made in the report: customer
requirements are complete, the ladder is meaningful, the experiment has the
required controls, and results contain evidence for reuse, transfer, quality,
reliability, and statistical comparison.
"""

import csv
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("x264", "nasa93dem", "health_closedissues")
SKILLS = ("remember", "guess", "certify", "flow", "blame", "justify", "choose", "fix", "spot", "race")
TREATMENTS = tuple(range(1, 8))


def requirement_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\w+)\s*\|$", line)
        if match:
            rows.append((int(match.group(1)), match.group(2), match.group(3)))
    return rows


class RequirementSemantics(unittest.TestCase):
    def test_each_graduate_domain_has_twenty_customer_asks(self):
        for domain in DOMAINS:
            path = ROOT / "docs/requirements" / ("health-closedissues.md" if domain == "health_closedissues" else f"{domain}.md")
            with self.subTest(domain=domain):
                rows = requirement_rows(path)
                self.assertEqual(len(rows), 20)
                self.assertEqual([number for number, _, _ in rows], list(range(1, 21)))

    def test_each_domain_tags_every_skill_twice_in_customer_language(self):
        for domain in DOMAINS:
            path = ROOT / "docs/requirements" / ("health-closedissues.md" if domain == "health_closedissues" else f"{domain}.md")
            rows = requirement_rows(path)
            tags = [tag for _, ask, tag in rows]
            with self.subTest(domain=domain):
                self.assertEqual(set(tags), set(SKILLS))
                self.assertEqual({tag: tags.count(tag) for tag in SKILLS}, {tag: 2 for tag in SKILLS})
                self.assertTrue(all(not any(word in ask.lower() for word in ("nsga", "randomforest", "sklearn", "algorithm")) for _, ask, _ in rows))

    def test_requirements_use_real_dataset_domain_terms(self):
        expected = {"x264": "encoding", "nasa93dem": "project", "health_closedissues": "model"}
        for domain, term in expected.items():
            path = ROOT / "docs/requirements" / ("health-closedissues.md" if domain == "health_closedissues" else f"{domain}.md")
            with self.subTest(domain=domain):
                self.assertGreaterEqual(path.read_text(encoding="utf-8").lower().count(term), 2)


class ExperimentDesignSemantics(unittest.TestCase):
    def test_manifest_defines_the_preregistered_seven_treatments(self):
        manifest = json.loads((ROOT / "experiments/treatments.json").read_text())
        self.assertEqual([item["id"] for item in manifest["treatments"]], list(TREATMENTS))
        self.assertEqual(manifest["ladder_steps"], 10)
        self.assertEqual(manifest["minimum_repeats"], 3)

    def test_reuse_pairs_have_matching_background_and_reuse_flags(self):
        treatments = {item["id"]: item for item in json.loads((ROOT / "experiments/treatments.json").read_text())["treatments"]}
        for reuse_id, cold_id in ((2, 1), (4, 3), (6, 5)):
            with self.subTest(pair=(cold_id, reuse_id)):
                self.assertTrue(treatments[reuse_id]["reuse_instructed"])
                self.assertFalse(treatments[cold_id]["reuse_instructed"])
                self.assertEqual(treatments[reuse_id]["background"], treatments[cold_id]["background"])

    def test_treatment7_ladder_has_three_domains_and_ten_skill_numbers(self):
        path = ROOT / "results/treatment7_ladder.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual({item["domain"] for item in data}, set(DOMAINS))
        for item in data:
            with self.subTest(domain=item["domain"]):
                self.assertEqual({step["step"] for step in item["steps"]}, set(range(1, 11)))
                self.assertTrue(all(step.get("payload") for step in item["steps"]))

    def test_results_plan_requires_the_evidence_needed_for_a_semantic_verdict(self):
        text = (ROOT / "result.md").read_text(encoding="utf-8").lower()
        required_terms = ("new-code histogram", "domain-transfer", "reliability", "same", "hypervolume", "pareto")
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, text)


class EvidenceSemantics(unittest.TestCase):
    def test_treatment7_summary_covers_all_domains_and_steps(self):
        path = ROOT / "results/treatment7_summary.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["domain"] for row in rows}, set(DOMAINS))
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                self.assertEqual({int(row["step"]) for row in rows if row["domain"] == domain}, set(range(1, 11)))

    def test_metrics_have_a_machine_readable_contract_for_each_completed_step(self):
        files = sorted((ROOT / "experiments/runs").glob("treatment[1-7]/trials/*/*/step*/metrics.csv"))
        self.assertTrue(files)
        required = {"step", "treatment", "trial", "domain", "skill", "total_loc", "new_loc", "test_pass", "runtime_seconds", "error_count", "notes"}
        for path in files:
            with self.subTest(path=str(path)):
                with path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    self.assertTrue(required.issubset(reader.fieldnames or []))
                    rows = list(reader)
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertTrue(row["skill"] in SKILLS)
                self.assertNotEqual(row["test_pass"], "")

    def test_treatment7_has_explicit_third_domain_evidence(self):
        data = json.loads((ROOT / "results/treatment7_ladder.json").read_text(encoding="utf-8"))
        health = next(item for item in data if item["domain"] == "health_closedissues")
        self.assertEqual(health["rows_used"], 300)
        self.assertEqual({step["step"] for step in health["steps"]}, set(range(1, 11)))


if __name__ == "__main__":
    unittest.main()
