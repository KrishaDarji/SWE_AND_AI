#!/usr/bin/env python3
import csv
import math
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(HERE, "..", "source"))
from choose_configuration import SelectionError, choose_configuration

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


def fixture(rows):
    handle = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    handle.close()
    return handle.name


class ChooseTests(unittest.TestCase):
    def test_cap_boundary_and_best(self):
        path = fixture([{"MRE-": "0.2", "ACC+": "0.5", "PRED40+": "0.5", "N_estimators": "100", "criterion": "a", "Min_sample_leaves": "2", "Min_impurity_decrease": "0", "Max_depth": "3"}, {"MRE-": "0.1", "ACC+": "0.6", "PRED40+": "0.6", "N_estimators": "101", "criterion": "b", "Min_sample_leaves": "2", "Min_impurity_decrease": "0", "Max_depth": "3"}])
        try:
            result = choose_configuration(path, 100)
            self.assertEqual(result["feasible_count"], 1)
            self.assertEqual(result["selected"]["record_id"], 1)
        finally: os.unlink(path)

    def test_tie_breakers_and_earliest_record(self):
        rows = [{"MRE-": "0.1", "ACC+": "0.5", "PRED40+": "0.9", "N_estimators": "1", "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "1"}, {"MRE-": "0.1", "ACC+": "0.6", "PRED40+": "0.1", "N_estimators": "2", "criterion": "b", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "1"}, {"MRE-": "0.1", "ACC+": "0.6", "PRED40+": "0.2", "N_estimators": "3", "criterion": "c", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "1"}, {"MRE-": "0.1", "ACC+": "0.6", "PRED40+": "0.2", "N_estimators": "4", "criterion": "d", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "1"}]
        path = fixture(rows)
        try: self.assertEqual(choose_configuration(path, 100)["selected"]["record_id"], 3)
        finally: os.unlink(path)

    def test_invalid_caps(self):
        path = fixture([])
        try:
            for cap in (-1, "inf", math.inf):
                with self.assertRaises(SelectionError): choose_configuration(path, cap)
        finally: os.unlink(path)

    def test_missing_nonfinite_and_criterion(self):
        for field, value in [("MRE-", "nan"), ("criterion", ""), ("N_estimators", "")]:
            row = {key: "1" for key in FIELDS}; row["criterion"] = "ok"; row[field] = value
            path = fixture([row])
            try: self.assertRaises(SelectionError if field == "MRE-" else SelectionError, choose_configuration, path, 100)
            finally: os.unlink(path)

    def test_empty_and_no_feasible(self):
        path = fixture([])
        try: self.assertRaises(SelectionError, choose_configuration, path, 100)
        finally: os.unlink(path)
        path = fixture([{"MRE-": "1", "ACC+": "1", "PRED40+": "1", "N_estimators": "101", "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "1"}])
        try: self.assertRaises(SelectionError, choose_configuration, path, 100)
        finally: os.unlink(path)


if __name__ == "__main__": unittest.main()
