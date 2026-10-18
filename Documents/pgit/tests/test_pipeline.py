"""Tests run with `python -m unittest discover -s tests` (or pytest, if installed)."""

import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src import feedback
from src.data import make_synthetic_bank, split_xy, three_way_split
from src.evaluation import (
    Economics,
    best_threshold,
    campaign_summary,
    classification_metrics,
    realized_profit,
    threshold_sweep,
)
from src.predict import load_bundle, predict_proba, recommend
from src.train import run_experiment


class FakeAnalyzer:
    """Stands in for VADER so tests do not need vaderSentiment installed."""

    def score_text(self, text):
        t = text.lower()
        compound = 0.8 if ("happy" in t or "great" in t or "love" in t or "excellent" in t) else (
            -0.8 if ("terrible" in t or "awful" in t or "bad" in t) else 0.0
        )
        return {"compound": compound, "pos": max(compound, 0), "neu": 1 - abs(compound),
                "neg": max(-compound, 0), "sentiment_label": "x"}


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_bank(4000, seed=1)
        self.X, self.y = split_xy(self.df)

    def test_leakage_column_removed(self):
        self.assertNotIn("duration", self.X.columns)
        self.assertNotIn("y", self.X.columns)

    def test_sizes_disjoint_and_stratified(self):
        s = three_way_split(self.X, self.y)
        n = len(self.X)
        self.assertAlmostEqual(len(s.X_test) / n, 0.2, delta=0.01)
        self.assertAlmostEqual(len(s.X_val) / n, 0.2, delta=0.01)
        idx = [set(s.X_train.index), set(s.X_val.index), set(s.X_test.index)]
        self.assertFalse(idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])
        self.assertEqual(sum(len(i) for i in idx), n)
        for part in (s.y_train, s.y_val, s.y_test):
            self.assertAlmostEqual(part.mean(), self.y.mean(), delta=0.01)


class EvaluationTests(unittest.TestCase):
    def test_economics_break_even(self):
        self.assertAlmostEqual(Economics(2, 20).break_even_probability, 0.1)
        with self.assertRaises(ValueError):
            Economics(1, 0)

    def test_realized_profit_by_hand(self):
        y = np.array([1, 0, 1, 0, 0])
        contacted = np.array([True, True, False, True, False])
        # conversions=1 (value 10), contacted=3 (cost 1 each) -> 10 - 3
        self.assertEqual(realized_profit(y, contacted, Economics(1, 10)), 7.0)

    def test_metrics_by_hand(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.4, 0.6, 0.1])
        m = classification_metrics(y, p, 0.5)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["f1"], 0.5)

    def test_profit_threshold_matches_break_even_when_calibrated(self):
        rng = np.random.default_rng(0)
        p = rng.random(200_000)
        y = (rng.random(200_000) < p).astype(int)  # perfectly calibrated
        econ = Economics(2, 10)  # break-even 0.2
        t, _ = best_threshold(threshold_sweep(y, p, econ), "profit")
        self.assertAlmostEqual(t, econ.break_even_probability, delta=0.03)

    def test_campaign_summary_baselines(self):
        y = np.array([1, 0, 0, 0])
        p = np.array([0.9, 0.2, 0.1, 0.1])
        s = campaign_summary(y, p, Economics(1, 10), 0.5)
        self.assertEqual(s["model_profit"], 9.0)
        self.assertEqual(s["contact_everyone_profit"], 10 - 4)
        self.assertEqual(s["uplift_vs_everyone"], 3.0)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.df = make_synthetic_bank(3000, seed=2)
        cls.result = run_experiment(
            cls.df, model_names=["logreg", "hist_gb"], n_iter=2, cv_folds=3,
            out_dir=cls.tmp.name, verbose=False, n_jobs=1,
        )
        cls.bundle = load_bundle(f"{cls.tmp.name}/models/campaign_model.joblib")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_report_structure(self):
        r = self.result["report"]
        self.assertIn(r["selected_model"], {"logreg", "hist_gb"})
        self.assertEqual(r["protocol"]["split"]["test"], 600)
        self.assertGreater(r["test_metrics"]["roc_auc"], 0.5)
        self.assertEqual(r["test_metrics"]["threshold"], r["threshold"])

    def test_threshold_not_chosen_on_test(self):
        """Threshold equals the optimum of the out-of-fold sweep, which never saw test rows."""
        sweep = self.result["sweep"]
        t, _ = best_threshold(sweep, "f1")
        self.assertEqual(t, self.result["report"]["threshold"])

    def test_features_exclude_leakage(self):
        self.assertNotIn("duration", self.bundle["feature_columns"])

    def test_predict_and_recommend_rule(self):
        sample = self.df.drop(columns=["y"]).head(50)
        proba = predict_proba(self.bundle, sample)
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        econ = Economics(1, 10)
        out = recommend(self.bundle, sample, econ)
        np.testing.assert_array_equal(out["contact"], proba * 10 - 1 > 0)
        capped = recommend(self.bundle, sample, Economics(0.0, 10), max_contacts=5)
        self.assertEqual(int(capped["contact"].sum()), 5)

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            predict_proba(self.bundle, self.df.drop(columns=["age"]).head(3))


class SentimentMergeTests(unittest.TestCase):
    def test_sentiment_features_and_pipeline(self):
        with mock.patch.object(feedback, "_default_analyzer", return_value=FakeAnalyzer()):
            f = feedback.SentimentFeatures().fit([""])
            out = f.transform(pd.DataFrame({"comment": ["I love it", "", None, "awful"]}))
            self.assertEqual(out.shape, (4, 5))
            self.assertEqual(out[1].tolist(), [0, 0, 0, 0, 0])  # empty -> zeros, has_comment 0
            self.assertEqual(out[2].tolist(), [0, 0, 0, 0, 0])
            self.assertEqual(out[0][4], 1.0)
            self.assertGreater(out[0][0], 0)
            self.assertLess(out[3][0], 0)

            df = make_synthetic_bank(2500, seed=3, with_comments=True)
            res = run_experiment(df, model_names=["logreg"], n_iter=1, cv_folds=3,
                                 text_column="comment", verbose=False, n_jobs=1)
            bundle = res["bundle"]
            self.assertEqual(bundle["text_column"], "comment")
            self.assertNotIn("comment", bundle["numeric"] + bundle["categorical"])
            # comment is optional at inference time
            sample = df.drop(columns=["y", "comment"]).head(5)
            self.assertEqual(len(predict_proba(bundle, sample)), 5)


if __name__ == "__main__":
    unittest.main()
