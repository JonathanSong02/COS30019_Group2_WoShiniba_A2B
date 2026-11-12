"""Tests for data_processing: grid shape, timestamps, splits and leakage.

The data-dependent tests need data/processed/ to exist; run
    python -m data_processing.scats_preprocess
first, otherwise they are skipped.
"""
import os
import unittest

import numpy as np
import pandas as pd

from data_processing.dataset import (
    SPLIT_DAYS, split_of, fit_minmax, apply_minmax, inverse_minmax,
    load_series, load_scaler, make_windows,
)

PROCESSED = os.path.join("data", "processed")
SERIES = os.path.join(PROCESSED, "scats_15min.csv")


def _toy_series():
    """Two sites, whole October at 15-min steps, value = step index."""
    times = pd.date_range("2006-10-01", "2006-10-31 23:45", freq="15min")
    frames = [pd.DataFrame({"scats_site": s, "datetime": times,
                            "traffic_volume_15min": np.arange(len(times), dtype=float)})
              for s in (100, 200)]
    return pd.concat(frames, ignore_index=True)


class SplitAndScalerTests(unittest.TestCase):
    def test_splits_are_ordered_and_do_not_overlap(self):
        (_, tr_end), (va_start, va_end), (te_start, _) = (
            SPLIT_DAYS["train"], SPLIT_DAYS["val"], SPLIT_DAYS["test"])
        self.assertLess(tr_end, va_start)
        self.assertLess(va_end, te_start)

    def test_every_october_timestamp_has_a_split(self):
        times = pd.date_range("2006-10-01", "2006-10-31 23:45", freq="15min")
        self.assertNotIn("unassigned", set(split_of(times)))

    def test_scaler_round_trip(self):
        scaler = fit_minmax([0, 50, 200])
        self.assertTrue(np.allclose(inverse_minmax(apply_minmax([10, 150], scaler), scaler), [10, 150]))

    def test_scaler_ignores_values_outside_train(self):
        df = _toy_series()
        train = df.loc[split_of(df["datetime"]) == "train", "traffic_volume_15min"]
        scaler = fit_minmax(train)
        self.assertLess(scaler["max"], df["traffic_volume_15min"].max())


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.df = _toy_series()
        self.scaler = fit_minmax(self.df["traffic_volume_15min"])

    def test_window_targets_stay_in_their_split(self):
        for name in ("train", "val", "test"):
            _, _, meta = make_windows(self.df, name, self.scaler, lags=12)
            self.assertEqual(set(split_of(meta["datetime"])), {name})

    def test_window_inputs_are_the_previous_steps(self):
        X, y, _ = make_windows(self.df, "train", self.scaler, lags=4)
        raw_x, raw_y = inverse_minmax(X[0], self.scaler), inverse_minmax(y[0], self.scaler)
        self.assertTrue(np.allclose(raw_x, raw_y - np.array([4, 3, 2, 1])))

    def test_windows_with_gaps_are_dropped(self):
        df = self.df.copy()
        df.loc[(df["scats_site"] == 100) & (df["datetime"] == "2006-10-02 12:00"),
               "traffic_volume_15min"] = np.nan
        full = len(make_windows(self.df, "train", self.scaler, lags=12, sites={100})[0])
        gapped = len(make_windows(df, "train", self.scaler, lags=12, sites={100})[0])
        self.assertEqual(full - gapped, 13)  # the target itself + 12 windows using it as input


@unittest.skipUnless(os.path.exists(SERIES), "run scats_preprocess first")
class ProcessedFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_series(SERIES)
        cls.scaler = load_scaler(os.path.join(PROCESSED, "scaler.json"))

    def test_columns_match_agreed_format(self):
        self.assertEqual(list(pd.read_csv(SERIES, nrows=1).columns),
                         ["scats_site", "datetime", "traffic_volume_15min"])

    def test_full_grid_40_sites_by_2976_steps(self):
        self.assertEqual(self.df["scats_site"].nunique(), 40)
        self.assertTrue((self.df.groupby("scats_site").size() == 31 * 96).all())

    def test_no_duplicate_site_timestamps(self):
        self.assertFalse(self.df.duplicated(["scats_site", "datetime"]).any())

    def test_timestamps_start_at_midnight_in_15_min_steps(self):
        site = self.df[self.df["scats_site"] == self.df["scats_site"].iloc[0]]
        self.assertEqual(site["datetime"].iloc[0], pd.Timestamp("2006-10-01 00:00"))
        self.assertTrue((site["datetime"].diff().dropna() == pd.Timedelta("15min")).all())

    def test_volumes_are_non_negative(self):
        self.assertTrue((self.df["traffic_volume_15min"].dropna() >= 0).all())

    def test_saved_scaler_was_fitted_on_train_only(self):
        train = self.df.loc[split_of(self.df["datetime"]) == "train", "traffic_volume_15min"]
        self.assertEqual(self.scaler["fitted_on"], "train")
        self.assertEqual(self.scaler["max"], train.max())


if __name__ == "__main__":
    unittest.main()
