import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from models.gru_model import (TrainingSettings, build_windows, predict_gru,
                              read_site_series, train_gru, validate_horizon)

class GRUBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv_path = Path(self.tmp.name) / 'processed.csv'
        start = datetime(2006, 10, 1)
        with self.csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['scats_site','datetime','traffic_volume_15min'])
            for i in range(240):
                writer.writerow(['2000', (start + timedelta(minutes=i*15)).isoformat(),
                                 100 + (i%96) + (i%7)*5])
    def test_loading_and_windows(self):
        series = read_site_series(self.csv_path,'2000')
        self.assertEqual(len(series),240)
        windows = build_windows(series, lookback=8, horizon_steps=2)
        self.assertEqual(len(windows),231)
        self.assertEqual((windows[0][1]-windows[0][0]).total_seconds(), 1800)
    def test_bad_horizons(self):
        for value in (0,10,-15):
            with self.assertRaises(ValueError):validate_horizon(value)
    def test_training_and_saved_prediction(self):
        output = Path(self.tmp.name)/'models'
        result = train_gru(self.csv_path, output,
            TrainingSettings(site='2000', horizon_minutes=15, lookback=8,
                             hidden_size=8,epochs=2,batch_size=32))
        self.assertTrue(Path(result['weights_path']).exists())
        self.assertGreater(result['test']['count'],0)
        prediction = predict_gru('2000','2006-10-03T00:00:00',15,self.csv_path,output)
        self.assertGreaterEqual(prediction, 0.0)
        with self.assertRaises(ValueError):
            predict_gru('2000','2026-11-12T00:00:00',15,self.csv_path,output)

if __name__=='__main__': unittest.main()
