import csv
import tempfile
import unittest
from pathlib import Path
from bernanke_1983 import load_csv, regression_design, simulate_data
from historical_data import validate_dates


class DataBoundaries(unittest.TestCase):
    def test_attachment_cannot_be_mislabeled_full_regression(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'attachment.csv'
            path.write_text('date,ip,cpi,baa\n1929-01,7,17,5.6\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'money_growth'):
                load_csv(path)

    def test_gaps_and_duplicates_do_not_become_one_month_lags(self):
        for dates in [('1929-01', '1929-03'), ('1929-01', '1929-01')]:
            with self.assertRaises(ValueError):
                validate_dates([{'date': d} for d in dates])

    def test_initial_lag_is_dropped_instead_of_zero_filled(self):
        data = simulate_data()
        y, a, b = regression_design(data)
        self.assertEqual(len(y), 59)
        self.assertEqual(a.shape[0], b.shape[0])
        self.assertEqual(b[0, 4], data['bank_distress'][0])

    def test_missing_history_is_not_synthesized(self):
        data = simulate_data()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'full.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(data))
                writer.writeheader()
                for i in range(8):
                    row = {k: v[i] for k, v in data.items()}
                    if i == 3: row['bank_distress'] = ''
                    writer.writerow(row)
            with self.assertRaisesRegex(ValueError, '缺失'):
                load_csv(path)


if __name__ == '__main__':
    unittest.main()
