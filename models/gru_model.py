"""GRU traffic-flow forecast baseline for COS30019 A2B.

Expected processed CSV columns: scats_site, datetime, traffic_volume_15min
Datetime is a naive local ISO timestamp, for example 2006-10-04T08:15:00.
Forecast semantics: observations through `as_of` -> prediction at
`as_of + horizon_minutes`, where horizon is a multiple of 15 minutes.
This module trains a model for ONE site and ONE horizon at a time.
Production integration across all SCATS sites is still future work.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

INTERVAL_MINUTES = 15


@dataclass
class TrainingSettings:
    site: str
    horizon_minutes: int = 15
    lookback: int = 8
    hidden_size: int = 32
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    seed: int = 42
    patience: int = 6


class GRUTrafficModel(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(x)
        return self.output(states[:, -1, :]).flatten()


def validate_horizon(horizon_minutes: int) -> int:
    if not isinstance(horizon_minutes, int) or horizon_minutes < 15 or horizon_minutes % 15:
        raise ValueError('horizon_minutes must be a positive multiple of 15')
    return horizon_minutes // INTERVAL_MINUTES


def read_site_series(csv_path: str | Path, scats_site: str | int):
    """Read a single site, rejecting duplicate/invalid dates and invalid volumes.

    Incomplete 15-minute sequences are skipped at window construction time.
    Data cleaning/imputation remains Darren's responsibility.
    """
    site = str(scats_site).strip()
    result = {}
    with open(csv_path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'scats_site', 'datetime', 'traffic_volume_15min'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV requires columns {sorted(required)}')
        for line_num, row in enumerate(reader, start=2):
            if str(row['scats_site']).strip() != site:
                continue
            timestamp = datetime.fromisoformat(row['datetime'].strip())
            if timestamp.tzinfo is not None:
                raise ValueError('Use local ISO timestamps without timezone offsets')
            if timestamp.minute % 15 or timestamp.second or timestamp.microsecond:
                raise ValueError(f'Non-quarter-hour timestamp at CSV row {line_num}')
            value = float(row['traffic_volume_15min'])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid traffic count at CSV row {line_num}')
            if timestamp in result:
                raise ValueError(f'Duplicate timestamp {timestamp.isoformat()} for site {site}')
            result[timestamp] = value
    if not result:
        raise ValueError(f'No records found for SCATS site {site}')
    return sorted(result.items())


def build_windows(series, lookback: int, horizon_steps: int):
    """Return (last_observation_time, target_time, past_volumes, target).

    Every required time step, including the prediction horizon, must exist.
    This prevents a missing observation from being treated as a 15-minute step.
    """
    if lookback < 1:
        raise ValueError('lookback must be positive')
    points = dict(series)
    samples = []
    delta = timedelta(minutes=INTERVAL_MINUTES)
    for target_time, target in series:
        as_of = target_time - horizon_steps * delta
        inputs = [as_of - (lookback - 1 - i) * delta for i in range(lookback)]
        required = inputs + [as_of + i*delta for i in range(1, horizon_steps + 1)]
        if not all(t in points for t in required):
            continue
        samples.append((as_of, target_time, [points[t] for t in inputs], target))
    return samples


def _mae_rmse(predicted, actual):
    if not actual: return {'mae': None, 'rmse': None, 'count': 0}
    errs = [p-a for p, a in zip(predicted, actual)]
    return {'mae': sum(abs(x) for x in errs)/len(errs),
            'rmse': math.sqrt(sum(x*x for x in errs)/len(errs)),
            'count': len(errs)}


def train_gru(processed_csv: str | Path, output_folder: str | Path, settings: TrainingSettings):
    """Train with a time-ordered 70/15/15 target-time split.

    Scaling is fitted exclusively on TRAINING HISTORY. Validation and test
    input windows may use earlier observed values, which is legitimate.
    """
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    horizon_steps = validate_horizon(settings.horizon_minutes)
    series = read_site_series(processed_csv, settings.site)
    if len(series) < 60:
        raise ValueError('Need at least 60 observations for a meaningful baseline split')
    train_cut = int(len(series) * .70)
    valid_cut = int(len(series) * .85)
    training_vals = [v for _,v in series[:train_cut]]
    mean = sum(training_vals)/len(training_vals)
    std = math.sqrt(sum((v-mean)**2 for v in training_vals)/len(training_vals)) or 1.0
    first_val_timestamp = series[train_cut][0]
    first_test_timestamp = series[valid_cut][0]
    windows = build_windows(series, settings.lookback, horizon_steps)
    train = [w for w in windows if w[1] < first_val_timestamp]
    valid = [w for w in windows if first_val_timestamp <= w[1] < first_test_timestamp]
    test = [w for w in windows if w[1] >= first_test_timestamp]
    if not all((train, valid, test)):
        raise ValueError('Not enough contiguous data in all chronological splits')

    def to_dataset(items):
        xs = torch.tensor([[[ (v-mean)/std ] for v in w[2]] for w in items], dtype=torch.float32)
        ys = torch.tensor([(w[3]-mean)/std for w in items], dtype=torch.float32)
        return TensorDataset(xs, ys)

    train_dl = DataLoader(to_dataset(train), batch_size=settings.batch_size, shuffle=True)
    valid_x, valid_y = to_dataset(valid).tensors
    model = GRUTrafficModel(settings.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loss_fn = nn.MSELoss()
    best_loss, best_state, stale = float('inf'), None, 0
    loss_history = []
    for epoch in range(settings.epochs):
        model.train()
        for xs, ys in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(xs), ys)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(valid_x), valid_y).item()
        loss_history.append({'epoch': epoch+1, 'validation_mse_scaled': validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss, stale = validation_loss, 0
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= settings.patience:
            break
    if best_state is None: raise RuntimeError('Training did not complete')
    model.load_state_dict(best_state)

    def evaluate(items):
        x, _ = to_dataset(items).tensors
        model.eval()
        with torch.no_grad():
            pred = [max(0.0, float(v)*std + mean) for v in model(x).tolist()]
        return _mae_rmse(pred, [w[3] for w in items])

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'gru_site_{settings.site}_h{settings.horizon_minutes}'
    checkpoint_path = output_dir / f'{filename}.pt'
    metadata_path = output_dir / f'{filename}.json'
    torch.save(best_state, checkpoint_path)
    metadata = {
        'settings': asdict(settings), 'training_mean': mean, 'training_std': std,
        'train_samples': len(train), 'validation_samples': len(valid),
        'test_samples': len(test), 'train_end': series[train_cut-1][0].isoformat(),
        'validation_start': first_val_timestamp.isoformat(),
        'test_start': first_test_timestamp.isoformat(),
        'validation_metrics': evaluate(valid), 'test_metrics': evaluate(test),
        'loss_history': loss_history,
        'prediction_semantics': 'as_of_time + horizon_minutes',
        'output_units': 'vehicles per 15 minutes',
        'note': 'Baseline; test metrics must not be used for hyperparameter selection',
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return {'weights_path': str(checkpoint_path), 'metadata_path': str(metadata_path),
            'validation': metadata['validation_metrics'], 'test': metadata['test_metrics']}


def predict_gru(scats_site: str | int, as_of_time: str | datetime,
                horizon_minutes: int, processed_csv: str | Path,
                model_dir: str | Path = 'saved_models') -> float:
    """Predict count at as_of + horizon, given historical observations through as_of.

    For a trained historical model this requires observed inputs at `as_of`.
    An arbitrary future date with no history will raise ValueError.
    """
    validate_horizon(horizon_minutes)
    site = str(scats_site).strip()
    root = Path(model_dir)
    prefix = f'gru_site_{site}_h{horizon_minutes}'
    metadata = json.loads((root / f'{prefix}.json').read_text(encoding='utf-8'))
    cfg = metadata['settings']
    as_of = (datetime.fromisoformat(as_of_time) if isinstance(as_of_time, str) else as_of_time)
    if as_of.tzinfo is not None: raise ValueError('as_of_time must be a naive local timestamp')
    observations = dict(read_site_series(processed_csv, site))
    sequence = [as_of - timedelta(minutes=INTERVAL_MINUTES * i) for i in reversed(range(cfg['lookback']))]
    if any(t not in observations for t in sequence):
        raise ValueError('Insufficient contiguous historical traffic observations for this time')
    mean, std = metadata['training_mean'], metadata['training_std']
    x = torch.tensor([[[ (observations[t]-mean)/std ] for t in sequence]], dtype=torch.float32)
    model = GRUTrafficModel(cfg['hidden_size'])
    model.load_state_dict(torch.load(root / f'{prefix}.pt', map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        y = model(x).item() * std + mean
    return max(0., float(y))


def main():
    parser = argparse.ArgumentParser(description='Train or infer a one-site A2B GRU baseline')
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--csv', required=True, help='Processed SCATS CSV path')
    parser.add_argument('--site', required=True)
    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--models-dir', default='saved_models')
    parser.add_argument('--lookback', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--time', help='Forecast origin: 2006-10-15T10:00:00')
    args = parser.parse_args()
    if args.command == 'train':
        result = train_gru(args.csv, args.models_dir,
                           TrainingSettings(site=args.site, horizon_minutes=args.horizon,
                                            lookback=args.lookback, epochs=args.epochs))
        print(json.dumps(result, indent=2))
    else:
        if not args.time: parser.error('predict requires --time')
        result = predict_gru(args.site, args.time, args.horizon, args.csv, args.models_dir)
        print(json.dumps({'scats_site': args.site, 'as_of': args.time,
                          'horizon_minutes': args.horizon,
                          'predicted_vehicles_per_15min': result}, indent=2))


if __name__ == '__main__':
    main()
