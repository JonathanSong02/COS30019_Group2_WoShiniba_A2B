"""GRU traffic-flow forecasting for COS30019 Assignment 2B.

Real SCATS training uses Darren's data_processing.dataset helpers so that
GRU and LSTM share exactly the same calendar splits, training-only scaler,
and valid sample windows. Blank/NaN traffic observations are missing, NEVER zero.

Input CSV: scats_site, datetime, traffic_volume_15min
Forecast: history up to `as_of` predicts the count at `as_of + horizon`.
One model is trained per site, horizon, and sequence length.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

INTERVAL_MINUTES = 15
SUPPORTED_HORIZONS = {15, 30, 60}


@dataclass
class TrainingSettings:
    site: str
    horizon_minutes: int = 15
    lookback: int = 8  # Same concept as LSTM's sequence_length
    hidden_size: int = 32
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    seed: int = 42
    patience: int = 6

    def validate(self):
        validate_horizon(self.horizon_minutes)
        if not isinstance(self.lookback, int) or self.lookback < 1:
            raise ValueError('lookback must be a positive integer')
        if self.hidden_size < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError('hidden_size, epochs and batch_size must be positive')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        if self.patience < 1:
            raise ValueError('patience must be positive')


class GRUTrafficModel(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(x)
        return self.output(states[:, -1, :]).flatten()


def validate_horizon(horizon_minutes: int) -> int:
    if type(horizon_minutes) is not int or horizon_minutes not in SUPPORTED_HORIZONS:
        raise ValueError('horizon_minutes must be 15, 30 or 60')
    return horizon_minutes // INTERVAL_MINUTES


def read_site_series(csv_path: str | Path, scats_site: str | int):
    """Load real-valued observations, skipping missing cells (not imputing zero).

    Any duplicate timestamp, including a repeated missing value, is an error.
    Gaps are retained implicitly in timestamps; only full windows are used.
    """
    site = str(scats_site).strip()
    if not site:
        raise ValueError('SCATS site is required')
    points = {}
    seen = set()
    with open(csv_path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'scats_site', 'datetime', 'traffic_volume_15min'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV requires columns {sorted(required)}')
        for line_number, row in enumerate(reader, start=2):
            if str(row['scats_site']).strip() != site:
                continue
            raw_time = (row['datetime'] or '').strip()
            try:
                timestamp = datetime.fromisoformat(raw_time)
            except ValueError as exc:
                raise ValueError(f'Invalid timestamp at CSV row {line_number}') from exc
            if timestamp.tzinfo is not None or timestamp.minute % 15 or timestamp.second or timestamp.microsecond:
                raise ValueError(f'Invalid 15-minute local timestamp at CSV row {line_number}')
            if timestamp in seen:
                raise ValueError(f'Duplicate timestamp {timestamp.isoformat()} for SCATS site {site}')
            seen.add(timestamp)
            raw_value = (row['traffic_volume_15min'] or '').strip()
            if raw_value.lower() in {'', 'nan', 'na', 'null', 'none'}:
                continue
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f'Invalid traffic count at CSV row {line_number}') from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid traffic count at CSV row {line_number}')
            points[timestamp] = value
    if not seen:
        raise ValueError(f'No records found for SCATS site {site}')
    if not points:
        raise ValueError(f'All observations are missing for SCATS site {site}')
    return sorted(points.items())


def build_windows(series, lookback: int, horizon_steps: int):
    """Return (as_of, target_time, historical_values, target_value).

    Requires each historical 15-minute input and the target to exist. Future
    intermediate timestamps are not inputs, matching shared make_windows.
    """
    if lookback < 1 or horizon_steps < 1:
        raise ValueError('lookback and horizon_steps must be positive')
    points = dict(series)
    delta = timedelta(minutes=INTERVAL_MINUTES)
    samples = []
    for target_time, target_value in series:
        as_of = target_time - horizon_steps * delta
        input_times = [as_of - (lookback - 1 - i)*delta for i in range(lookback)]
        if all(t in points for t in input_times):
            samples.append((as_of, target_time, [points[t] for t in input_times], target_value))
    return samples


def _metrics(predicted, actual):
    if len(actual) != len(predicted) or not actual:
        raise ValueError('predicted and actual must be matching nonempty sequences')
    errors = [p-a for p,a in zip(predicted, actual)]
    rmse = math.sqrt(sum(e*e for e in errors)/len(errors))
    observed_range = max(actual) - min(actual)
    return {
        'count': len(actual),
        'mae': sum(abs(e) for e in errors)/len(errors),
        'rmse': rmse,
        'nrmse': (rmse/observed_range if observed_range else None),
        'nrmse_normalisation': 'actual_range',
    }


def _shared_scaler_path(csv_path, scaler_path):
    if scaler_path is not None:
        resolved = Path(scaler_path)
        if not resolved.exists():
            raise FileNotFoundError(f'Shared scaler file not found: {resolved}')
        return resolved
    candidate = Path(csv_path).parent / 'scaler.json'
    if candidate.exists():
        return candidate
    # Never silently use a different split/scaler for the actual SCATS file.
    if Path(csv_path).name.lower() == 'scats_15min.csv':
        raise FileNotFoundError(f'Real SCATS training requires shared scaler.json alongside {csv_path}')
    return None


def _prepare_shared(processed_csv, site, settings, scaler_path):
    """Use the EXACT preprocessing windows used by the group's LSTM model."""
    from data_processing.dataset import (SPLIT_DAYS, load_scaler, load_series,
                                         make_windows, inverse_minmax)
    config_file = Path(processed_csv).parent / 'split_config.json'
    if config_file.is_file():
        expected = {k: [str(a.date()), str(b.date())] for k,(a,b) in SPLIT_DAYS.items()}
        if json.loads(config_file.read_text(encoding='utf-8')) != expected:
            raise ValueError('split_config.json disagrees with data_processing.dataset.SPLIT_DAYS')
    scaler = load_scaler(scaler_path)
    if (scaler.get('method') != 'minmax' or scaler.get('fitted_on') != 'train'
            or scaler['max'] <= scaler['min']):
        raise ValueError('Shared scaler must be a valid train-only minmax scaler')
    df = load_series(processed_csv)
    site_num = int(site)
    if (df['scats_site'] == site_num).sum() == 0:
        raise ValueError(f'No SCATS data for site {site_num}')
    site_df = df.loc[df['scats_site'] == site_num]
    splits = {}
    for name in ('train','val','test'):
        xs, ys, meta = make_windows(site_df, name, scaler,
                                   lags=settings.lookback,
                                   horizon=settings.horizon_minutes//INTERVAL_MINUTES,
                                   sites={site_num})
        if len(xs) == 0:
            raise ValueError(f'No complete GRU {name} windows for SCATS site {site_num}')
        splits[name] = (torch.tensor(xs, dtype=torch.float32).unsqueeze(-1),
                        torch.tensor(ys, dtype=torch.float32),
                        [t.to_pydatetime() for t in meta['datetime']])
    return splits, scaler, (lambda values: inverse_minmax(values, scaler)), 'shared_calendar_days'


def _prepare_standalone(processed_csv, site, settings):
    """For existing synthetic unit tests only; not for real A2B comparisons."""
    series = read_site_series(processed_csv, site)
    if len(series) < 60:
        raise ValueError('Need at least 60 observations for a standalone baseline')
    train_cut = int(len(series)*0.70)
    valid_cut = int(len(series)*0.85)
    start_val = series[train_cut][0]
    start_test = series[valid_cut][0]
    train_values = [v for _,v in series[:train_cut]]
    minimum, maximum = min(train_values), max(train_values)
    if maximum <= minimum:
        raise ValueError('Cannot scale a constant training series')
    scaler = {'method':'minmax','fitted_on':'train', 'min':minimum, 'max':maximum}
    def scale(v): return (v-minimum)/(maximum-minimum)
    def inverse(values): return [v*(maximum-minimum)+minimum for v in values]
    windows = build_windows(series, settings.lookback, settings.horizon_minutes//INTERVAL_MINUTES)
    parts = {'train':[], 'val':[], 'test':[]}
    for window in windows:
        target_time = window[1]
        name = 'train' if target_time < start_val else ('val' if target_time < start_test else 'test')
        parts[name].append(window)
    splits = {}
    for name, items in parts.items():
        if not items:
            raise ValueError(f'No complete {name} data windows')
        splits[name] = (torch.tensor([[[scale(v)] for v in w[2]] for w in items], dtype=torch.float32),
                        torch.tensor([scale(w[3]) for w in items], dtype=torch.float32),
                        [w[1] for w in items])
    return splits, scaler, inverse, 'standalone_test_only'


def train_gru(processed_csv: str | Path, output_folder: str | Path,
              settings: TrainingSettings, scaler_path: str | Path | None = None):
    """Train one GRU using the shared data split, unless testing a synthetic CSV."""
    settings.validate()
    settings.site = str(settings.site)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    shared_path = _shared_scaler_path(processed_csv, scaler_path)
    if shared_path:
        splits, scaler, inverse, split_method = _prepare_shared(
            processed_csv, settings.site, settings, shared_path)
    else:
        splits, scaler, inverse, split_method = _prepare_standalone(
            processed_csv, settings.site, settings)
    train_x, train_y, _ = splits['train']
    val_x, val_y, _ = splits['val']
    train_loader = DataLoader(TensorDataset(train_x,train_y),
                              batch_size=settings.batch_size, shuffle=True)
    model = GRUTrafficModel(settings.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    objective = nn.MSELoss()
    best_loss, best_weights, stale = float('inf'), None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, settings.epochs+1):
        model.train()
        running = 0.0
        for xs,ys in train_loader:
            optimizer.zero_grad()
            loss = objective(model(xs),ys)
            loss.backward()
            optimizer.step()
            running += loss.item()*len(xs)
        model.eval()
        with torch.no_grad():
            val_loss = objective(model(val_x),val_y).item()
        history.append({'epoch':epoch,
                        'training_mse_scaled':running/len(train_y),
                        'validation_mse_scaled':val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss, stale = val_loss, 0
            best_weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= settings.patience:
            break
    training_seconds = time.perf_counter()-started
    if best_weights is None:
        raise RuntimeError('No valid GRU checkpoint was produced')
    model.load_state_dict(best_weights)
    model.eval()

    def evaluate(name):
        inputs, labels, times = splits[name]
        started_prediction = time.perf_counter()
        with torch.no_grad():
            scaled = model(inputs).tolist()
        prediction_seconds = time.perf_counter()-started_prediction
        predicted = [max(0.0,float(v)) for v in inverse(scaled)]
        actual = [float(v) for v in inverse(labels.tolist())]
        results = _metrics(predicted, actual)
        results['prediction_time_seconds_total'] = prediction_seconds
        results['prediction_time_seconds_per_sample'] = prediction_seconds/len(predicted)
        return results, list(zip(times,actual,predicted))

    validation_metrics, _ = evaluate('val')
    test_metrics, predictions = evaluate('test')
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True,exist_ok=True)
    prefix = f'gru_site_{settings.site}_h{settings.horizon_minutes}_s{settings.lookback}'
    weights = output_dir/f'{prefix}.pt'
    metadata_file = output_dir/f'{prefix}.json'
    prediction_file = output_dir/f'{prefix}_test_predictions.csv'
    torch.save(best_weights,weights)
    with prediction_file.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scats_site','target_datetime','actual_vehicles_15min','predicted_vehicles_15min'])
        writer.writerows((settings.site,t.isoformat(),actual,predicted)
                         for t,actual,predicted in predictions)
    metadata = {
        'model_type':'GRU', 'scats_site':settings.site,
        'settings':asdict(settings), 'scaler':scaler,
        'split_method':split_method,
        'training_samples':len(train_y), 'validation_samples':len(val_y),
        'test_samples':len(splits['test'][1]),
        'training_time_seconds':training_seconds,
        'best_validation_mse_scaled':best_loss,
        'validation_metrics':validation_metrics, 'test_metrics':test_metrics,
        'loss_history':history,
        'prediction_semantics':'as_of_time + horizon_minutes',
        'output_units':'vehicles per 15-minute interval',
        'note':'Do not tune using test metrics; test dataset is held out.',
    }
    metadata_file.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return {'weights_path':str(weights), 'metadata_path':str(metadata_file),
            'test_predictions_path':str(prediction_file),
            'training_time_seconds':training_seconds,
            'validation':validation_metrics,'test':test_metrics}


def predict_gru(scats_site: str | int, as_of_time: str | datetime,
                horizon_minutes: int, processed_csv: str | Path,
                model_dir: str | Path = 'saved_models',
                sequence_length: int = 8,
                scaler_path: str | Path | None = None) -> float:
    """Predict a site's future 15-minute count from available observed history."""
    validate_horizon(horizon_minutes)
    site = str(scats_site).strip()
    root = Path(model_dir)
    prefix = f'gru_site_{site}_h{horizon_minutes}_s{sequence_length}'
    metadata_file = root/f'{prefix}.json'
    weight_file = root/f'{prefix}.pt'
    if not metadata_file.exists() or not weight_file.exists():
        raise FileNotFoundError(f'No GRU model trained for site={site}, horizon={horizon_minutes}, sequence={sequence_length}')
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    if (metadata['scats_site'] != site or
            metadata['settings']['horizon_minutes'] != horizon_minutes or
            metadata['settings']['lookback'] != sequence_length):
        raise ValueError('Saved GRU model does not match the requested settings')
    as_of = datetime.fromisoformat(as_of_time) if isinstance(as_of_time,str) else as_of_time
    if not isinstance(as_of,datetime) or as_of.tzinfo is not None or as_of.minute%15 or as_of.second or as_of.microsecond:
        raise ValueError('as_of_time must be a 15-minute aligned naive local datetime')
    observations = dict(read_site_series(processed_csv,site))
    required = [as_of-timedelta(minutes=15*i) for i in reversed(range(sequence_length))]
    if any(t not in observations for t in required):
        raise ValueError('Insufficient nonmissing historical observations for this time')
    scaler = metadata['scaler']
    if scaler_path is not None and json.loads(Path(scaler_path).read_text(encoding='utf-8')) != scaler:
        raise ValueError('Provided scaler differs from the scaler used for training')
    minimum,maximum = scaler['min'],scaler['max']
    x = torch.tensor([[[ (observations[t]-minimum)/(maximum-minimum)] for t in required]],dtype=torch.float32)
    model = GRUTrafficModel(metadata['settings']['hidden_size'])
    model.load_state_dict(torch.load(weight_file,map_location='cpu',weights_only=True))
    model.eval()
    with torch.no_grad():
        output = model(x).item()*(maximum-minimum)+minimum
    return max(0.0,float(output))


def main():
    parser = argparse.ArgumentParser(description='Train or predict a site-specific A2B GRU')
    parser.add_argument('command',choices=['train','predict'])
    parser.add_argument('--csv',required=True)
    parser.add_argument('--site',required=True)
    parser.add_argument('--horizon',type=int,default=15)
    parser.add_argument('--models-dir',default='saved_models')
    parser.add_argument('--lookback','--sequence-length',dest='lookback',type=int,default=8)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--scaler',default=None,help='Optional path to shared scaler.json')
    parser.add_argument('--time',help='Forecast origin, e.g. 2006-10-27T12:00:00')
    args = parser.parse_args()
    if args.command == 'train':
        result = train_gru(args.csv,args.models_dir,
                           TrainingSettings(site=args.site,horizon_minutes=args.horizon,
                                            lookback=args.lookback,epochs=args.epochs),
                           scaler_path=args.scaler)
        print(json.dumps(result,indent=2))
    else:
        if not args.time:
            parser.error('predict requires --time')
        prediction = predict_gru(args.site,args.time,args.horizon,args.csv,
                                 args.models_dir,sequence_length=args.lookback,
                                 scaler_path=args.scaler)
        print(json.dumps({'scats_site':args.site,'as_of':args.time,
                          'horizon_minutes':args.horizon,
                          'predicted_vehicles_per_15min':prediction},indent=2))


if __name__=='__main__':
    main()
