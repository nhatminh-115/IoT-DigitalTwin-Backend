"""PyTorch LSTM forecaster for multivariate one-step-ahead prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PredictorError(RuntimeError):
    """Raised when the predictor cannot fit or infer reliably."""


class _LSTMForecaster(nn.Module):
    """Neural network backbone for sequence-to-vector forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2 if hidden_size >= 2 else 1),
            nn.ReLU(),
            nn.Linear(hidden_size // 2 if hidden_size >= 2 else 1, output_size),
        )

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Runs forward pass over a batch of input windows.

        The LSTM transforms each sequence into hidden states. The final hidden
        state encodes temporal context and is fed to a dense regressor to output
        the next-step multivariate vector.
        """
        lstm_out, _ = self.lstm(batch_x)
        temporal_embedding = lstm_out[:, -1, :]
        return self.regressor(temporal_embedding)


@dataclass
class PredictorConfig:
    """Hyperparameters for LSTM training and inference."""

    sequence_length: int = 20
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    learning_rate: float = 1e-3
    epochs: int = 40
    batch_size: int = 64
    weight_decay: float = 1e-6


class DeepTimeSeriesPredictor:
    """Encapsulates training and one-step inference for multivariate series.

     Stationarization and reconstruction math:
     1. First-order differencing is applied before sequence modeling:
         Delta y_t = y_t - y_{t-1}.
         This reduces non-stationary level effects that often cause the naive
         lag-1 behavior where y_hat_{t+1} ~ y_t.
     2. Differenced features are standardized per channel:
         Delta y'_t = (Delta y_t - mu_delta) / sigma_delta.
     3. The network predicts Delta y_hat_{t+1}. Absolute prediction is recovered by
         y_hat_{t+1} = y_t + Delta y_hat_{t+1}, where y_t is the last observed level.

     Public methods continue returning predictions in the original absolute units
     so Streamlit plotting code remains unchanged.
    """

    def __init__(self, config: PredictorConfig | None = None) -> None:
        """Initializes model state and runtime device."""
        self._preprocessing_mode = "first_order_differencing_v1"
        self.config = config or PredictorConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: _LSTMForecaster | None = None
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._columns: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        """Returns whether model and scaling state are initialized."""
        return (
            self._model is not None
            and self._feature_means is not None
            and self._feature_stds is not None
            and self._columns is not None
        )

    @property
    def feature_columns(self) -> list[str]:
        """Returns fitted feature schema, or an empty list if not fitted."""
        return list(self._columns) if self._columns is not None else []

    def fit(self, dataframe: pd.DataFrame) -> None:
        """Fits the LSTM model on cleaned multivariate historical data.

        Args:
            dataframe: Clean numeric data indexed by time.

        Raises:
            PredictorError: If there are not enough samples for sequence training.
        """
        if dataframe is None or dataframe.empty:
            raise PredictorError("Training data is empty.")

        numeric_df = dataframe.copy()
        numeric_df = numeric_df.select_dtypes(include=[np.number])

        if numeric_df.shape[0] <= self.config.sequence_length + 1:
            raise PredictorError(
                "Not enough rows to train the differenced LSTM with the configured sequence length."
            )

        values = numeric_df.to_numpy(dtype=np.float32)
        differenced = self._difference_values(values)
        means = differenced.mean(axis=0)
        stds = differenced.std(axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)

        standardized = (differenced - means) / stds
        train_x, train_y = self._build_sliding_windows(standardized)

        if train_x.shape[0] == 0:
            raise PredictorError("Sliding window generation produced zero samples.")

        self._model = _LSTMForecaster(
            input_size=train_x.shape[-1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            output_size=train_y.shape[-1],
        ).to(self.device)

        dataset = TensorDataset(
            torch.from_numpy(train_x).float(),
            torch.from_numpy(train_y).float(),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                prediction = self._model(batch_x)
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()

        self._feature_means = means
        self._feature_stds = stds
        self._columns = list(numeric_df.columns)

    def predict_next_step(self, dataframe: pd.DataFrame) -> pd.Series:
        """Predicts the next multivariate step t+1 from the latest sequence.

        Args:
            dataframe: Clean numeric history aligned with training columns.

        Returns:
            A Series of predicted next-step feature values.

        Raises:
            PredictorError: If model is not trained or data shape is invalid.
        """
        if not self.is_fitted:
            raise PredictorError("Model is not fitted. Call fit() before predict_next_step().")

        if dataframe is None or dataframe.empty:
            raise PredictorError("Prediction data is empty.")

        assert self._columns is not None
        aligned = self._validate_and_align_dataframe(dataframe)

        if len(aligned) < self.config.sequence_length + 1:
            raise PredictorError(
                "Insufficient rows for prediction window. "
                f"Need at least {self.config.sequence_length + 1} observations."
            )

        absolute_values = aligned.to_numpy(dtype=np.float32)
        differenced = self._difference_values(absolute_values)
        standardized = self._standardize(differenced)

        last_window = standardized[-self.config.sequence_length :]
        model_input = torch.from_numpy(last_window).float().unsqueeze(0).to(self.device)

        self._model.eval()
        with torch.no_grad():
            output_standardized = self._model(model_input).cpu().numpy().reshape(-1)

        predicted_delta = self._destandardize(output_standardized)
        last_actual = absolute_values[-1]
        predicted_absolute = last_actual + predicted_delta
        return pd.Series(predicted_absolute, index=self._columns, name="t+1_prediction")

    def predict_historical_with_uncertainty(
        self,
        dataframe: pd.DataFrame,
        mc_samples: int = 30,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates delayed one-step predictions and predictive uncertainty.

        For each timestamp t >= sequence_length, the model predicts y_t from
        history y_{t-L} ... y_{t-1}. Epistemic uncertainty is approximated using
        Monte Carlo dropout by repeated stochastic forward passes.

        Args:
            dataframe: Clean numeric history aligned with fitted schema.
            mc_samples: Number of Monte Carlo stochastic passes.

        Returns:
            Tuple of (ground_truth, predicted_mean, predicted_std), each indexed
            by aligned datetime timestamps.

        Raises:
            PredictorError: If fitted state or data shape is invalid.
        """
        if not self.is_fitted:
            raise PredictorError(
                "Model is not fitted. Call fit() before predict_historical_with_uncertainty()."
            )

        assert self._columns is not None
        aligned = self._validate_and_align_dataframe(dataframe)

        if len(aligned) <= self.config.sequence_length + 1:
            raise PredictorError(
                "Insufficient rows for delayed prediction evaluation. "
                f"Need more than {self.config.sequence_length + 1} observations."
            )

        absolute_values = aligned.to_numpy(dtype=np.float32)
        differenced = self._difference_values(absolute_values)
        standardized = self._standardize(differenced)
        train_x, _ = self._build_sliding_windows(standardized)
        if train_x.shape[0] == 0:
            raise PredictorError("No historical windows available for evaluation.")

        pred_mean_std, pred_sigma_std = self._predict_with_mc_dropout(train_x, mc_samples=mc_samples)
        pred_mean_delta = self._destandardize(pred_mean_std)
        pred_sigma = pred_sigma_std * self._feature_stds

        base_levels = absolute_values[self.config.sequence_length : -1]
        pred_mean_absolute = base_levels + pred_mean_delta

        pred_index = aligned.index[self.config.sequence_length + 1 :]
        ground_truth = aligned.iloc[self.config.sequence_length + 1 :].copy()
        predicted_mean = pd.DataFrame(pred_mean_absolute, index=pred_index, columns=self._columns)
        predicted_std = pd.DataFrame(pred_sigma, index=pred_index, columns=self._columns)
        return ground_truth, predicted_mean, predicted_std

    @staticmethod
    def _difference_values(values: np.ndarray) -> np.ndarray:
        """Computes first-order differencing along the time axis.

        For values y with shape (T, F), the output has shape (T-1, F):
            Delta y_t = y_t - y_{t-1}.
        """
        if values.shape[0] < 2:
            return np.empty((0, values.shape[1]), dtype=np.float32)
        return np.diff(values, axis=0).astype(np.float32)

    def _validate_and_align_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Validates schema consistency and returns aligned numeric frame."""
        assert self._columns is not None
        numeric_df = dataframe.copy().select_dtypes(include=[np.number])
        aligned = numeric_df.reindex(columns=self._columns)
        if aligned.isna().sum().sum() > 0:
            raise PredictorError(
                "Prediction data columns are inconsistent with the fitted schema."
            )
        return aligned

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        """Applies z-score transformation with fitted moments."""
        assert self._feature_means is not None
        assert self._feature_stds is not None
        return (values - self._feature_means) / self._feature_stds

    def _destandardize(self, values: np.ndarray) -> np.ndarray:
        """Maps standardized vectors back to original sensor units."""
        assert self._feature_means is not None
        assert self._feature_stds is not None
        return values * self._feature_stds + self._feature_means

    def _predict_with_mc_dropout(
        self,
        batch_windows: np.ndarray,
        mc_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predicts a batch with uncertainty estimated from MC dropout.

        If K stochastic forward passes are sampled, the predictive mean and
        standard deviation are estimated as:
            mu = (1/K) * sum_k f_k(x)
            sigma = sqrt((1/(K-1)) * sum_k (f_k(x) - mu)^2)
        where f_k denotes the network with dropout mask k.
        """
        if mc_samples < 1:
            raise PredictorError("mc_samples must be at least 1.")

        assert self._model is not None
        model_input = torch.from_numpy(batch_windows).float().to(self.device)

        if mc_samples == 1:
            self._model.eval()
            with torch.no_grad():
                deterministic = self._model(model_input).cpu().numpy()
            sigma = np.zeros_like(deterministic)
            return deterministic, sigma

        previous_mode_training = self._model.training
        self._model.train()
        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(mc_samples):
                sample_pred = self._model(model_input).cpu().numpy()
                samples.append(sample_pred)

        stacked = np.stack(samples, axis=0)
        mean_pred = stacked.mean(axis=0)
        std_pred = stacked.std(axis=0, ddof=1)

        if not previous_mode_training:
            self._model.eval()

        return mean_pred, std_pred

    def _build_sliding_windows(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generates supervised training tensors using a fixed-length window.

        Given sequence length L, each sample uses values[t-L:t] as input and
        values[t] as the one-step-ahead target.
        """
        sequence_length = self.config.sequence_length
        features = values.shape[1]

        sample_count = values.shape[0] - sequence_length
        if sample_count <= 0:
            return (
                np.empty((0, sequence_length, features), dtype=np.float32),
                np.empty((0, features), dtype=np.float32),
            )

        x = np.zeros((sample_count, sequence_length, features), dtype=np.float32)
        y = np.zeros((sample_count, features), dtype=np.float32)

        for idx in range(sample_count):
            x[idx] = values[idx : idx + sequence_length]
            y[idx] = values[idx + sequence_length]

        return x, y

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Saves model and preprocessing state to disk.

        Args:
            checkpoint_path: Target path for serialized checkpoint.

        Raises:
            PredictorError: If model is not fitted.
        """
        if not self.is_fitted:
            raise PredictorError("Cannot save checkpoint before fitting the model.")

        assert self._model is not None
        assert self._feature_means is not None
        assert self._feature_stds is not None
        assert self._columns is not None

        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self._model.state_dict(),
            "feature_means": self._feature_means.astype(np.float32),
            "feature_stds": self._feature_stds.astype(np.float32),
            "columns": list(self._columns),
            "preprocessing_mode": self._preprocessing_mode,
            "config": {
                "sequence_length": self.config.sequence_length,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "weight_decay": self.config.weight_decay,
            },
        }
        torch.save(payload, path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> bool:
        """Loads model and preprocessing state from disk if available.

        Args:
            checkpoint_path: Path to serialized checkpoint.

        Returns:
            True if checkpoint exists and loaded successfully, otherwise False.

        Raises:
            PredictorError: If checkpoint content is invalid.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        required_keys = {"model_state_dict", "feature_means", "feature_stds", "columns", "config"}
        if not required_keys.issubset(set(checkpoint.keys())):
            raise PredictorError("Checkpoint is missing required keys.")

        checkpoint_mode = checkpoint.get("preprocessing_mode")
        if checkpoint_mode != self._preprocessing_mode:
            raise PredictorError(
                "Checkpoint preprocessing mode is incompatible with current predictor. "
                "Please retrain with first-order differencing and save a new checkpoint."
            )

        config_raw = checkpoint["config"]
        try:
            self.config = PredictorConfig(**config_raw)
        except TypeError as exc:
            raise PredictorError("Checkpoint config is incompatible with PredictorConfig.") from exc

        columns = checkpoint["columns"]
        if not isinstance(columns, list) or len(columns) == 0:
            raise PredictorError("Checkpoint columns metadata is invalid.")

        input_size = len(columns)
        self._model = _LSTMForecaster(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            output_size=input_size,
        ).to(self.device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._feature_means = np.asarray(checkpoint["feature_means"], dtype=np.float32)
        self._feature_stds = np.asarray(checkpoint["feature_stds"], dtype=np.float32)
        self._columns = columns

        if self._feature_means.shape[0] != input_size or self._feature_stds.shape[0] != input_size:
            raise PredictorError("Checkpoint scaler dimensions do not match feature schema.")

        return True
