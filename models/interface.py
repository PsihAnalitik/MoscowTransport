import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    values: np.ndarray
    model_name: str
    target_date: pd.Timestamp
    steps: int

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'date': self.dates,
            'forecast': self.values,
            'model': self.model_name,
        })

    @property
    def target_value(self) -> float:
        """Прогноз именно на запрошенную дату."""
        return self.values[-1]


class BaseTSModel(ABC):
    """
    Абстрактный базовый класс для моделей прогнозирования
    временных рядов (trips, sales, price и т.д.).

    Parameters
    ----------
    historic_points : int
        Количество последних точек ряда, необходимых модели для прогноза.
    max_forecast_days : int
        Максимальный горизонт прогноза (дней вперёд от последних данных).
    min_history_points : int | None
        Минимум точек истории, без которых прогноз невозможен.
        По умолчанию = historic_points.
    """

    def __init__(
        self,
        historic_points: int,
        max_forecast_days: int = 365,
        min_history_points: Optional[int] = None,
        date_col: str = 'date',
        value_col: str = 'value',
    ):
        self.historic_points = historic_points
        self.max_forecast_days = max_forecast_days
        self.min_history_points = min_history_points or historic_points
        self.date_col = date_col
        self.value_col = value_col

        self._full_data: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False

    # ── абстрактные методы ──────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Человекочитаемое название модели."""
        ...

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseTSModel':
        """Обучить модель на исторических данных. Возвращает self."""
        ...

    @abstractmethod
    def predict_one_point(self, history: pd.Series) -> float:
        """Предсказать одну следующую точку по хвосту ряда."""
        ...

    # ── загрузка данных ─────────────────────────────────

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загружает ПОЛНЫЙ датасет (обрезка — на этапе predict)."""
        try:
            df = pd.read_csv(filepath, parse_dates=[self.date_col])
            df = df.sort_values(self.date_col).reset_index(drop=True)
            self._full_data = df
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        except KeyError as e:
            raise KeyError(f"Колонка {e} не найдена в {filepath}")

    # ── валидация запроса ───────────────────────────────

    def _validate_request(
        self, datetime_point: pd.Timestamp, available: pd.DataFrame
    ) -> int:
        """
        Проверяет, можно ли дать прогноз на datetime_point.

        Returns
        -------
        steps : int
            Количество шагов (дней) для рекурсивного прогноза.

        Raises
        ------
        RuntimeError  — модель не обучена
        ValueError    — мало данных / дата за пределами лимитов
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"[{self.name}] Модель не обучена. Вызовите fit() сначала."
            )

        # --- хватает ли истории? ---
        if len(available) < self.min_history_points:
            raise ValueError(
                f"Недостаточно данных для прогноза на {datetime_point:%Y-%m-%d}: "
                f"нужно минимум {self.min_history_points} точек, "
                f"доступно {len(available)}. "
                f"Дата слишком далеко в прошлом?"
            )

        last_known = available[self.date_col].max()
        steps = (datetime_point - last_known).days

        # --- дата в прошлом (уже есть факт)? ---
        if steps <= 0:
            raise ValueError(
                f"Дата {datetime_point:%Y-%m-%d} уже присутствует в данных "
                f"(последняя доступная точка: {last_known:%Y-%m-%d}). "
                f"Прогноз не требуется."
            )

        # --- не слишком ли далеко? ---
        if steps > self.max_forecast_days:
            raise ValueError(
                f"Дата {datetime_point:%Y-%m-%d} слишком далеко: "
                f"{steps} дней от последних данных, "
                f"лимит модели — {self.max_forecast_days} дней."
            )

        return steps

    # ── основной прогноз ────────────────────────────────

    def predict(
        self,
        datetime_point,
        data: Optional[pd.DataFrame] = None,
    ) -> ForecastResult:
        """
        Прогноз значения на конкретную дату.

        Parameters
        ----------
        datetime_point : str | datetime | pd.Timestamp
            Целевая дата прогноза.
        data : pd.DataFrame | None
            Если передан — используется вместо self._full_data.

        Returns
        -------
        ForecastResult
            Полная траектория от последней известной точки до datetime_point.
        """
        datetime_point = pd.Timestamp(datetime_point)
        df = data if data is not None else self._full_data

        if df is None:
            raise ValueError(
                "Нет данных для прогноза. "
                "Передайте data или вызовите load_data()."
            )

        # 1. Фильтруем: только точки ДО запрошенной даты
        available = df[df[self.date_col] < datetime_point]

        # 2. Валидация
        steps = self._validate_request(datetime_point, available)

        # 3. Обрезаем до нужного хвоста
        history = available.iloc[-self.historic_points:]
        values = history[self.value_col].values.tolist()
        last_known = history[self.date_col].max()

        # 4. Рекурсивный прогноз
        forecast_dates = pd.date_range(
            start=last_known + pd.Timedelta(days=1),
            periods=steps,
            freq='D',
        )

        forecasts = []
        for _ in range(steps):
            pred = self.predict_one_point(
                pd.Series(values[-self.historic_points:])
            )
            forecasts.append(pred)
            values.append(pred)

        return ForecastResult(
            dates=forecast_dates,
            values=np.array(forecasts),
            model_name=self.name,
            target_date=datetime_point,
            steps=steps,
        )

    # ── метрики ─────────────────────────────────────────

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """MAE, RMSE, sMAPE между фактом и прогнозом."""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
        return {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'smape': round(smape * 100, 1),
        }

    def __repr__(self) -> str:
        status = 'fitted' if self._is_fitted else 'not fitted'
        return (
            f"<{self.name} | "
            f"history={self.historic_points} | "
            f"max_horizon={self.max_forecast_days}d | "
            f"{status}>"
        )
