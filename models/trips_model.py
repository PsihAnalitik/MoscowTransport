from models.interface import BaseTSModel, ForecastResult


class TripsMA(BaseTSModel):
    @property
    def name(self): return "MA(8)-Trips"

    def fit(self, data):
        self._is_fitted = True
        return self

    def predict_one_point(self, history):
        return history.iloc[-8:].mean()

# ═══ Актуальная модель ═══════════════════════════════════
Trips = TripsMA

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATA_DIR

    model = TripsMA(historic_points=8, max_forecast_days=90)
    model.load_data(os.path.join(DATA_DIR, 'trips.csv'))
    model.fit(model._full_data)

    # прогноз на завтра
    result = model.predict("2026-02-01")
    print("\n")
    print(result.target_value)

    # прогноз на 3 месяца вперёд — ок
    result = model.predict("2026-05-01")
    print(result.target_value)

    # прогноз на 2 года вперёд — ValueError (> 90 дней)
    #result = model.predict("2028-01-01")

    # прогноз на дату, где мало истории — ValueError
    #result = model.predict("2023-01-03")