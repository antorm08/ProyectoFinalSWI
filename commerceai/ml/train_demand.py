"""2. Pronóstico de la demanda diaria de pedidos (SARIMA con estacionalidad semanal).

Salida: backend/app/models/demand_model.joblib
"""
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import MODELS, load


def daily_series() -> pd.Series:
    ts = pd.to_datetime(load("orders")["order_purchase_timestamp"])
    s = ts.dt.floor("D").value_counts().sort_index().asfreq("D", fill_value=0)
    return s.loc["2017-01-01":"2018-08-15"].astype(float)   # recorta extremos incompletos


def fit(s: pd.Series):
    return SARIMAX(s, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False)


def main():
    s = daily_series()
    test_days = 30
    train, test = s.iloc[:-test_days], s.iloc[-test_days:]
    pred = fit(train).get_forecast(test_days).predicted_mean
    mae = np.mean(np.abs(pred.values - test.values))
    mape = np.mean(np.abs((pred.values - test.values) / test.clip(lower=1).values)) * 100
    print(f"Validación últimos {test_days} días -> MAE: {mae:.1f} pedidos/día | MAPE: {mape:.1f}%")

    results = fit(s)  # reentrenar con toda la serie
    joblib.dump({"results": results, "history": s}, MODELS / "demand_model.joblib")
    print("Guardado en", MODELS / "demand_model.joblib")


if __name__ == "__main__":
    main()
