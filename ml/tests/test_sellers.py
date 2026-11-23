import joblib
import pandas as pd
import pytest

from ml.train_sellers import FEATURES, build_pipeline, label_clusters, seller_metrics


def _synthetic_olist():
    orders = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(12)],
        "order_status": ["delivered"] * 12,
        "order_purchase_timestamp": pd.date_range("2024-01-01", periods=12, freq="D"),
        "order_delivered_customer_date": pd.date_range("2024-01-05", periods=12, freq="D"),
        "order_estimated_delivery_date": pd.date_range("2024-01-07", periods=12, freq="D"),
    })
    # s1 tiene 6 pedidos (elegible), s2 tiene 6 pedidos (elegible), s3 tiene 2 (filtrado)
    sellers = ["s1"] * 5 + ["s2"] * 5 + ["s3"] * 2
    items = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(12)],
        "order_item_id": [1] * 12,
        "seller_id": sellers,
        "price": [100.0] * 5 + [1500.0] * 5 + [50.0] * 2,
    })
    reviews = pd.DataFrame({
        "review_id": [f"r{i}" for i in range(12)],
        "order_id": [f"o{i}" for i in range(12)],
        "review_score": [5] * 5 + [2] * 5 + [4] * 2,
    })
    return orders, items, reviews


def test_seller_metrics_filters_min_orders():
    orders, items, reviews = _synthetic_olist()
    df = seller_metrics(orders, items, reviews)
    assert list(df.index) == ["s1", "s2"]
    assert df.loc["s1", "n_pedidos"] == 5
    assert df.loc["s2", "facturacion"] == pytest.approx(7500.0)


def test_pipeline_fit_predict_and_labels(tmp_path):
    orders, items, reviews = _synthetic_olist()
    df = seller_metrics(orders, items, reviews)
    pipe = build_pipeline(k=2, random_state=42).fit(df[FEATURES])
    df["cluster"] = pipe["km"].labels_
    labels = label_clusters(df.groupby("cluster")[FEATURES].mean())
    assert len(set(labels.values())) == 2

    model_path = tmp_path / "seller_kmeans.joblib"
    joblib.dump({"pipeline": pipe, "etiquetas": labels}, model_path)
    loaded = joblib.load(model_path)
    pred = loaded["pipeline"].predict(df[FEATURES].iloc[:1])
    assert int(pred[0]) in labels