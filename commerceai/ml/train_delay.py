"""1. Predicción del riesgo de retraso de un pedido.

Objetivo: retrasado = fecha real de entrega > fecha estimada.
Salida: backend/app/models/delay_model.joblib
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import MODELS, delivered_orders, load

NUM = ["precio", "flete", "peso_g", "n_items", "dias_estimados", "mes", "dia_semana", "mismo_estado"]
CAT = ["estado_cliente", "estado_vendedor"]


def build_dataset() -> pd.DataFrame:
    orders = delivered_orders()
    items = load("order_items").merge(load("products")[["product_id", "product_weight_g"]], on="product_id")
    items = items.merge(load("sellers")[["seller_id", "seller_state"]], on="seller_id")
    agg = items.groupby("order_id").agg(
        precio=("price", "sum"), flete=("freight_value", "sum"), peso_g=("product_weight_g", "sum"),
        n_items=("order_item_id", "count"), estado_vendedor=("seller_state", "first"))
    df = (orders.merge(load("customers")[["customer_id", "customer_state"]], on="customer_id")
                .merge(agg, on="order_id"))
    df["estado_cliente"] = df["customer_state"]
    df["dias_estimados"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
    df["mes"] = df["order_purchase_timestamp"].dt.month
    df["dia_semana"] = df["order_purchase_timestamp"].dt.dayofweek
    df["mismo_estado"] = (df["estado_cliente"] == df["estado_vendedor"]).astype(int)
    return df[NUM + CAT + ["retrasado"]].dropna()


def main():
    df = build_dataset()
    print(f"Pedidos: {len(df):,} | tasa de retraso: {df['retrasado'].mean():.2%}")
    X, y = df[NUM + CAT], df["retrasado"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                                   remainder="passthrough", sparse_threshold=0)),
        ("clf", HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, random_state=42)),
    ])
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_te, proba):.3f}")
    print(classification_report(y_te, proba >= 0.5, digits=3))
    joblib.dump(model, MODELS / "delay_model.joblib")
    print("Guardado en", MODELS / "delay_model.joblib")


if __name__ == "__main__":
    main()
