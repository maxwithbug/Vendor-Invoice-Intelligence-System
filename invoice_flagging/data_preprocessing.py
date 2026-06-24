import sqlite3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DB_PATH = Path(__file__).parent.parent / "data" / "inventory.db"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_invoice_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand)                               AS total_brands,
            SUM(p.Quantity)                                       AS total_item_quantity,
            SUM(p.Dollars)                                        AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity                                               AS invoice_quantity,
        vi.Dollars                                                AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate))        AS days_po_to_invoice,
        (julianday(vi.PayDate)     - julianday(vi.InvoiceDate))   AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa ON vi.PONumber = pa.PONumber
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def create_invoice_risk_label(row):
    if (
        pd.notna(row["total_item_dollars"])
        and abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5
    ):
        return 1
    if pd.notna(row["avg_receiving_delay"]) and row["avg_receiving_delay"] > 10:
        return 1
    return 0


def apply_labels(df):
    df = df.copy()
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


def clean_features(df, features):
    df = df.dropna(subset=["invoice_quantity", "invoice_dollars", "Freight"])
    join_cols = [
        "total_brands",
        "total_item_quantity",
        "total_item_dollars",
        "avg_receiving_delay",
    ]
    for col in join_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)
    return df


def split_data(df, features, target):
    df = clean_features(df, features)
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    return X_train_scaled, X_test_scaled
