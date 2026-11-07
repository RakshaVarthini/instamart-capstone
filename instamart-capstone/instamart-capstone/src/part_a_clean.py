"""Part A: audit raw data, clean it, store to CSV + SQLite."""
import os
import sqlite3
import numpy as np
import pandas as pd

RAW = "data/raw/"
cu = pd.read_csv(RAW + "customers.csv")
pr = pd.read_csv(RAW + "products.csv")
# keep_default_na=False + na_values=[""] so the literal text "N/A" stays a string and is counted as unparseable
od = pd.read_csv(RAW + "orders.csv", keep_default_na=False, na_values=[""])

print("=== AUDIT: shape, dtypes, missing ===")
for name, df in [("customers", cu), ("products", pr), ("orders", od)]:
    print(f"\n[{name}] shape={df.shape}")
    t = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                      "missing_pct": (df.isna().mean() * 100).round(2)})
    print(t.to_string())

amt_num = pd.to_numeric(od["order_amount"], errors="coerce")
audit = {
    "orders raw rows": len(od),
    "exact duplicate rows": int(od.duplicated().sum()),
    "pending rows": int((od["status"] == "pending").sum()),
    "distinct order_id": od["order_id"].nunique(),
    "non-numeric order_amount": int(amt_num.isna().sum() - od["order_amount"].isna().sum()),
    "negative delivery_minutes": int((od["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(od["delivery_minutes"].isna().sum()),
    "customers blank age": int(cu["age"].isna().sum()),
    "ages outside 10-100": int(((cu["age"] < 10) | (cu["age"] > 100)).sum()),
    "distinct raw city spellings": cu["city"].nunique(),
    "distinct non-blank gender spellings": cu["gender"].dropna().nunique(),
    "blank gender": int(cu["gender"].isna().sum()),
}
print("\n=== AUDIT FIGURES ===")
for k, v in audit.items():
    print(f"{k}: {v}")

# ---------------- customers ----------------
c = cu.copy()
c["city"] = c["city"].str.strip().str.title()
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
bad_age = (c["age"] < 10) | (c["age"] > 100)
c.loc[bad_age, "age"] = np.nan
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median")).round().astype(int)
c["signup_date"] = pd.to_datetime(c["signup_date"])
print("\n=== CLEAN CUSTOMERS ===")
print(c["city"].value_counts().to_string())
print(c["gender"].value_counts().to_string())
print("missing age:", int(c["age"].isna().sum()), "age range:", c["age"].min(), c["age"].max())

# ---------------- orders ----------------
o = od.copy()
n0 = len(o)
o["order_ts"] = pd.to_datetime(o["order_ts"])
o = o.drop_duplicates()
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
n2 = len(o)
nbad_amt = int(pd.to_numeric(o["order_amount"], errors="coerce").isna().sum())
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
o = o.merge(pr[["product_id", "category", "price_inr"]], on="product_id", how="left")
fill = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(fill)
nneg = int((o["delivery_minutes"] < 0).sum())
o.loc[o["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
comp = o["status"] == "completed"
med = o["delivery_minutes"].where(comp).groupby(o["category"]).transform("median")
nfill_del = int((comp & o["delivery_minutes"].isna()).sum())
o.loc[comp, "delivery_minutes"] = o.loc[comp, "delivery_minutes"].fillna(med[comp])
o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
print("\n=== CLEAN ORDERS: rule effects ===")
print(f"rows start {n0}; after exact-dup removal {n1} (-{n0-n1}); after latest-record-per-order {n2} (-{n1-n2})")
print(f"unparseable amounts imputed from price: {nbad_amt}")
print(f"negative delivery set missing: {nneg}; completed missing delivery imputed with category median: {nfill_del}")
print("rows", len(o), o["status"].value_counts().to_dict())
print("missing amount", int(o["order_amount"].isna().sum()),
      "missing delivery completed", int(o.loc[o.status == 'completed', 'delivery_minutes'].isna().sum()),
      "missing delivery cancelled", int(o.loc[o.status == 'cancelled', 'delivery_minutes'].isna().sum()))
print("completed revenue", round(o.loc[o.status == 'completed', 'order_amount'].sum(), 2))

# ---------------- store ----------------
os.makedirs("data/clean", exist_ok=True)
c.to_csv("data/clean/customers.csv", index=False)
pr.to_csv("data/clean/products.csv", index=False)
o_out = o.copy()
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
o_out.to_csv("data/clean/orders.csv", index=False)
c_db = c.copy(); c_db["signup_date"] = c_db["signup_date"].dt.strftime("%Y-%m-%d")
con = sqlite3.connect("data/instamart.db")
c_db.to_sql("customers", con, if_exists="replace", index=False)
pr.to_sql("products", con, if_exists="replace", index=False)
o_out.to_sql("orders", con, if_exists="replace", index=False)
con.close()
print("\nstored CSVs and data/instamart.db")
