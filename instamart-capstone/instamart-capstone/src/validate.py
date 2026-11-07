"""Assert Part A reference checkpoints. Prints ALL CHECKPOINTS PASSED when all hold."""
import sqlite3
import pandas as pd

raw_o = pd.read_csv("data/raw/orders.csv", keep_default_na=False, na_values=[""])
raw_c = pd.read_csv("data/raw/customers.csv")
c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
con = sqlite3.connect("data/instamart.db")
db_o = pd.read_sql_query("SELECT * FROM orders", con)
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
con.close()
comp = o[o.status == "completed"]
checks = {
    "raw order rows 13084": len(raw_o) == 13084,
    "119 exact duplicates": int(raw_o.duplicated().sum()) == 119,
    "656 pending rows": int((raw_o.status == "pending").sum()) == 656,
    "12309 distinct order_id": raw_o.order_id.nunique() == 12309,
    "120 non-numeric amounts": int(pd.to_numeric(raw_o.order_amount, errors="coerce").isna().sum()) == 120,
    "106 negative delivery": int((raw_o.delivery_minutes < 0).sum()) == 106,
    "2541 blank delivery": int(raw_o.delivery_minutes.isna().sum()) == 2541,
    "61 blank ages": int(raw_c.age.isna().sum()) == 61,
    "24 raw city spellings": raw_c.city.nunique() == 24,
    "6 clean cities": c.city.nunique() == 6,
    "genders 610/560/30": c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "ages 18-60, none missing": c.age.notna().all() and c.age.between(18, 60).all(),
    "12309 clean orders unique": len(o) == 12309 and o.order_id.is_unique,
    "11110 completed / 1199 cancelled": (len(comp), int((o.status == "cancelled").sum())) == (11110, 1199),
    "no pending": not (o.status == "pending").any(),
    "no missing amount": o.order_amount.notna().all(),
    "no missing delivery (completed), 1199 missing (cancelled)": comp.delivery_minutes.notna().all() and int(o[o.status == "cancelled"].delivery_minutes.isna().sum()) == 1199,
    "completed revenue 2998668.80": round(comp.order_amount.sum(), 2) == 2998668.80,
    "db has 3 tables": {"customers", "products", "orders"} <= tables,
    "db orders == csv orders": len(db_o) == len(o),
}
for k, v in checks.items():
    assert v, f"FAILED: {k}"
    print("ok:", k)
print("ALL CHECKPOINTS PASSED")
