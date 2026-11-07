"""Part A: run the seven SQL queries and the pandas cross-check / EDA."""
import re, sqlite3
import numpy as np
import pandas as pd

con = sqlite3.connect("data/instamart.db")
sql = open("sql/queries.sql").read()
queries = [q.strip() for q in re.split(r"(?m)^-- Q\d+:.*$", sql) if q.strip()]
assert len(queries) == 7
res = [pd.read_sql_query(q, con) for q in queries]
con.close()
q1, q2, q3, q4, q5, q6, q7 = res

print("Q1 customers:", len(q1))
print("Q2 rows:", len(q2), "sum:", round(q2.revenue.sum(), 2), "max cell:", q2.iloc[0].to_dict())
print("Q3 products:", len(q3))
print("Q4 customers without completed order:", len(q4))
print("Q5 customers above average:", len(q5), "average:", q5.avg_total.iloc[0])
top = q6[q6.row_num == 1][["city", "customer_id", "total_spend"]]
print("Q6 top per city:\n", top.to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
print("Q7 daily rows:", len(q7), "last cumulative:", q7.cumulative_revenue.iloc[-1])

# pandas cross-check
o = pd.read_csv("data/clean/orders.csv"); c = pd.read_csv("data/clean/customers.csv"); p = pd.read_csv("data/clean/products.csv")
comp = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(p[["product_id", "category"]], on="product_id")
pt = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("\npivot grand total:", round(pt.loc["All", "All"], 2), "== SQL total:", round(pt.loc["All", "All"], 2) == round(q2.revenue.sum(), 2))
print(pt.round(2).to_string())
g = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count")).round(2)
print("\nper city:\n", g.to_string())

# EDA
a = comp.order_amount
q1_, q3_ = a.quantile(.25), a.quantile(.75); iqr = q3_ - q1_
lo, hi = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
print(f"\nmean {a.mean():.2f} median {a.median():.2f} skew {a.skew():.2f} Q1 {q1_:.2f} Q3 {q3_:.2f} IQR {iqr:.2f}")
print(f"fences lower {lo:.2f} upper {hi:.2f}; below {int((a < lo).sum())}; above {int((a > hi).sum())}")
cm = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(cm.round(3).to_string())
tri = cm.where(np.triu(np.ones(cm.shape, dtype=bool), 1)).stack()
print("largest |corr|:", tri.abs().idxmax(), round(tri[tri.abs().idxmax()], 3))
