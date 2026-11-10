"""Loads the data into SQLite, runs sql/campaign_kpis.sql and checks every result against results/results.json.
Run after run_analysis.py:  python run_sql.py"""
import json, sqlite3, pandas as pd
d = pd.read_csv("data/bank-additional-full.csv.gz", sep=";"); d["y"] = (d.y == "yes").astype(int)
con = sqlite3.connect(":memory:"); d[["campaign", "contact", "poutcome", "y"]].to_sql("calls", con, index=False)
R = json.load(open("results/results.json"))
qs = [q.strip() for q in open("sql/campaign_kpis.sql").read().split(";") if "SELECT" in q]
out = [pd.read_sql(q, con) for q in qs]
for i, o in enumerate(out, 1): print(f"-- query {i}\n{o.to_string(index=False)}\n")
h, b, seg, cap = out
assert h.clients[0] == R["data"]["clients"] and h.conversions[0] == R["data"]["conversions"] and h.total_calls[0] == R["data"]["total_calls"]
for r in b.itertuples(): assert r.clients == R["by_contacts"][r.bucket]["clients"] and abs(r.conversion - R["by_contacts"][r.bucket]["conversion"]) < 1e-4
for r in seg.itertuples():
    src = R["by_channel"] if r.cut == "channel" else R["by_previous_outcome"]
    assert r.clients == src[r.segment]["clients"] and abs(r.conversion - src[r.segment]["conversion"]) < 1e-4
for r in cap.itertuples():
    c = R["call_cap"][f"cap_{r.cap}"]; assert r.calls_saved == c["calls_saved"] and r.conversions_at_risk == c["conversions_at_risk"]
print("All SQL results match the Python analysis.")
