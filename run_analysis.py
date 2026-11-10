"""Fintech deposit campaign analytics: what drives conversion on a phone-sales campaign for a term-deposit
(fixed-deposit) product, how much calling effort is wasted, who to call first, and how to test a better policy.
Data: UCI Bank Marketing (Moro, Cortez & Rita, 2014): 41,188 clients of a Portuguese bank contacted by phone
between May 2008 and Nov 2010, with whether each subscribed to a term deposit. Rows are in date order.
python run_analysis.py   (writes results/)"""
import json, numpy as np, pandas as pd
from math import sqrt
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

R = {}
d = pd.read_csv("data/bank-additional-full.csv.gz", sep=";")

# ---------- 1. data checks ----------
R["validation"] = dict(rows=len(d), exact_duplicates=int(d.duplicated().sum()), missing=int(d.isna().sum().sum()),
                       unknown_values=int((d == "unknown").sum().sum()))
d["y"] = (d.y == "yes").astype(int)
d["order"] = np.arange(len(d))                      # rows are in date order (May 2008 -> Nov 2010)
R["data"] = dict(clients=len(d), conversions=int(d.y.sum()), conversion_rate=float(d.y.mean()),
                 total_calls=int(d.campaign.sum()), calls_per_conversion=float(d.campaign.sum() / d.y.sum()))

# ---------- 2. conversion drivers ----------
bucket = pd.cut(d.campaign, [0, 1, 2, 3, 5, 10, 100], labels=["1", "2", "3", "4-5", "6-10", "11+"])
R["by_contacts"] = {str(k): dict(clients=int(len(g)), conversion=float(g.y.mean())) for k, g in d.groupby(bucket, observed=True)}
R["by_channel"] = {k: dict(clients=int(len(g)), conversion=float(g.y.mean())) for k, g in d.groupby("contact")}
R["by_previous_outcome"] = {k: dict(clients=int(len(g)), conversion=float(g.y.mean())) for k, g in d.groupby("poutcome")}
R["channel_lift"] = R["by_channel"]["cellular"]["conversion"] / R["by_channel"]["telephone"]["conversion"]
R["previous_success_lift"] = R["by_previous_outcome"]["success"]["conversion"] / R["data"]["conversion_rate"]

# ---------- 3. calling effort: what a cap on attempts would save and cost ----------
# each row is a client; campaign = calls made to that client in this campaign, including the last one.
# Under a cap of K calls, calls saved = sum(max(campaign - K, 0)); conversions at risk = converters with campaign > K
# (an upper bound: they converted on a call the cap would have removed).
cap = {}
for K in [3, 5]:
    saved = int((d.campaign - K).clip(lower=0).sum())
    lost = int(d.y[d.campaign > K].sum())
    cap[f"cap_{K}"] = dict(calls_saved=saved, share_of_calls_saved=saved / R["data"]["total_calls"],
                           conversions_at_risk=lost, share_of_conversions_at_risk=lost / R["data"]["conversions"],
                           # worst case: every at-risk conversion is lost; conversions per 100 calls before and after the cap
                           conv_per_100_calls_before=100 * R["data"]["conversions"] / R["data"]["total_calls"],
                           conv_per_100_calls_after=100 * (R["data"]["conversions"] - lost) / (R["data"]["total_calls"] - saved))
    cap[f"cap_{K}"]["efficiency_gain"] = cap[f"cap_{K}"]["conv_per_100_calls_after"] / cap[f"cap_{K}"]["conv_per_100_calls_before"] - 1
R["call_cap"] = cap

# ---------- 4. time trend: fewer calls, higher conversion ----------
d["period"] = pd.qcut(d.order, 5, labels=["P1 (earliest)", "P2", "P3", "P4", "P5 (latest)"])
R["by_period"] = {str(k): dict(clients=int(len(g)), conversion=float(g.y.mean()), avg_calls_per_client=float(g.campaign.mean()),
                               share_cellular=float((g.contact == "cellular").mean()))
                  for k, g in d.groupby("period", observed=True)}

# ---------- 5. who to call first: a pre-call propensity model ----------
# 'duration' (length of the call) is only known after the call ends, so it is excluded: using it would leak the outcome.
# 'campaign' is also excluded, so the model ranks clients before any call is made.
CAT = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
NUM_CLIENT = ["age", "pdays", "previous"]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
X = d[CAT + NUM_CLIENT + MACRO].copy()
for c in CAT: X[c] = X[c].astype("category")
cut = int(len(d) * 0.8)                            # train on the earliest 80% of calls, test on the latest 20%
tr, te = slice(0, cut), slice(cut, len(d))
def fit_eval(cols):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                       categorical_features="from_dtype", random_state=7).fit(X.iloc[tr][cols], d.y.iloc[tr])
    p = m.predict_proba(X.iloc[te][cols])[:, 1]; y = d.y.iloc[te].to_numpy()
    order = np.argsort(-p); out = dict(auc=float(roc_auc_score(y, p)), test_clients=int(len(y)), test_base_rate=float(y.mean()))
    for share in [0.2, 0.3, 0.5]:
        top = order[: int(len(y) * share)]
        out[f"top{int(share*100)}_capture"] = float(y[top].sum() / y.sum())
        out[f"top{int(share*100)}_conversion"] = float(y[top].mean())
    return out, p
R["model_client_only"], _ = fit_eval(CAT + NUM_CLIENT)
R["model_with_macro"], p = fit_eval(CAT + NUM_CLIENT + MACRO)
R["model_note"] = "time split: trained on the earliest 80% of calls, tested on the latest 20%; call duration excluded (only known after the call)"

# ---------- 6. experiment design: testing the targeting policy ----------
# two-proportion test, sample size per arm: n = (z_a + z_b)^2 * [p1(1-p1) + p2(1-p2)] / (p2 - p1)^2
# p1 = control conversion, p2 = treatment conversion, z_a = z for a two-sided 5% significance level, z_b = z for 80% power
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return int(np.ceil((za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2))
p1 = R["data"]["conversion_rate"]
R["experiment"] = dict(
    hypothesis="Calling clients in model-score order with a 3-call cap raises conversion per call vs the current calling order",
    primary_metric="conversions per 100 calls", guardrails=["total conversions", "complaint / do-not-call rate", "calls per client"],
    baseline_conversion=p1,
    n_per_arm={f"+{int(u*1000)/10} pp": n_per_arm(p1, p1 + u) for u in [0.01, 0.015, 0.02]},
    note="sample sizes are clients per arm at 5% significance and 80% power; randomise at the client level with a holdout")

# ---------- outputs ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
k = list(R["by_contacts"]); v = [R["by_contacts"][x]["conversion"] * 100 for x in k]
ax[0].bar(k, v, color="#0052CC"); [ax[0].text(i, v[i] + .3, f"{v[i]:.1f}%", ha="center", fontsize=8) for i in range(len(k))]
ax[0].set_xlabel("Calls made to the client in this campaign"); ax[0].set_ylabel("Conversion (%)"); ax[0].set_title("Conversion falls with every extra call", fontsize=10)
per = list(R["by_period"]); cv = [R["by_period"][x]["conversion"] * 100 for x in per]; cc = [R["by_period"][x]["avg_calls_per_client"] for x in per]
ax[1].bar(range(5), cv, color="#9AA5AD"); ax[1].set_xticks(range(5)); ax[1].set_xticklabels([x.split(" ")[0] for x in per]); ax[1].set_ylabel("Conversion (%)")
a2 = ax[1].twinx(); a2.plot(range(5), cc, color="#E07A2F", marker="o"); a2.set_ylabel("Calls per client", color="#E07A2F"); a2.set_ylim(0, 3.6)
ax[1].set_title("Latest blocks: about 2 calls per client and far higher conversion\n(5 equal blocks of clients in date order, May 2008 to Nov 2010)", fontsize=10)
y = d.y.iloc[te].to_numpy(); o = np.argsort(-p); cum = np.cumsum(y[o]) / y.sum(); xs = np.arange(1, len(y) + 1) / len(y)
ax[2].plot(xs * 100, cum * 100, color="#0052CC", label=f"model (AUC {R['model_with_macro']['auc']:.2f})"); ax[2].plot([0, 100], [0, 100], "--", color="#9AA5AD", label="random order")
t30 = R["model_with_macro"]["top30_capture"] * 100; ax[2].scatter([30], [t30], color="#E07A2F", zorder=3); ax[2].text(32, t30 - 6, f"top 30% of clients\nhold {t30:.0f}% of conversions", fontsize=8)
ax[2].set_xlabel("Share of clients called, in model order (%)"); ax[2].set_ylabel("Share of conversions captured (%)"); ax[2].legend(fontsize=8, loc="lower right")
ax[2].set_title("Who to call first (tested on the latest 20% of calls)", fontsize=10)
for a in ax: a.spines[["top"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/summary.png", dpi=150)
json.dump(R, open("results/results.json", "w"), indent=2)
print(json.dumps(R, indent=2))
