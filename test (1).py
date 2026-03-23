import numpy as np
import torch
from stable_baselines3 import SAC
from final_RTS96_DCOPF import IEEE96DCOPF
import pandas as pd

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

env   = IEEE96DCOPF()
model = SAC.load("sac_rts96_dcopf_single_ts_1M", env=env)

N_EVAL = 50


def segment_cost_breakdown(env, dispatch):
    """Per-generator segment MW used and cost."""
    seg_limits = [env.seg_1, env.seg_2, env.seg_3, env.seg_4]
    seg_costs  = [env.seg_cost_1, env.seg_cost_2, env.seg_cost_3, env.seg_cost_4]
    results = []
    for g in range(env.n_gen):
        remaining = float(dispatch[g])
        seg_mw, seg_cost = [], []
        for s_lim, s_cost in zip(seg_limits, seg_costs):
            alloc = max(min(remaining, float(s_lim[g])), 0.0)
            seg_mw.append(alloc)
            seg_cost.append(alloc * float(s_cost[g]))
            remaining -= alloc
        no_load    = float(env.no_load_cost[g])
        total_cost = no_load + sum(seg_cost)
        results.append({
            "seg_mw":      seg_mw,
            "seg_cost":    seg_cost,
            "no_load_cost": no_load,
            "total_cost":  total_cost,
        })
    return results


# ── Evaluation loop ────────────────────────────────────────────────────────────
summary_rows  = []
dispatch_rows = []

print("\n" + "=" * 60)
print("POLICY GENERALIZATION EVALUATION (50 episodes)")
print("=" * 60)

for ep in range(N_EVAL):
    obs, _      = env.reset(seed=SEED + ep)
    demand_now  = float(np.sum(env.demand))
    action, _   = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    balance_err = abs(sum(info["agent_dispatch"]) - demand_now)
    flow_viol   = info["flow_violation_mw"]
    dispatch    = info["agent_dispatch"]
    breakdown       = segment_cost_breakdown(env, dispatch)
    total_cost      = sum(b["total_cost"] for b in breakdown)

    print(f"\nEpisode {ep+1:>2} | Reward: {reward:>10.4f} | "
          f"Balance err: {balance_err:>7.2f} MW | "
          f"Flow viol: {flow_viol:>7.2f} MW | "
          f"Cost: ${total_cost:,.2f}")
    print(f"           Demand: {demand_now:>8.2f} MW | "
          f"Total Dispatch: {float(np.sum(dispatch)):>8.2f} MW")

    # ── Summary row ────────────────────────────────────────────────────────────
    summary_rows.append({
        "Episode":              ep + 1,
        "Reward":               round(reward, 4),
        "Total Demand (MW)":    round(demand_now, 2),
        "Total Dispatch (MW)":  round(float(np.sum(dispatch)), 2),
        "Balance Error (MW)":   round(balance_err, 4),
        "Flow Violation (MW)":  round(flow_viol, 4),
        "Total Cost ($)":       round(total_cost, 2),
    })

    # ── Per-generator dispatch rows ────────────────────────────────────────────
    for g in range(env.n_gen):
        b = breakdown[g]
        dispatch_rows.append({
            "Episode":      ep + 1,
            "Gen":          g + 1,
            "Bus":          env.gen_bus[g] + 1,
            "Dispatch (MW)": round(float(dispatch[g]), 2),
            "Pmin (MW)":    round(float(env.p_min[g]), 2),
            "Pmax (MW)":    round(float(env.p_max[g]), 2),
            "NoLoad ($)":   round(b["no_load_cost"], 2),
            "S1 MW":        round(b["seg_mw"][0], 2),
            "S1 Cost ($)":  round(b["seg_cost"][0], 2),
            "S2 MW":        round(b["seg_mw"][1], 2),
            "S2 Cost ($)":  round(b["seg_cost"][1], 2),
            "S3 MW":        round(b["seg_mw"][2], 2),
            "S3 Cost ($)":  round(b["seg_cost"][2], 2),
            "S4 MW":        round(b["seg_mw"][3], 2),
            "S4 Cost ($)":  round(b["seg_cost"][3], 2),
            "Total Cost ($)": round(b["total_cost"], 2),
        })

# ── Summary statistics ─────────────────────────────────────────────────────────
df_summary  = pd.DataFrame(summary_rows)
df_dispatch = pd.DataFrame(dispatch_rows)

rewards    = df_summary["Reward"].tolist()
bal_errs   = df_summary["Balance Error (MW)"].tolist()
flow_viols = df_summary["Flow Violation (MW)"].tolist()
costs      = df_summary["Total Cost ($)"].tolist()

print("\n" + "=" * 60)
print("SUMMARY ACROSS 50 EPISODES")
print("=" * 60)
print(f"{'Metric':<25} {'Mean':>12} {'Std':>10} {'Min':>12} {'Max':>12}")
print("-" * 73)
for label, vals in [
    ("Reward",              rewards),
    ("Balance error (MW)",  bal_errs),
    ("Flow violation (MW)", flow_viols),
    ("Total Cost ($)",      costs),
]:
    print(f"{label:<25} {np.mean(vals):>12.4f} {np.std(vals):>10.4f} "
          f"{np.min(vals):>12.4f} {np.max(vals):>12.4f}")

# ── Export to Excel ────────────────────────────────────────────────────────────
output_file = "RTS96_SAC_Evaluation_9.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    # Sheet 1 — Episode summary
    df_summary.to_excel(writer, sheet_name="Episode Summary", index=False)

    # Sheet 2 — Per-generator dispatch (all 10 episodes)
    df_dispatch.to_excel(writer, sheet_name="Generator Dispatch", index=False)

    # Sheet 3 — Episode 10 dispatch only (matches printed table)
    df_ep10 = df_dispatch[df_dispatch["Episode"] == N_EVAL].copy()
    df_ep10.to_excel(writer, sheet_name="Episode 10 Dispatch", index=False)

    # Sheet 4 — Stats summary
    stats_rows = []
    for label, vals in [
        ("Reward",              rewards),
        ("Balance Error (MW)",  bal_errs),
        ("Flow Violation (MW)", flow_viols),
        ("Total Cost ($)",      costs),
    ]:
        stats_rows.append({
            "Metric": label,
            "Mean":   round(float(np.mean(vals)), 4),
            "Std":    round(float(np.std(vals)),  4),
            "Min":    round(float(np.min(vals)),  4),
            "Max":    round(float(np.max(vals)),  4),
        })
    pd.DataFrame(stats_rows).to_excel(writer, sheet_name="Statistics", index=False)

print(f"\nResults saved to {output_file}")
print("  Sheet 1 — Episode Summary")
print("  Sheet 2 — Generator Dispatch (all 10 episodes)")
print("  Sheet 3 — Episode 10 Dispatch")
print("  Sheet 4 — Statistics")
