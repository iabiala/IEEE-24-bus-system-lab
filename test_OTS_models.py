"""
Test file for trained OTS SAC models.

Loads each saved model (j=0 through j=7 and unconstrained) and evaluates
them across three load scenarios mirroring the paper's analysis:
    - Peak load     (load factor = 0.93)
    - Shoulder load (load factor = 0.8556)
    - Off-peak load (load factor = 0.5487)

Outputs:
    - Per-scenario cost, balance error, flow violation, lines opened
    - Cost savings vs j=0 baseline
    - Which lines were opened at each load level
    - Comparison plots saved to disk
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from final_RTS96_OTS import IEEE96OTS

# ── Configuration ────────────────────────────────────────────────────────────

SEED       = 42
J_VALUES   = [0, 1, 2, 3, 4, 5, 6, 7, None]

# Load scenarios: name → load factor applied to peak demand
SCENARIOS = {
    "Peak"     : 0.93,
    "Shoulder" : 0.8556,
    "Off-peak" : 0.5487,
}

# Model filename template (must match what the SAC training file saved)
def model_fname(j):
    label = f"j={j}" if j is not None else "j=unconstrained"
    return f"sac_rts96_ots_{label.replace('=', '').replace(' ', '_')}_500k"


# ── Helper: evaluate one model across all scenarios ──────────────────────────

def evaluate_model(model, env, j):
    """Run the model on each load scenario and return results."""
    results = {}

    for scenario_name, lf in SCENARIOS.items():
        # Reset environment then fix demand to the scenario load factor
        obs, _ = env.reset(seed=SEED)
        env.demand = env.peak_demand * lf
        obs = env._get_obs()

        # Get deterministic action from the trained policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, info = env.step(action)

        # Balance error: total absolute mismatch across all buses
        balance_err = float(np.sum(np.abs(info["bus_balance_mw"])))

        results[scenario_name] = {
            "reward"          : float(reward),
            "generation_cost" : info["generation_cost"],
            "balance_err_mw"  : balance_err,
            "flow_viol_mw"    : info["flow_violation_mw"],
            "lines_open"      : info["lines_open"],
            "open_indices"    : info["open_line_indices"],
        }

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

all_results = []   # one entry per j value

print("=" * 80)
print("OTS MODEL EVALUATION")
print("=" * 80)

for j in J_VALUES:
    label = f"j={j}" if j is not None else "j=unconstrained"
    fname = model_fname(j) + ".zip"

    # Check model file exists
    if not os.path.exists(fname):
        print(f"\n[SKIP] {label} — model file not found: {fname}")
        continue

    print(f"\n{'─' * 60}")
    print(f"  Loading model: {label}  ({fname})")
    print(f"{'─' * 60}")

    # Load model and create matching environment
    env   = IEEE96OTS(max_open_lines=j)
    model = SAC.load(fname, env=env)

    # Evaluate across all scenarios
    scenario_results = evaluate_model(model, env, j)

    # Print per-scenario results
    for scenario_name, r in scenario_results.items():
        print(f"  {scenario_name:<10} | "
              f"Cost: ${r['generation_cost']:>10.2f} | "
              f"Reward: {r['reward']:>10.4f} | "
              f"Bal err: {r['balance_err_mw']:>7.2f} MW | "
              f"Flow viol: {r['flow_viol_mw']:>7.2f} MW | "
              f"Lines open: {r['lines_open']} {r['open_indices']}")

    all_results.append({
        "j"      : j,
        "label"  : label,
        "results": scenario_results,
    })


# ── Cost savings table ────────────────────────────────────────────────────────

# Use j=0 as baseline
baseline = next((r for r in all_results if r["j"] == 0), None)

if baseline is not None:
    print("\n" + "=" * 80)
    print("COST SAVINGS vs j=0 BASELINE")
    print("=" * 80)

    for scenario_name in SCENARIOS:
        base_cost = baseline["results"][scenario_name]["generation_cost"]
        print(f"\n  Scenario: {scenario_name}  (base cost = ${base_cost:.2f})")
        print(f"  {'j':>14}  {'Cost ($)':>12}  {'Saving (%)':>10}  "
              f"{'Lines Open':>10}  Lines Opened")
        print(f"  {'─'*70}")

        for r in all_results:
            cost     = r["results"][scenario_name]["generation_cost"]
            n_open   = r["results"][scenario_name]["lines_open"]
            indices  = r["results"][scenario_name]["open_indices"]
            saving   = 100 * (base_cost - cost) / base_cost if base_cost > 0 else 0.0
            print(f"  {r['label']:>14}  {cost:>12.2f}  {saving:>9.1f}%  "
                  f"{n_open:>10}  {indices}")


# ── Export results to Excel ──────────────────────────────────────────────────

EXCEL_FILE = "ots_test_results.xlsx"

with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:

    # Sheet 1: Raw results — one row per (j, scenario)
    raw_rows = []
    for r in all_results:
        for scenario_name, s in r["results"].items():
            base_cost = baseline["results"][scenario_name]["generation_cost"] if baseline else None
            saving    = 100 * (base_cost - s["generation_cost"]) / base_cost \
                        if base_cost and base_cost > 0 else None
            raw_rows.append({
                "j"                  : r["label"],
                "Scenario"           : scenario_name,
                "Generation Cost ($)": round(s["generation_cost"], 2),
                "Savings vs j=0 (%)": round(saving, 2) if saving is not None else 0.0,
                "Reward"             : round(s["reward"], 4),
                "Balance Error (MW)" : round(s["balance_err_mw"], 4),
                "Flow Violation (MW)": round(s["flow_viol_mw"], 4),
                "Lines Open"         : s["lines_open"],
                "Open Line Indices"  : str(s["open_indices"]),
            })
    pd.DataFrame(raw_rows).to_excel(writer, sheet_name="Raw Results", index=False)

    # Sheet 2: Cost savings pivot — rows = j, columns = scenarios
    if baseline is not None:
        savings_rows = []
        for r in all_results:
            row = {"j": r["label"]}
            for scenario_name in SCENARIOS:
                base_cost = baseline["results"][scenario_name]["generation_cost"]
                cost      = r["results"][scenario_name]["generation_cost"]
                row[f"{scenario_name} Cost ($)"]    = round(cost, 2)
                row[f"{scenario_name} Saving (%)"]  = round(
                    100 * (base_cost - cost) / base_cost if base_cost > 0 else 0.0, 2)
            savings_rows.append(row)
        pd.DataFrame(savings_rows).to_excel(writer, sheet_name="Cost Savings", index=False)

    # Sheet 3: Lines opened — rows = j, columns = scenarios
    lines_rows = []
    for r in all_results:
        row = {"j": r["label"]}
        for scenario_name in SCENARIOS:
            row[f"{scenario_name} Lines Open"]    = r["results"][scenario_name]["lines_open"]
            row[f"{scenario_name} Line Indices"]  = str(r["results"][scenario_name]["open_indices"])
        lines_rows.append(row)
    pd.DataFrame(lines_rows).to_excel(writer, sheet_name="Lines Opened", index=False)

print(f"\nResults saved → {EXCEL_FILE}")
print(f"  Sheets: 'Raw Results', 'Cost Savings', 'Lines Opened'")


# ── Plot 1: Cost savings vs j for each scenario ──────────────────────────────

if baseline is not None:
    fig, ax = plt.subplots(figsize=(11, 5))

    for scenario_name in SCENARIOS:
        base_cost = baseline["results"][scenario_name]["generation_cost"]
        labels    = [r["label"] for r in all_results]
        savings   = [
            100 * (base_cost - r["results"][scenario_name]["generation_cost"]) / base_cost
            for r in all_results
        ]
        ax.plot(labels, savings, marker="o", linewidth=1.5, label=scenario_name)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Max open lines (j)")
    ax.set_ylabel("Cost savings vs j=0 (%)")
    ax.set_title("OTS Cost Savings by Load Scenario")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("test_cost_savings_by_scenario.png", dpi=150)
    print("\nPlot saved → test_cost_savings_by_scenario.png")
    plt.show()


# ── Plot 2: Generation cost vs j for each scenario ───────────────────────────

fig, ax = plt.subplots(figsize=(11, 5))

for scenario_name in SCENARIOS:
    labels = [r["label"] for r in all_results]
    costs  = [r["results"][scenario_name]["generation_cost"] for r in all_results]
    ax.plot(labels, costs, marker="o", linewidth=1.5, label=scenario_name)

ax.set_xlabel("Max open lines (j)")
ax.set_ylabel("Generation cost ($)")
ax.set_title("Generation Cost vs Line-Opening Budget (j)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("test_generation_cost_vs_j.png", dpi=150)
print("Plot saved → test_generation_cost_vs_j.png")
plt.show()


# ── Plot 3: Average lines opened vs j ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
labels     = [r["label"] for r in all_results]
avg_opens  = [
    np.mean([r["results"][s]["lines_open"] for s in SCENARIOS])
    for r in all_results
]
ax.bar(labels, avg_opens, color="steelblue")
ax.set_xlabel("Max open lines (j)")
ax.set_ylabel("Average lines opened")
ax.set_title("Average Number of Lines Opened per j")
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("test_lines_opened_vs_j.png", dpi=150)
print("Plot saved → test_lines_opened_vs_j.png")
plt.show()

print("\nDone.")
