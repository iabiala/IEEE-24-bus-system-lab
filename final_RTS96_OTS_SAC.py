import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from final_RTS96_OTS import IEEE96OTS
import matplotlib.pyplot as plt
import pandas as pd

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# j values to sweep: 0 = no switching (baseline), 1..7 mirrors paper Table II, None = unconstrained
J_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, None]


class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        reward = self.locals.get("rewards")
        if reward is not None:
            self.rewards.append(float(reward[0]))
        return True


def run_trial(j):
    label = f"j={j}" if j is not None else "j=unconstrained"
    print("\n" + "=" * 60)
    print(f"TRAINING  {label}")
    print("=" * 60)

    env = IEEE96OTS(max_open_lines=j)
    callback = RewardCallback()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        seed=SEED,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=256,
        gamma=0.00,
        tau=0.005,
        ent_coef=0.1,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[64, 64])
    )

    model.learn(total_timesteps=500_000, callback=callback)
    fname = f"sac_rts96_ots_{label.replace('=', '').replace(' ', '_')}_500k"
    model.save(fname)
    print(f"Model saved → {fname}")

    # --- Evaluation (10 episodes) ---
    # Episodes alternate between random load (generalisation) and
    # fixed peak load (comparable to the paper's fixed-load analysis)
    N_EVAL = 10
    eval_results = []

    print(f"\nEVALUATION  {label}")
    print("-" * 60)

    for ep in range(N_EVAL):
        obs, _ = env.reset(seed=SEED + ep)
        if ep == 0:
            # Fix to peak load for the first episode
            env.demand = env.peak_demand.copy()
            obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, _, info = env.step(action)

        balance_err = float(np.sum(np.abs(info["bus_balance_mw"])))
        eval_results.append({
            "episode":           ep + 1,
            "reward":            reward,
            "balance_err_mw":    balance_err,
            "flow_violation_mw": info["flow_violation_mw"],
            "lines_open":        info["lines_open"],
            "generation_cost":   info["generation_cost"],
            "open_line_indices": info["open_line_indices"],
        })

        print(f"  Ep {ep+1:>2} | Reward: {reward:>10.4f} | "
              f"Cost: ${info['generation_cost']:>10.2f} | "
              f"Balance err: {balance_err:>7.2f} MW | "
              f"Flow viol: {info['flow_violation_mw']:>7.2f} MW | "
              f"Lines open: {info['lines_open']} {info['open_line_indices']}")

    rewards    = [r["reward"]            for r in eval_results]
    bal_errors = [r["balance_err_mw"]   for r in eval_results]
    flow_viols = [r["flow_violation_mw"] for r in eval_results]
    lines_open = [r["lines_open"]        for r in eval_results]
    gen_costs  = [r["generation_cost"]   for r in eval_results]

    # best episode = lowest generation cost
    best_ep = eval_results[int(np.argmin(gen_costs))]

    peak_ep = eval_results[0]   # ep=0 was fixed to peak load
    print(f"\n  Mean reward      : {np.mean(rewards):>10.4f}")
    print(f"  Mean cost        : ${np.mean(gen_costs):>10.2f}")
    print(f"  Mean balance err : {np.mean(bal_errors):>10.4f} MW")
    print(f"  Mean flow viol   : {np.mean(flow_viols):>10.4f} MW")
    print(f"  Mean lines open  : {np.mean(lines_open):>10.2f}")
    print(f"  Peak load cost   : ${peak_ep['generation_cost']:>10.2f}  lines: {peak_ep['open_line_indices']}")
    print(f"  Best ep cost     : ${best_ep['generation_cost']:>10.2f}  lines: {best_ep['open_line_indices']}")

    return {
        "j":                j,
        "label":            label,
        "training_rewards": callback.rewards,
        "mean_reward":      np.mean(rewards),
        "mean_cost":        np.mean(gen_costs),
        "mean_balance_err": np.mean(bal_errors),
        "mean_flow_viol":   np.mean(flow_viols),
        "mean_lines_open":  np.mean(lines_open),
        "best_open_lines":  best_ep["open_line_indices"],
        "peak_cost":        peak_ep["generation_cost"],
        "peak_open_lines":  peak_ep["open_line_indices"],
    }


# --- Run all trials ---
all_results = [run_trial(j) for j in J_VALUES]

# --- Comparison table ---
# j=0 (first result) is the no-switching baseline
base_cost = all_results[0]["mean_cost"]

print("\n" + "=" * 80)
print("COMPARISON ACROSS ALL j VALUES  (baseline = j=0, no switching)")
print("=" * 80)
print(f"{'j':>14}  {'Peak Cost ($)':>13}  {'Savings vs j=0':>14}  "
      f"{'Lines Open':>10}  Lines Opened at Peak Load")
print("-" * 80)
base_peak_cost = all_results[0]["peak_cost"]
for r in all_results:
    savings_pct = 100 * (base_peak_cost - r["peak_cost"]) / base_peak_cost if base_peak_cost > 0 else 0.0
    print(f"{r['label']:>14}  {r['peak_cost']:>13.2f}  "
          f"{savings_pct:>13.1f}%  "
          f"{len(r['peak_open_lines']):>10}  {r['peak_open_lines']}")

# --- Training curve plot (all j on one figure) ---
fig, ax = plt.subplots(figsize=(12, 6))
for r in all_results:
    raw = pd.Series(r["training_rewards"])
    window = min(4000, len(raw) // 10)
    smoothed = raw.rolling(window).mean()
    ax.plot(smoothed, linewidth=1.2, label=r["label"])

ax.set_xlabel("Timestep")
ax.set_ylabel("Reward (smoothed)")
ax.set_title("SAC Training — RTS-96 OTS, all j values")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("ots_training_curves.png", dpi=150)
plt.show()

# --- Cost savings vs j bar chart ---
labels        = [r["label"] for r in all_results]
savings_pcts  = [
    100 * (base_cost - r["mean_cost"]) / base_cost if base_cost > 0 else 0.0
    for r in all_results
]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(labels, savings_pcts, color="steelblue")
ax.set_xlabel("Max open lines (j)")
ax.set_ylabel("Cost savings vs j=0 (%)")
ax.set_title("OTS Cost Savings vs. Line-Opening Budget (j)")
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("ots_cost_savings_vs_j.png", dpi=150)
plt.show()
