import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from final_RTS96_DCOPF import IEEE96DCOPF
import matplotlib.pyplot as plt
import pandas as pd

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        reward = self.locals.get("rewards")
        if reward is not None:
            self.rewards.append(float(reward[0]))
        return True


env = IEEE96DCOPF()
callback = RewardCallback()

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    seed=SEED,
    learning_rate=3e-4,
    buffer_size=500_000,    # replay buffer size
    batch_size=256,         # minibatch size for gradient updates
    gamma=0.00,              # single-step episodes — retains previous flows/dispatch in obs
    tau=0.005,              # soft update coefficient
    ent_coef=0.1,        # fixed entropy tuning
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(net_arch=[64, 64])
)

model.learn(total_timesteps=500_000, callback=callback)
model.save("sac_rts96_dcopf_single_ts_500k")

# --- Multi-Episode Evaluation (10 episodes) ---
N_EVAL = 10
eval_results = []

print("\n" + "=" * 60)
print("POLICY GENERALIZATION EVALUATION (10 episodes)")
print("=" * 60)

for ep in range(N_EVAL):
    obs, _ = env.reset(seed=SEED + ep)
    demand_now = float(np.sum(env.demand))
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    balance_err      = abs(sum(info["agent_dispatch"]) - demand_now)
    flow_viol        = info["flow_violation_mw"]

    eval_results.append({
        "episode":           ep + 1,
        "reward":            reward,
        "balance_err_mw":    balance_err,
        "flow_violation_mw": flow_viol,
    })

    print(f"\nEpisode {ep + 1:>2} | Reward: {reward:>10.4f} | "
          f"Balance err: {balance_err:>7.2f} MW | "
          f"Flow viol: {flow_viol:>7.2f} MW")

# --- Summary Statistics ---
rewards       = [r["reward"]             for r in eval_results]
bal_errors    = [r["balance_err_mw"]    for r in eval_results]
flow_viols    = [r["flow_violation_mw"] for r in eval_results]

print("\n" + "=" * 60)
print("SUMMARY ACROSS 10 EPISODES")
print("=" * 60)
print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 65)
for label, vals in [
    ("Reward",              rewards),
    ("Balance error (MW)",  bal_errors),
    ("Flow violation (MW)", flow_viols),
]:
    print(f"{label:<25} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
          f"{np.min(vals):>10.4f} {np.max(vals):>10.4f}")

# --- Per-Generator Dispatch with Segment Cost Breakdown (last episode) ---
def segment_cost_breakdown(env, dispatch):
    """Returns list of dicts with per-segment MW used and cost for each generator."""
    seg_limits = [env.seg_1, env.seg_2, env.seg_3, env.seg_4]
    seg_costs  = [env.seg_cost_1, env.seg_cost_2, env.seg_cost_3, env.seg_cost_4]
    results = []
    for g in range(env.n_gen):
        remaining = float(dispatch[g])
        seg_mw   = []
        seg_cost = []
        for s_lim, s_cost in zip(seg_limits, seg_costs):
            alloc = min(remaining, float(s_lim[g]))
            alloc = max(alloc, 0.0)
            seg_mw.append(alloc)
            seg_cost.append(alloc * float(s_cost[g]))
            remaining -= alloc
        no_load = float(env.no_load_cost[g])
        total_cost = no_load + sum(seg_cost)
        results.append({
            "seg_mw": seg_mw,
            "seg_cost": seg_cost,
            "no_load_cost": no_load,
            "total_cost": total_cost,
        })
    return results

last_dispatch = info["agent_dispatch"]
breakdown = segment_cost_breakdown(env, last_dispatch)

print("\n--- Per-Generator Dispatch & Segment Costs - Episode 10 ---")
header = (f"{'Gen':>4}  {'Bus':>4}  {'Disp(MW)':>9}  {'Pmin':>7}  {'Pmax':>7}  "
          f"{'NoLoad($)':>10}  "
          f"{'S1 MW':>7}{'S1 $':>8}  "
          f"{'S2 MW':>7}{'S2 $':>8}  "
          f"{'S3 MW':>7}{'S3 $':>8}  "
          f"{'S4 MW':>7}{'S4 $':>8}  "
          f"{'Total($)':>10}")
print(header)
print("-" * len(header))
total_cost_all = 0.0
for g in range(env.n_gen):
    b_g = breakdown[g]
    seg_str = "  ".join(
        f"{b_g['seg_mw'][s]:>7.2f}{b_g['seg_cost'][s]:>8.2f}"
        for s in range(4)
    )
    print(f"{g+1:>4}  {env.gen_bus[g]+1:>4}  {last_dispatch[g]:>9.2f}  "
          f"{env.p_min[g]:>7.2f}  {env.p_max[g]:>7.2f}  "
          f"{b_g['no_load_cost']:>10.2f}  "
          f"{seg_str}  "
          f"{b_g['total_cost']:>10.2f}")
    total_cost_all += b_g["total_cost"]
print("-" * len(header))
print(f"{'TOTAL':>{len(header) - 10}}{total_cost_all:>10.2f}")

# --- Per-Bus Demand (last episode) ---
print("\n--- Per-Bus Demand - Episode 10 (MW) ---")
print(f"{'Bus':>4}  {'Demand':>10}")
print("-" * 18)
for b in range(env.n_bus):
    print(f"{b+1:>4}  {env.demand[b]:>10.2f}")

# --- Plot ---
window = min(4000, len(callback.rewards) // 10)
smoothed = pd.Series(callback.rewards).rolling(window).mean()

plt.figure(figsize=(10, 5))
plt.plot(callback.rewards, color='darkgray', alpha=0.5,
         linewidth=0.5, label='Raw reward')
plt.plot(smoothed, color='blue', linewidth=1.5,
         label=f'Sliding average (window={window})')
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("SAC Training - RTS-96 DCOPF_Single-Step Episode_Lab")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
