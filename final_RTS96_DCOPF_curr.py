import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class IEEE96DCOPF_Curriculum(gym.Env):
    """
    DC Optimal Power Flow environment

    Single-step episode (one dispatch decision per episode)

    Network: IEEE RTS-96 (24-bus, 38-line)

    Action space: Box(n_gen)
        action[0:n_gen] — generator dispatch in MW (continuous)

    Observation space: Box(1 + n_line + n_gen)
        [load_scale (1), normalised line flows (n_line), normalised generator dispatch (n_gen)]
    """

    def __init__(self, curriculum_episodes=250000):
        super().__init__()
        self.curriculum_episodes = curriculum_episodes

        # Generator data
        gen_df = pd.read_excel("RTS96_SystemData_WithWind.xlsx", sheet_name="Generator")
        self.gen_bus = gen_df["Bus Number"].to_numpy() - 1
        self.p_max = gen_df["Capacity (MW)"].to_numpy()
        self.p_min = gen_df["Minimum generation (MW)"].to_numpy()
        self.n_gen = len(self.gen_bus)

        self.seg_1 = gen_df["Seg 1"].to_numpy()
        self.seg_2 = gen_df["Seg 2"].to_numpy()
        self.seg_3 = gen_df["Seg 3"].to_numpy()
        self.seg_4 = gen_df["Seg 4"].to_numpy()

        self.seg_cost_1 = gen_df["Cost 1"].to_numpy()
        self.seg_cost_2 = gen_df["Cost 2"].to_numpy()
        self.seg_cost_3 = gen_df["Cost 3"].to_numpy()
        self.seg_cost_4 = gen_df["Cost 4"].to_numpy()

        self.no_load_cost = gen_df["NoLoadCost"].to_numpy()

        self.merit_order = list(np.argsort(
            self.seg_cost_1 + self.seg_cost_2 + self.seg_cost_3 + self.seg_cost_4
        ))
        self.max_cost = sum(
            self.no_load_cost[g]
            + self.seg_cost_1[g] * self.seg_1[g]
            + self.seg_cost_2[g] * self.seg_2[g]
            + self.seg_cost_3[g] * self.seg_3[g]
            + self.seg_cost_4[g] * self.seg_4[g]
            for g in range(self.n_gen)
        )

        # Line data
        line_df = pd.read_excel("RTS96_SystemData_WithWind.xlsx", sheet_name="Line")
        self.from_bus = line_df["From Bus"].to_numpy() - 1
        self.to_bus = line_df["To Bus"].to_numpy() - 1
        self.b = line_df["susceptance (pu)"].to_numpy()
        self.f_max = line_df["Capacity (MW)"].to_numpy()
        self.n_line = len(self.from_bus)
        self.b_base = self.b

        # Bus data
        bus_df = pd.read_excel("RTS96_SystemData_WithWind.xlsx", sheet_name="Bus")
        self.peak_demand = bus_df["Load (MW)"].to_numpy()
        self.n_bus = len(self.peak_demand)

        self.demand = self.peak_demand.copy()

        # Load factor
        load_factor_df = pd.read_excel("RTS96_SystemData_WithWind.xlsx", sheet_name="load factor", index_col=0)
        self.load_factor = load_factor_df.loc["Load factor"].to_numpy()
        self._lf_full_min = float(self.load_factor.min())
        self._lf_full_max = float(self.load_factor.max())
        self._curriculum_lf = self._lf_full_min
        self._reset_count = 0

        # Action space: generator dispatch only
        gen_high = np.maximum(self.p_max, self.p_min + 1.0)  # prevent degenerate SAC dims
        self.action_space = spaces.Box(
            low=self.p_min,
            high=gen_high,
            dtype=np.float32
        )

        # Observation space: load_scale + normalised line flows + normalised gen dispatch
        obs_size = 1 + self.n_line + self.n_gen
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.current_flows = np.zeros(self.n_line)
        self.current_bus_balance = np.zeros(self.n_bus)
        self.current_dispatch = np.zeros(self.n_gen)

        # All lines are always energised — precompute fixed susceptance matrix
        self.B_full = np.zeros((self.n_bus, self.n_bus))
        for k in range(self.n_line):
            i, j = self.from_bus[k], self.to_bus[k]
            self.B_full[i, i] += self.b_base[k]
            self.B_full[j, j] += self.b_base[k]
            self.B_full[i, j] -= self.b_base[k]
            self.B_full[j, i] -= self.b_base[k]

        # Pre-factorise reduced B matrix (rows/cols 1..n_bus-1, slack bus removed)
        self.B_reduced = self.B_full[1:, 1:]

        self._needs_reset = True
        self.reset()

    def compute_line_flows(self, theta):
        # All lines are on; vectorised flow computation
        return self.b_base * (theta[self.from_bus] - theta[self.to_bus])

    def compute_bus_balance(self, P, flows):
        balance = -self.demand.copy()
        np.add.at(balance, self.gen_bus, P)
        np.add.at(balance, self.from_bus, -flows)
        np.add.at(balance, self.to_bus,    flows)
        return balance

    def _piecewise_cost(self, P):
        seg_limits = [self.seg_1, self.seg_2, self.seg_3, self.seg_4]
        seg_costs  = [self.seg_cost_1, self.seg_cost_2, self.seg_cost_3, self.seg_cost_4]

        total = 0.0
        for g in range(self.n_gen):
            cost_g    = self.no_load_cost[g]
            remaining = float(P[g])
            for s_lim, s_cost in zip(seg_limits, seg_costs):
                if remaining <= 0:
                    break
                alloc      = min(remaining, float(s_lim[g]))
                cost_g    += s_cost[g] * alloc
                remaining -= alloc
            total += cost_g
        return total

    def _get_obs(self):
        total_demand = float(np.sum(self.demand))
        peak_total   = float(np.sum(self.peak_demand))
        load_scale   = total_demand / peak_total if peak_total > 0 else 0.0

        flow_ratio = np.zeros(self.n_line)
        for k in range(self.n_line):
            if self.f_max[k] > 0:
                flow_ratio[k] = self.current_flows[k] / self.f_max[k]

        gen_ratio = np.zeros(self.n_gen)
        for g in range(self.n_gen):
            if self.p_max[g] > 0:
                gen_ratio[g] = self.current_dispatch[g] / self.p_max[g]

        obs = np.concatenate([
            [load_scale],
            flow_ratio,
            gen_ratio,
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # Reset function
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._reset_count < self.curriculum_episodes:
            # Phase 1: cyclic curriculum over [lf_min, lf_max]
            lf = self._curriculum_lf
            self._curriculum_lf += 0.001
            if self._curriculum_lf > self._lf_full_max:
                self._curriculum_lf = self._lf_full_min
        else:
            # Phase 2: random uniform sampling for generalization
            lf = float(self.np_random.uniform(self._lf_full_min, self._lf_full_max))
        self._reset_count += 1
        self.demand = self.peak_demand * lf

        # current_flows and current_dispatch are retained from the previous episode
        # so the agent sees the last dispatch result in the next episode's observation
        self.current_bus_balance    = np.zeros(self.n_bus)

        self._needs_reset = False
        return self._get_obs(), {}

    # Step function
    def step(self, action):
        # total_demand = float(np.sum(self.demand))

        # --- Parse and clip action ---
        P       = np.array(action[:self.n_gen])
        seg_max = self.seg_1 + self.seg_2 + self.seg_3 + self.seg_4
        P       = np.clip(P, self.p_min, np.minimum(self.p_max, seg_max))
        self.current_dispatch = P

        # --- Net injections ---
        injections = -self.demand.copy()
        for g in range(self.n_gen):
            injections[self.gen_bus[g]] += P[g]

        # --- DC power flow (bus 0 is slack) ---
        try:
            theta = np.concatenate([[0.0], np.linalg.solve(self.B_reduced, injections[1:])])
        except np.linalg.LinAlgError:
            self._needs_reset = True
            return self._get_obs(), -20000, True, False, {"error": "singular_B_matrix"}

        flows = self.compute_line_flows(theta)
        self.current_flows = flows

        bus_balance = self.compute_bus_balance(P, flows)
        self.current_bus_balance = bus_balance

        # --- Reward components ---
        # 1. Generation cost, normalised to [0, 1]
        cost_term = self._piecewise_cost(P) / self.max_cost

        # 2. Per-bus power balance penalty
        bus_balance_penalty = np.sum(np.abs(bus_balance)) / float(np.sum(self.peak_demand))

        # 3. Thermal limit penalty
        flow_penalty = sum(
            max(0.0, abs(flows[k]) - self.f_max[k]) / self.f_max[k]
            for k in range(self.n_line)
        )

        reward = -(
            1000 * cost_term
            + 15000 * bus_balance_penalty
            + 2000  * flow_penalty
        )

        self._needs_reset = True

        flow_violation_mw = sum(
            max(0.0, abs(flows[k]) - self.f_max[k]) for k in range(self.n_line)
        )

        return self._get_obs(), reward, True, False, {
            "bus_balance_mw":     bus_balance.tolist(),
            "flow_violation_mw":  float(flow_violation_mw),
            "agent_dispatch":     P.tolist(),
        }



