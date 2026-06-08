import pandas as pd
import matplotlib.pyplot as plt

USE_CARDIFF = True

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 26,
        "font.size": 18,
        "legend.fontsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    }
)

SMOOTH_WINDOW = 5  # try 3, 5, or 7

NUM_TIMESTEPS = 100


def load_runs(pattern, run_ids):
    dfs = []

    for i in run_ids:
        fname = pattern.format(i=i)

        df = pd.read_csv(fname)

        df["epoch"] = (
            df["epoch"]
            .astype(int)
        )
        df["avg_impact_actions"] = df["avg_impact_actions"] / NUM_TIMESTEPS  # normalize to [0, 1]

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    grouped = (
        all_df.groupby("epoch")["avg_impact_actions"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("epoch")
    )

    grouped["std"] = grouped["std"].fillna(0)
    

    # Smooth mean/std
    # grouped["mean_smooth"] = (
    #     grouped["mean"]
    #     .rolling(
    #         window=SMOOTH_WINDOW,
    #         min_periods=1,
    #     )
    #     .mean()
    # )

    # grouped["std_smooth"] = (
    #     grouped["std"]
    #     .rolling(
    #         window=SMOOTH_WINDOW,
    #         min_periods=1,
    #     )
    #     .mean()
    # )

    return grouped


def plot_group(grouped, label):
    x = grouped["epoch"]
    y = grouped["mean"]
    s = grouped["std"]

    plt.plot(x, y, linewidth=2, label=label)

    plt.fill_between(
        x,
        y - s,
        (y + s).clip(upper=1.0),  # clip upper bound to 1.0
        alpha=0.25,
    )


# Which runs to load
run_ids = [0, 1, 2, 3]
# run_ids = range(5)

# Load all experiment groups
base_reward_stealth_grouped = load_runs(
    "logs/red/red_gcn_B_base_reward_stealth_{i}.txt" if not USE_CARDIFF else "logs/red/cardiff_red_gcn_B_base_reward_stealth_{i}.txt",
    run_ids,
)
base_reward_flood_grouped = load_runs(
    "logs/red/red_gcn_B_base_reward_flood_{i}.txt" if not USE_CARDIFF else "logs/red/cardiff_red_gcn_B_base_reward_flood_{i}.txt",
    run_ids,
)
base_reward_stealth_flood_grouped = load_runs(
    "logs/red/red_gcn_B_base_reward_stealth_flood_{i}.txt" if not USE_CARDIFF else "logs/red/cardiff_red_gcn_B_base_reward_stealth_flood_{i}.txt",
    run_ids,
)

rbt_no_phase_stealth_grouped = load_runs(
    "logs/red/red_gcn_B_rbt_no_phase_stealth_{i}.txt" if not USE_CARDIFF else "logs/red/cardiff_red_gcn_B_rbt_no_phase_stealth_{i}.txt",
    run_ids,
)
rbt_no_phase_flood_grouped = load_runs(
    "logs/red/red_gcn_B_rbt_no_phase_flood_{i}.txt" if not USE_CARDIFF else "logs/red/cardiff_red_gcn_B_rbt_no_phase_flood_{i}.txt",
    run_ids,
)
rbt_no_phase_stealth_flood_grouped = load_runs(
    "logs/red/red_gcn_B_rbt_no_phase_stealth_flood_{i}.txt" if not USE_CARDIFF else "logs/red/cardiff_red_gcn_B_rbt_no_phase_stealth_flood_{i}.txt",
    run_ids,
)

rbt_stealth_grouped = load_runs(
    "logs/red/red_gcn_B_rbt_stealth_{i}.txt" if not USE_CARDIFF else "logs/red/cardiff_red_gcn_B_rbt_stealth_{i}.txt",
    run_ids,
)
rbt_flood_grouped = load_runs(
    "logs/red/red_gcn_B_rbt_flood_{i}.txt" if not USE_CARDIFF else "logs/red/cardiff_red_gcn_B_rbt_flood_{i}.txt",
    run_ids,
)
rbt_stealth_flood_grouped = load_runs(
    "logs/red/red_gcn_B_rbt_stealth_flood_{i}.txt" if not USE_CARDIFF else "logs/red/cardiff_red_gcn_B_rbt_stealth_flood_{i}.txt",
    run_ids,
)

# Plot
plt.figure(figsize=(8, 8))

plot_group(base_reward_stealth_grouped, "ARL (CC2)")
# plt.plot(base_reward_stealth_grouped["epoch"], [1.0]*base_reward_stealth_grouped.shape[0], label="ARLBT (Reward Only)", linestyle="--", linewidth=2, alpha=0.5)
plot_group(rbt_no_phase_stealth_grouped, "ARLBT (Reward Only)")
plot_group(rbt_stealth_grouped, "ARLBT (Full)")
plt.plot(base_reward_stealth_grouped["epoch"], [1.0]*base_reward_stealth_grouped.shape[0], label="Theoretical Maximum", linestyle="--", linewidth=4, alpha=0.5)

plt.xlabel("Training Iteration")
plt.ylabel("Mean Attacker Success")
plt.legend(loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.savefig("img/keep_red_eval_stealth.pdf" if not USE_CARDIFF else "img/cardiff_red_eval_stealth.pdf")

# clear the plot
plt.clf()

plt.figure(figsize=(8, 8))

plot_group(base_reward_flood_grouped, "ARL (CC2)")
plot_group(rbt_no_phase_flood_grouped, "ARLBT (Reward Only)")
plot_group(rbt_flood_grouped, "ARLBT (Full)")
plt.plot(base_reward_flood_grouped["epoch"], [1.0]*base_reward_flood_grouped.shape[0], label="Theoretical Maximum", linestyle="--", linewidth=4, alpha=0.5)

plt.xlabel("Training Iteration")
plt.ylabel("Mean Attacker Success")
plt.legend(loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.savefig("img/keep_red_eval_flood.pdf" if not USE_CARDIFF else "img/cardiff_red_eval_flood.pdf")

# clear the plot
plt.clf()

plt.figure(figsize=(8, 8))

plot_group(base_reward_stealth_flood_grouped, "ARL (CC2)")
plot_group(rbt_no_phase_stealth_flood_grouped, "ARLBT (Reward Only)")
plot_group(rbt_stealth_flood_grouped, "ARLBT (Full)")
plt.plot(base_reward_stealth_flood_grouped["epoch"], [1.0]*base_reward_stealth_flood_grouped.shape[0], label="Theoretical Maximum", linestyle="--", linewidth=4, alpha=0.5)

plt.xlabel("Training Iteration")
plt.ylabel("Mean Attacker Success")
plt.legend(loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.savefig("img/keep_red_eval_stealth_flood.pdf" if not USE_CARDIFF else "img/cardiff_red_eval_stealth_flood.pdf")

