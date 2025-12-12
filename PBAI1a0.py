#!/usr/bin/env python3
"""
pbai_deep_sim.py

Deep PBAI Casino Test simulation (tabular Q-learning + Bayesian-like language learner)
Reproduces the deep run: logs drives (Comfort, Desire, Fear, Joy, Stability),
goal achievement, belief accuracy, Q evolution, and saves plots + CSV.

Usage:
    python pbai_deep_sim.py

Outputs (saved to current working directory):
    - pbai_total_reward.png
    - pbai_goal_achievement.png
    - pbai_avg_q.png
    - pbai_comfort.png
    - pbai_joy_fear.png
    - pbai_desire.png
    - pbai_stability.png
    - pbai_belief_accuracy.png
    - pbai_deep_sim_results.csv
"""

import os
import math
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Configuration / Hyperparams
# -------------------------
SEED = 1
random.seed(SEED)
np.random.seed(SEED)

NUM_GAMES = 5
KNOWN_LANGUAGE_INDEX = 0
TOKENS_PER_LANGUAGE = 6  # tokens per hidden language
PLAY_COST = 0.15

# Hidden game parameters: varied probabilities and payouts to encourage exploration
GAMES_TRUE = [
    {"p": 0.75, "payout": 0.8},   # safe, small win
    {"p": 0.45, "payout": 2.0},   # medium risk
    {"p": 0.25, "payout": 4.5},   # risky high reward
    {"p": 0.15, "payout": 7.0},   # very risky
    {"p": 0.50, "payout": 1.5},   # balanced
]

# Q-learning hyperparams
ALPHA_Q = 0.08
GAMMA_Q = 0.98
EPSILON_START = 0.5
EPSILON_END = 0.02
EPSILON_DECAY = 0.9985

# Drives & goal
GOAL_PER_EPISODE = 40.0
XI_NOISE = 0.01
EPSILON_STAB = 1e-6

# Simulation settings
EPISODES = 2000
MAX_STEPS = 100
BUDGET_INITIAL = 20.0

OUT_DIR = os.getcwd()  # change if you want outputs in a specific folder

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Environment helpers
# -------------------------
def make_languages(num_games, tokens_per_language):
    return [[f"l{gi}_t{ti}" for ti in range(tokens_per_language)] for gi in range(num_games)]

languages = make_languages(NUM_GAMES, TOKENS_PER_LANGUAGE)

def emit_token(game_idx):
    return random.choice(languages[game_idx])

def play_game(game_idx):
    g = GAMES_TRUE[game_idx]
    win = random.random() < g["p"]
    reward = g["payout"] if win else 0.0
    return reward - PLAY_COST

# -------------------------
# Agent initialization
# -------------------------
ACTIONS = list(range(NUM_GAMES))
Q = np.zeros(len(ACTIONS))
dirichlet_counts = [defaultdict(int) for _ in range(NUM_GAMES)]
# seed known language tokens as prior for known language
for tok in languages[KNOWN_LANGUAGE_INDEX]:
    dirichlet_counts[KNOWN_LANGUAGE_INDEX][tok] += 5

# -------------------------
# Simulation / Logging
# -------------------------
records = {
    "episode": [], "total_reward": [], "goal_achieved": [], "belief_accuracy": [], "avg_Q": [],
    "mean_comfort": [], "mean_joy": [], "mean_fear": [], "mean_desire": [], "mean_stability": []
}

epsilon = EPSILON_START

def belief_matches_true(game_idx, counts):
    observed = set(counts.keys())
    true_tokens = set(languages[game_idx])
    return len(observed & true_tokens) > 0

for ep in range(EPISODES):
    budget = BUDGET_INITIAL
    total_reward_ep = 0.0
    comforts = []
    joys = []
    fears = []
    desires = []
    stabilities = []
    for step in range(MAX_STEPS):
        # action selection (epsilon-greedy)
        if random.random() < epsilon:
            a = random.choice(ACTIONS)
        else:
            a = int(np.argmax(Q))
        token = emit_token(a)
        prev_count = dirichlet_counts[a].get(token, 0)
        env_reward = play_game(a)
        budget += env_reward
        total_reward_ep += env_reward
        # update token counts
        dirichlet_counts[a][token] += 1
        # compute info_gain proxy (log change in normalized token prob)
        total_tokens_for_game = sum(dirichlet_counts[a].values())
        prev_norm = (prev_count / total_tokens_for_game) if total_tokens_for_game > 0 else 0.0
        curr_norm = dirichlet_counts[a][token] / total_tokens_for_game if total_tokens_for_game > 0 else 0.0
        # guard values and compute a stable info gain proxy:
        info_gain = math.log(curr_norm / (prev_norm + 1e-9) + 1e-9)

        # drives
        joy = max(env_reward, 0.0)
        fear = max(-env_reward, 0.0)
        # comfort uses current budget as denominator (like ln(1 + x/S))
        comfort = math.log(1.0 + max(env_reward, 0.0) / max(budget, 1e-6))
        progress = max(0.0, total_reward_ep)
        desire = (max(GOAL_PER_EPISODE - progress, 0.0) / GOAL_PER_EPISODE) + XI_NOISE
        stability = (comfort * joy) / (desire * (fear + 1e-6) + EPSILON_STAB) + EPSILON_STAB

        # shaped reward (same weighting used in the run)
        r = (1.0 * env_reward +
             0.25 * joy -
             0.6 * fear +
             0.12 * comfort +
             0.35 * info_gain +
             0.18 * stability -
             0.05 * PLAY_COST)

        # Q-learning update (single-state)
        best_next = np.max(Q)
        td_target = r + GAMMA_Q * best_next
        td_error = td_target - Q[a]
        Q[a] += ALPHA_Q * td_error

        # log drives
        comforts.append(comfort)
        joys.append(joy)
        fears.append(fear)
        desires.append(desire)
        stabilities.append(stability)

        # allow deeper negative budget as in the original run
        if budget <= -10:
            break

    # end of episode logging
    records["episode"].append(ep)
    records["total_reward"].append(total_reward_ep)
    records["goal_achieved"].append(1 if total_reward_ep >= GOAL_PER_EPISODE else 0)
    matches = sum(1 for i in range(NUM_GAMES) if belief_matches_true(i, dirichlet_counts[i]))
    records["belief_accuracy"].append(matches / NUM_GAMES)
    records["avg_Q"].append(np.mean(Q))
    records["mean_comfort"].append(np.mean(comforts) if comforts else 0.0)
    records["mean_joy"].append(np.mean(joys) if joys else 0.0)
    records["mean_fear"].append(np.mean(fears) if fears else 0.0)
    records["mean_desire"].append(np.mean(desires) if desires else 0.0)
    records["mean_stability"].append(np.mean(stabilities) if stabilities else 0.0)

    # decay exploration
    epsilon = max(epsilon * EPSILON_DECAY, EPSILON_END)

# finish sim
df = pd.DataFrame(records)

# -------------------------
# Plotting helpers + save
# -------------------------
def save_figure(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved {path}")

WINDOW = 50

# Total reward per episode (raw + smoothed)
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["total_reward"], label="total_reward", color="#d78b00", alpha=0.7)
plt.plot(df["episode"], df["total_reward"].rolling(WINDOW, min_periods=1).mean(), label="smoothed_reward", color="#2b9bd3")
plt.title("Total Reward per Episode (raw and smoothed)")
plt.xlabel("Episode")
plt.ylabel("Total reward this episode")
plt.grid(True)
plt.legend()
save_figure(fig, "pbai_total_reward.png")
plt.close(fig)

# Goal achievement
goal_rate = df["goal_achieved"].rolling(WINDOW, min_periods=1).mean()
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["goal_achieved"], label="goal_achieved (0/1)", color="#d78b00", alpha=0.7)
plt.plot(df["episode"], goal_rate, label=f"smoothed_goal_rate (window={WINDOW})", color="#2b9bd3")
plt.title("Goal Achievement (per episode and smoothed rate)")
plt.xlabel("Episode")
plt.ylabel("Goal achieved (1) / not (0) or smoothed rate")
plt.grid(True)
plt.legend()
save_figure(fig, "pbai_goal_achievement.png")
plt.close(fig)

# Q-value evolution (average)
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["avg_Q"], label="avg_Q (mean of Q-table)", color="#d78b00", alpha=0.8)
plt.plot(df["episode"], df["avg_Q"].rolling(WINDOW, min_periods=1).mean(), label="smoothed_avg_Q", color="#2b9bd3")
plt.title("Q-value evolution (average over actions)")
plt.xlabel("Episode")
plt.ylabel("Average Q-value")
plt.grid(True)
plt.legend()
save_figure(fig, "pbai_avg_q.png")
plt.close(fig)

# Comfort
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["mean_comfort"], label="comfort (mean per episode)", color="#d78b00", alpha=0.7)
plt.plot(df["episode"], df["mean_comfort"].rolling(WINDOW, min_periods=1).mean(), label="comfort (smoothed)", color="#2b9bd3")
plt.title("Comfort over Episodes")
plt.xlabel("Episode")
plt.ylabel("Comfort (mean per episode)")
plt.grid(True)
plt.legend()
save_figure(fig, "pbai_comfort.png")
plt.close(fig)

# Joy & Fear
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["mean_joy"], label="joy (mean per episode)", color="#d78b00", alpha=0.6)
plt.plot(df["episode"], df["mean_fear"], label="fear (mean per episode)", color="#2b9bd3", alpha=0.6)
plt.plot(df["episode"], df["mean_joy"].rolling(WINDOW, min_periods=1).mean(), label="joy (smoothed)", color="#0b7a3f")
plt.plot(df["episode"], df["mean_fear"].rolling(WINDOW, min_periods=1).mean(), label="fear (smoothed)", color="#cabf2f")
plt.title("Joy & Fear over Episodes")
plt.xlabel("Episode")
plt.ylabel("Value (mean per episode)")
plt.grid(True)
plt.legend()
save_figure(fig, "pbai_joy_fear.png")
plt.close(fig)

# Desire
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["mean_desire"], label="desire (mean per episode)", color="#d78b00", alpha=0.7)
plt.plot(df["episode"], df["mean_desire"].rolling(WINDOW, min_periods=1).mean(), label="desire (smoothed)", color="#2b9bd3")
plt.title("Desire over Episodes")
plt.xlabel("Episode")
plt.ylabel("Desire (mean per episode)")
plt.grid(True)
plt.legend()
save_figure(fig, "pbai_desire.png")
plt.close(fig)

# Stability
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["mean_stability"], label="stability (mean per episode)", color="#d78b00", alpha=0.7)
plt.plot(df["episode"], df["mean_stability"].rolling(WINDOW, min_periods=1).mean(), label="stability (smoothed)", color="#2b9bd3")
plt.title("Stability over Episodes")
plt.xlabel("Episode")
plt.ylabel("Stability (mean per episode)")
plt.grid(True)
plt.legend()
save_figure(fig, "pbai_stability.png")
plt.close(fig)

# Belief accuracy
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["belief_accuracy"], label="belief_accuracy", color="#d78b00", alpha=0.7)
plt.plot(df["episode"], df["belief_accuracy"].rolling(WINDOW, min_periods=1).mean(), label="smoothed_belief_accuracy", color="#2b9bd3")
plt.title("Language-belief Accuracy over Episodes")
plt.xlabel("Episode")
plt.ylabel("Fraction of games recognized")
plt.grid(True)
plt.legend()
save_figure(fig, "pbai_belief_accuracy.png")
plt.close(fig)

# Save results CSV
csv_path = os.path.join(OUT_DIR, "pbai_deep_sim_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved results CSV to {csv_path}")

# Final textual summary printed to console
counts_summary = {i: len(dirichlet_counts[i]) for i in range(NUM_GAMES)}
print("Distinct tokens observed per game:", counts_summary)
print(f"Avg episode reward (last 100): {np.mean(df['total_reward'][-100:]):.3f}")
print(f"Goal achievement rate (last 100): {np.mean(df['goal_achieved'][-100:]):.3f}")
print(f"Final belief accuracy: {df['belief_accuracy'].iloc[-1]:.3f}")

# Final Q-table printed and saved
final_q_table = pd.DataFrame({
    "game": list(range(NUM_GAMES)),
    "Q_value": Q,
    "true_p": [g["p"] for g in GAMES_TRUE],
    "true_payout": [g["payout"] for g in GAMES_TRUE]
})
q_path = os.path.join(OUT_DIR, "pbai_final_q_table.csv")
final_q_table.to_csv(q_path, index=False)
print(f"Saved final Q-table to {q_path}")

print("Done.")