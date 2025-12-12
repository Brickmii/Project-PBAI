#!/usr/bin/env python3
"""
pbai_ac_deep_sim.py

PyTorch Actor-Critic implementation of the PBAI Casino Test.

- Actor-Critic network chooses among 5 games.
- Observation vector: flattened normalized per-game token counts + budget_norm + progress_norm.
- Drives (Comfort, Desire, Fear, Joy, Stability) computed and used in a shaped reward.
- Tracks and saves graphs and CSV to ./outputs (or change OUT_DIR).
"""

import os
import math
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Configuration (tweak as needed)
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Environment
NUM_GAMES = 5
TOKENS_PER_LANGUAGE = 6
KNOWN_LANGUAGE_INDEX = 0
PLAY_COST = 0.15
languages = [[f"l{g}_t{t}" for t in range(TOKENS_PER_LANGUAGE)] for g in range(NUM_GAMES)]
# Hidden game parameters (same as prior experiments)
GAMES_TRUE = [
    {"p": 0.75, "payout": 0.8},
    {"p": 0.45, "payout": 2.0},
    {"p": 0.25, "payout": 4.5},
    {"p": 0.15, "payout": 7.0},
    {"p": 0.50, "payout": 1.5},
]

def emit_token(game_idx):
    return random.choice(languages[game_idx])

def play_game(game_idx):
    g = GAMES_TRUE[game_idx]
    win = random.random() < g["p"]
    reward = g["payout"] if win else 0.0
    return reward - PLAY_COST

# -------------------------
# Actor-Critic model
# -------------------------
OBS_COUNTS = NUM_GAMES * TOKENS_PER_LANGUAGE
OBS_EXTRA = 2   # budget_norm, progress_norm
OBS_DIM = OBS_COUNTS + OBS_EXTRA
HIDDEN = 128

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value  = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value

# -------------------------
# Hyperparameters
# -------------------------
LR = 3e-4
GAMMA = 0.99
ENTROPY_BETA = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

EPISODES = 2000          # increase if you run locally and have time
MAX_STEPS = 100
BUDGET_INITIAL = 20.0
GOAL_PER_EPISODE = 40.0
XI_NOISE = 0.01
EPSILON_STAB = 1e-6

device = torch.device("cpu")
model = ActorCritic(OBS_DIM, HIDDEN, NUM_GAMES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Observation builder
# -------------------------
def obs_from_counts(dirichlet_counts, budget, cumulative_reward):
    flat = []
    for gi in range(NUM_GAMES):
        total = sum(dirichlet_counts[gi].values()) + 1e-9
        for ti in range(TOKENS_PER_LANGUAGE):
            tok = languages[gi][ti]
            flat.append(dirichlet_counts[gi].get(tok, 0) / total)
    budget_norm = np.tanh(budget / (BUDGET_INITIAL * 2.0))
    progress_norm = np.tanh(cumulative_reward / (GOAL_PER_EPISODE * 2.0))
    return np.array(flat + [budget_norm, progress_norm], dtype=np.float32)

# -------------------------
# Logging structures
# -------------------------
records = {
    "episode": [], "total_reward": [], "goal_achieved": [], "belief_accuracy": [], "avg_value": [], "avg_entropy": [],
    "mean_comfort": [], "mean_joy": [], "mean_fear": [], "mean_desire": [], "mean_stability": []
}

# Dirichlet-like token counts (agent's beliefs)
dirichlet_counts = [defaultdict(int) for _ in range(NUM_GAMES)]
# seed known language with a prior
for tok in languages[KNOWN_LANGUAGE_INDEX]:
    dirichlet_counts[KNOWN_LANGUAGE_INDEX][tok] += 5

def belief_matches_true(game_idx, counts):
    observed = set(counts.keys())
    true_tokens = set(languages[game_idx])
    return len(observed & true_tokens) > 0

# -------------------------
# Training loop (A2C-like on-policy per-episode updates)
# -------------------------
for ep in range(EPISODES):
    budget = BUDGET_INITIAL
    cumulative_reward = 0.0
    log_probs = []
    values = []
    rewards = []
    entropies = []

    comforts = []; joys = []; fears = []; desires = []; stabilities = []

    for step in range(MAX_STEPS):
        obs_vec = obs_from_counts(dirichlet_counts, budget, cumulative_reward)
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0).to(device)
        logits, value = model(obs_t.float())
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        logp = dist.log_prob(torch.tensor(action))
        entropy = dist.entropy().item()

        # environment step
        token = emit_token(action)
        prev_count = dirichlet_counts[action].get(token, 0)
        env_reward = play_game(action)
        budget += env_reward
        cumulative_reward += env_reward
        dirichlet_counts[action][token] += 1

        # info gain proxy (log change in normalized token probability)
        total_tokens_for_game = sum(dirichlet_counts[action].values())
        prev_norm = (prev_count / total_tokens_for_game) if total_tokens_for_game > 0 else 0.0
        curr_norm = dirichlet_counts[action][token] / total_tokens_for_game if total_tokens_for_game > 0 else 0.0
        info_gain = math.log(curr_norm / (prev_norm + 1e-9) + 1e-9)

        # drives
        joy = max(env_reward, 0.0)
        fear = max(-env_reward, 0.0)
        comfort = math.log(1.0 + max(env_reward, 0.0) / max(budget, 1e-6))
        progress = max(0.0, cumulative_reward)
        desire = (max(GOAL_PER_EPISODE - progress, 0.0) / GOAL_PER_EPISODE) + XI_NOISE
        stability = (comfort * joy) / (desire * (fear + 1e-6) + EPSILON_STAB) + EPSILON_STAB

        # shaped reward
        shaped_r = (1.0 * env_reward +
                    0.25 * joy -
                    0.6 * fear +
                    0.12 * comfort +
                    0.35 * info_gain +
                    0.18 * stability -
                    0.05 * PLAY_COST)

        log_probs.append(logp)
        values.append(value.squeeze(0))
        rewards.append(shaped_r)
        entropies.append(entropy)

        comforts.append(comfort); joys.append(joy); fears.append(fear); desires.append(desire); stabilities.append(stability)

        if budget <= -10:
            break

    # Compute returns and update actor-critic
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)

    if len(returns) > 0:
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values_tensor = torch.stack(values).to(device)
        logp_tensor = torch.stack(log_probs).to(device)
        entropies_tensor = torch.tensor(entropies, dtype=torch.float32).to(device)

        advantages = returns - values_tensor.detach().squeeze(-1)
        actor_loss = -(logp_tensor * advantages).mean() - ENTROPY_BETA * entropies_tensor.mean()
        critic_loss = VALUE_COEF * (returns - values_tensor.squeeze(-1)).pow(2).mean()
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    # Logging
    records["episode"].append(ep)
    records["total_reward"].append(cumulative_reward)
    records["goal_achieved"].append(1 if cumulative_reward >= GOAL_PER_EPISODE else 0)
    matches = sum(1 for i in range(NUM_GAMES) if belief_matches_true(i, dirichlet_counts[i]))
    records["belief_accuracy"].append(matches / NUM_GAMES)
    records["avg_value"].append(values_tensor.mean().item() if len(returns) > 0 else 0.0)
    records["avg_entropy"].append(entropies_tensor.mean().item() if len(entropies) > 0 else 0.0)
    records["mean_comfort"].append(float(np.mean(comforts)) if comforts else 0.0)
    records["mean_joy"].append(float(np.mean(joys)) if joys else 0.0)
    records["mean_fear"].append(float(np.mean(fears)) if fears else 0.0)
    records["mean_desire"].append(float(np.mean(desires)) if desires else 0.0)
    records["mean_stability"].append(float(np.mean(stabilities)) if stabilities else 0.0)

# -------------------------
# Save results and plots
# -------------------------
df = pd.DataFrame(records)
WINDOW = max(1, EPISODES // 40)

def save_plot(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)

# Total reward
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["total_reward"], label="total_reward")
plt.plot(df["episode"], df["total_reward"].rolling(WINDOW, min_periods=1).mean(), label="smoothed")
plt.title("Total Reward per Episode (Actor-Critic)")
plt.xlabel("Episode"); plt.ylabel("Total reward"); plt.legend(); plt.grid(True)
save_plot(fig, "pbai_ac_total_reward.png"); plt.close(fig)

# Goal achievement
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["goal_achieved"], label="goal_achieved (0/1)")
plt.plot(df["episode"], df["goal_achieved"].rolling(WINDOW, min_periods=1).mean(), label="smoothed_goal_rate")
plt.title("Goal Achievement"); plt.xlabel("Episode"); plt.ylabel("Goal (0/1)"); plt.legend(); plt.grid(True)
save_plot(fig, "pbai_ac_goal_achievement.png"); plt.close(fig)

# Value evolution
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["avg_value"], label="avg_value")
plt.plot(df["episode"], df["avg_value"].rolling(WINDOW, min_periods=1).mean(), label="smoothed_avg_value")
plt.title("Value evolution"); plt.xlabel("Episode"); plt.ylabel("Average value"); plt.legend(); plt.grid(True)
save_plot(fig, "pbai_ac_value.png"); plt.close(fig)

# Emotion plots
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["mean_comfort"], label="comfort")
plt.plot(df["episode"], df["mean_comfort"].rolling(WINDOW, min_periods=1).mean(), label="comfort smoothed")
plt.title("Comfort"); plt.legend(); plt.grid(True)
save_plot(fig, "pbai_ac_comfort.png"); plt.close(fig)

fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["mean_joy"], label="joy")
plt.plot(df["episode"], df["mean_fear"], label="fear")
plt.plot(df["episode"], df["mean_joy"].rolling(WINDOW, min_periods=1).mean(), label="joy smoothed")
plt.plot(df["episode"], df["mean_fear"].rolling(WINDOW, min_periods=1).mean(), label="fear smoothed")
plt.title("Joy & Fear"); plt.legend(); plt.grid(True)
save_plot(fig, "pbai_ac_joy_fear.png"); plt.close(fig)

fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["mean_desire"], label="desire")
plt.plot(df["episode"], df["mean_desire"].rolling(WINDOW, min_periods=1).mean(), label="desire smoothed")
plt.title("Desire"); plt.legend(); plt.grid(True)
save_plot(fig, "pbai_ac_desire.png"); plt.close(fig)

fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["mean_stability"], label="stability")
plt.plot(df["episode"], df["mean_stability"].rolling(WINDOW, min_periods=1).mean(), label="stability smoothed")
plt.title("Stability"); plt.legend(); plt.grid(True)
save_plot(fig, "pbai_ac_stability.png"); plt.close(fig)

# Belief accuracy
fig = plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["belief_accuracy"], label="belief_accuracy")
plt.plot(df["episode"], df["belief_accuracy"].rolling(WINDOW, min_periods=1).mean(), label="smoothed")
plt.title("Language-belief Accuracy"); plt.legend(); plt.grid(True)
save_plot(fig, "pbai_ac_belief_accuracy.png"); plt.close(fig)

# Save CSV and final policy
csv_path = os.path.join(OUT_DIR, "pbai_ac_deep_sim_results.csv")
df.to_csv(csv_path, index=False)

# Final policy: evaluate action probabilities on current belief state
def make_eval_obs(counts):
    flat = []
    for gi in range(NUM_GAMES):
        total = sum(counts[gi].values()) + 1e-9
        for ti in range(TOKENS_PER_LANGUAGE):
            tok = languages[gi][ti]
            flat.append(counts[gi].get(tok, 0) / total)
    budget_norm = np.tanh(BUDGET_INITIAL / (BUDGET_INITIAL * 2.0))
    progress_norm = np.tanh(0.0 / (GOAL_PER_EPISODE * 2.0))
    return np.array(flat + [budget_norm, progress_norm], dtype=np.float32)

eval_vec = torch.from_numpy(make_eval_obs(dirichlet_counts)).unsqueeze(0)
with torch.no_grad():
    logits, value = model(eval_vec)
    probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)

final_policy_df = pd.DataFrame({
    "game": list(range(NUM_GAMES)),
    "action_prob": probs,
    "true_p": [g["p"] for g in GAMES_TRUE],
    "true_payout": [g["payout"] for g in GAMES_TRUE]
})
final_policy_df.to_csv(os.path.join(OUT_DIR, "pbai_ac_final_policy.csv"), index=False)

# Summary print
print("Saved outputs (plots + CSV + final policy) to:", OUT_DIR)
print("Avg episode reward (last 100):", df["total_reward"].tail(100).mean())
print("Goal achievement rate (last 100):", df["goal_achieved"].tail(100).mean())
print("Final belief accuracy:", df["belief_accuracy"].iloc[-1])