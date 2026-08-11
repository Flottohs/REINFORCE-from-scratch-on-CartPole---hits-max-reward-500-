# REINFORCE-from-scratch-on-CartPole---hits-max-reward-500-


# REINFORCE on CartPole

A from-scratch implementation of REINFORCE (vanilla policy gradient) trained on OpenAI Gymnasium's CartPole-v1, built as part of self-directed reinforcement learning study.

## Result

Trained agent reaches CartPole's maximum reward of 500 (the pole balanced for the full episode length).

## What this is

A minimal policy-gradient agent:
- A small 2-layer MLP policy network (`Linear(4,64) → ReLU → Linear(64,2)`) outputting a `Categorical` distribution over actions
- Trajectories are collected by sampling actions from the current policy (on-policy, Monte Carlo)
- Discounted reward-to-go is computed per timestep (not full-trajectory return), so each action is only credited for the reward it could have actually caused
- Returns are normalised (mean/std) as a variance-reducing baseline
- The policy is updated via a loss function constructed so that `loss.backward()` produces exactly the policy gradient: `∇_θ J(θ) = E[∇_θ log π_θ(a|s) · G_t]`

## What this doesn't use

No value function, no Bellman equation, no critic — this is a pure Monte Carlo policy gradient method. Value-function-based baselines (Actor-Critic, PPO) are a natural next step on top of this.

## Requirements

```
torch
gymnasium
numpy
```

## Run

```bash
python reinforce_cartpole.py
```

## Background

Built while working through Stanford CS224R (Deep Reinforcement Learning), deriving the policy gradient theorem — expectation, the log-derivative trick, Monte Carlo estimation, and reward-to-go — from first principles before implementing.
