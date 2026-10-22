import random

from agents import random_agent, rule_based_agent
from outbound_gym.environment import ProspectingEnv

NUM_EPISODES = 100
# Seed distincte pour l'agent, pour que son hasard soit indépendant de celui de l'environnement
AGENT_SEED_OFFSET = 1_000_000


def evaluate(name, play):
    rewards = []
    for i in range(NUM_EPISODES):
        # Chaque épisode i est autonome : on peut le rejouer seul avec les mêmes seeds,
        # et tous les agents voient exactement les mêmes entreprises
        env = ProspectingEnv(seed=i)
        rewards.append(play(env, i))

    success_rate = sum(1 for r in rewards if r == 1) / NUM_EPISODES
    mean_reward = sum(rewards) / NUM_EPISODES

    print(f"{name} — {NUM_EPISODES} episodes")
    print(f"Success rate: {success_rate:.0%}")
    print(f"Mean reward: {mean_reward:.2f}")


def main():
    evaluate(
        "Random Agent",
        lambda env, i: random_agent.run_episode(env, random.Random(AGENT_SEED_OFFSET + i)),
    )
    print()
    evaluate("Rule-Based Agent", lambda env, i: rule_based_agent.run_episode(env))


if __name__ == "__main__":
    main()
