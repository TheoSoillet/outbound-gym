import random

from agents.random_agent import run_episode
from outbound_gym.environment import ProspectingEnv

NUM_EPISODES = 100
# Seed distincte pour l'agent, pour que son hasard soit indépendant de celui de l'environnement
AGENT_SEED_OFFSET = 1_000_000


def main():
    rewards = []
    for i in range(NUM_EPISODES):
        # Chaque épisode i est autonome : on peut le rejouer seul avec les mêmes seeds
        env = ProspectingEnv(seed=i)
        rng = random.Random(AGENT_SEED_OFFSET + i)
        rewards.append(run_episode(env, rng))

    success_rate = sum(1 for r in rewards if r == 1) / NUM_EPISODES
    mean_reward = sum(rewards) / NUM_EPISODES

    print(f"Random Agent — {NUM_EPISODES} episodes")
    print(f"Success rate: {success_rate:.0%}")
    print(f"Mean reward: {mean_reward:.2f}")


if __name__ == "__main__":
    main()
