def run_episode(env, rng):
    """
    Joue un épisode complet.
    Retourne la reward obtenue.
    """
    # L'agent n'utilise que ce que reset() et step() lui renvoient
    companies = env.reset()
    company = rng.choice(companies)
    observation, reward, done, info = env.step({"type": "select", "company_id": company["id"]})
    return reward
