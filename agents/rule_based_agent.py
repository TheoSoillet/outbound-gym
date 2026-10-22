def matches_icp(industry, headcount, growth_rate):
    return industry == "SaaS" and 50 <= headcount <= 500 and growth_rate > 0.20


def select(env, company_id):
    observation, reward, done, info = env.step({"type": "select", "company_id": company_id})
    return reward


def run_episode(env):
    """
    Joue un épisode avec une stratégie déterministe.
    Retourne la reward finale.
    """
    # L'agent n'utilise que ce que reset() et step() lui renvoient
    companies = env.reset()
    saas = [c for c in companies if c["industry"] == "SaaS"]

    researched = set()
    for company in saas:
        observation, reward, done, info = env.step({"type": "research", "company_id": company["id"]})
        if info["refused"]:
            # Budget épuisé : l'agent l'apprend par le refus, sans compter lui-même
            break
        researched.add(company["id"])
        if matches_icp(company["industry"], observation["headcount"], observation["growth_rate"]):
            return select(env, company["id"])

    # Secours : une SaaS non recherchée a encore des chances d'être qualifiée,
    # contrairement à une SaaS déjà recherchée ou à une autre industrie
    unresearched = [c for c in saas if c["id"] not in researched]
    fallback = unresearched[0] if unresearched else companies[0]
    return select(env, fallback["id"])
