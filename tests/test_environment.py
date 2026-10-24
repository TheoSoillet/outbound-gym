import pytest

from outbound_gym.environment import ProspectingEnv


def matches_icp(industry, headcount, growth_rate):
    # ICP réécrit ici, indépendamment de is_qualified(), pour que le test ne soit pas circulaire
    return industry == "SaaS" and 50 <= headcount <= 500 and growth_rate > 0.20


def find_company_id(env, qualified):
    # Le test est l'arbitre : il a le droit de lire les vraies données de l'épisode
    return next(
        c["id"]
        for c in env.companies
        if matches_icp(c["industry"], c["headcount"], c["growth_rate"]) == qualified
    )


def test_icp_company_gets_plus_one():
    env = ProspectingEnv(seed=0)
    env.reset()
    company_id = find_company_id(env, qualified=True)
    observation, reward, done, info = env.step({"type": "select", "company_id": company_id})
    assert reward == 1
    assert done is True


def test_non_icp_company_gets_minus_one():
    env = ProspectingEnv(seed=0)
    env.reset()
    company_id = find_company_id(env, qualified=False)
    observation, reward, done, info = env.step({"type": "select", "company_id": company_id})
    assert reward == -1
    assert done is True


def test_hidden_attributes_absent_after_reset():
    env = ProspectingEnv(seed=0)
    companies = env.reset()
    assert len(companies) == 20
    for company in companies:
        assert "headcount" not in company
        assert "growth_rate" not in company


def test_research_costs_two_credits():
    env = ProspectingEnv(seed=0)
    env.reset()
    assert env.budget == 10
    observation, reward, done, info = env.step({"type": "research", "company_id": 0})
    assert env.budget == 8
    assert observation["headcount"] == env.companies[0]["headcount"]
    assert observation["growth_rate"] == env.companies[0]["growth_rate"]
    assert reward == 0
    assert done is False


def test_research_refused_when_budget_below_two():
    env = ProspectingEnv(seed=0)
    env.reset()
    for company_id in range(5):  # 5 recherches : 10 -> 0 crédit
        env.step({"type": "research", "company_id": company_id})
    observation, reward, done, info = env.step({"type": "research", "company_id": 5})
    assert info["refused"] is True
    assert "headcount" not in observation
    assert env.budget == 0


def test_same_seed_gives_same_episodes():
    env_a = ProspectingEnv(seed=42)
    env_b = ProspectingEnv(seed=42)
    for _ in range(3):  # la même succession d'épisodes, pas seulement le premier
        assert env_a.reset() == env_b.reset()
        assert env_a.companies == env_b.companies


def test_successive_resets_give_different_companies():
    env = ProspectingEnv(seed=42)
    env.reset()
    first = env.companies
    env.reset()
    assert env.companies != first


def test_reward_matches_icp_after_research():
    # Le test joue un agent parfait : il ne décide qu'avec reset() + research()
    for qualified in (True, False):
        env = ProspectingEnv(seed=7)
        companies = env.reset()
        company_id = find_company_id(env, qualified)
        observation, _, _, _ = env.step({"type": "research", "company_id": company_id})
        decision = matches_icp(
            companies[company_id]["industry"], observation["headcount"], observation["growth_rate"]
        )
        _, reward, done, _ = env.step({"type": "select", "company_id": company_id})
        assert decision is qualified
        assert reward == pytest.approx((1 if decision else -1) - 0.05 * 2)  # une recherche = 2 crédits
        assert done is True


@pytest.mark.parametrize(
    "qualified, num_researches, expected_reward, expected_credits",
    [
        (True, 0, 1.00, 0),    # réussite, 0 crédit
        (True, 1, 0.90, 2),    # réussite, 2 crédits
        (False, 2, -1.20, 4),  # échec, 4 crédits
    ],
)
def test_select_reward_includes_cost_penalty(qualified, num_researches, expected_reward, expected_credits):
    env = ProspectingEnv(seed=0)
    env.reset()
    for company_id in range(num_researches):
        env.step({"type": "research", "company_id": company_id})
    company_id = find_company_id(env, qualified)
    observation, reward, done, info = env.step({"type": "select", "company_id": company_id})
    assert reward == pytest.approx(expected_reward)
    assert info["credits_spent"] == expected_credits
