from outbound_gym.environment import ProspectingEnv


def test_icp_company_gets_plus_one():
    env = ProspectingEnv()
    env.reset()
    observation, reward, done, info = env.step({"type": "select", "company_id": 0})  # Nova Labs : SaaS, 120 employés, +35 %
    assert reward == 1
    assert done is True


def test_non_icp_company_gets_minus_one():
    env = ProspectingEnv()
    env.reset()
    observation, reward, done, info = env.step({"type": "select", "company_id": 4})  # Iron Works : Manufacturing, 800 employés, +5 %
    assert reward == -1
    assert done is True


def test_hidden_attributes_absent_after_reset():
    env = ProspectingEnv()
    companies = env.reset()
    assert len(companies) == 20
    for company in companies:
        assert "headcount" not in company
        assert "growth_rate" not in company


def test_research_costs_two_credits():
    env = ProspectingEnv()
    env.reset()
    assert env.budget == 10
    observation, reward, done, info = env.step({"type": "research", "company_id": 0})
    assert env.budget == 8
    assert observation["headcount"] == 120
    assert observation["growth_rate"] == 0.35
    assert reward == 0
    assert done is False


def test_research_refused_when_budget_below_two():
    env = ProspectingEnv()
    env.reset()
    for company_id in range(5):  # 5 recherches : 10 -> 0 crédit
        env.step({"type": "research", "company_id": company_id})
    observation, reward, done, info = env.step({"type": "research", "company_id": 5})
    assert info["refused"] is True
    assert "headcount" not in observation
    assert env.budget == 0
