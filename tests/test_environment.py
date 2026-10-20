from outbound_gym.environment import ProspectingEnv


def test_icp_company_gets_plus_one():
    env = ProspectingEnv()
    env.reset()
    observation, reward, done, info = env.step(0)  # Nova Labs : SaaS, 120 employés, +35 %
    assert reward == 1
    assert done is True


def test_non_icp_company_gets_minus_one():
    env = ProspectingEnv()
    env.reset()
    observation, reward, done, info = env.step(4)  # Iron Works : Manufacturing, 800 employés, +5 %
    assert reward == -1
    assert done is True
