import random

INDUSTRIES = ["SaaS", "Fintech", "Retail", "Manufacturing"]
NAME_PREFIXES = ["Nova", "Blue", "Arc", "Peak", "Iron", "Luma", "Vertex", "Orbit", "Cedar", "Pulse"]
NAME_SUFFIXES = ["Labs", "Systems", "Cloud", "Works", "Data", "Pay", "Robotics", "Retail"]


def generate_companies(rng, n=20):
    # Tous les tirages passent par rng : la seed contrôle toute la génération
    names = rng.sample([f"{p} {s}" for p in NAME_PREFIXES for s in NAME_SUFFIXES], n)
    return [
        {
            "id": i,
            "name": names[i],
            "industry": rng.choice(INDUSTRIES),
            "headcount": rng.randint(10, 1000),
            "growth_rate": rng.uniform(0, 1),
        }
        for i in range(n)
    ]


ICP = {"industry": "SaaS", "min_headcount": 50, "max_headcount": 500, "min_growth_rate": 0.20}


def is_qualified(company):
    return (
        company["industry"] == ICP["industry"]
        and ICP["min_headcount"] <= company["headcount"] <= ICP["max_headcount"]
        and company["growth_rate"] > ICP["min_growth_rate"]
    )


INITIAL_BUDGET = 10
RESEARCH_COST = 2
COST_PENALTY = 0.05


class ProspectingEnv:
    def __init__(self, seed=None):
        # Le générateur est créé une seule fois : chaque reset() continue sa séquence
        self.rng = random.Random(seed)

    def reset(self):
        # Régénère tant que l'épisode n'a aucune entreprise qualifiée
        while True:
            self.companies = generate_companies(self.rng)
            if any(is_qualified(c) for c in self.companies):
                break
        self.budget = INITIAL_BUDGET
        # Retourne les entreprises disponibles, sans headcount ni growth_rate
        return [
            {"id": c["id"], "name": c["name"], "industry": c["industry"]}
            for c in self.companies
        ]

    def step(self, action):
        company = self.companies[action["company_id"]]

        if action["type"] == "research":
            # Recherche refusée si le budget ne suffit pas : il ne devient jamais négatif
            if self.budget < RESEARCH_COST:
                observation = {"budget": self.budget}
                return observation, 0, False, {"refused": True}
            self.budget -= RESEARCH_COST
            observation = {
                "id": company["id"],
                "headcount": company["headcount"],
                "growth_rate": company["growth_rate"],
                "budget": self.budget,
            }
            return observation, 0, False, {"refused": False}

        if action["type"] == "select":
            # Évalue la sélection : c'est l'environnement qui calcule la qualification
            qualified = is_qualified(company)
            # Pénalité appliquée une seule fois, sur tous les crédits dépensés pendant l'épisode
            credits_spent = INITIAL_BUDGET - self.budget
            reward = (1 if qualified else -1) - COST_PENALTY * credits_spent
            observation = {"id": company["id"], "budget": self.budget}
            return observation, reward, True, {"qualified": qualified, "credits_spent": credits_spent}

        raise ValueError(f"unknown action type: {action['type']!r}")
