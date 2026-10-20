COMPANIES = [
    {"id": 0, "name": "Nova Labs", "industry": "SaaS", "headcount": 120, "growth_rate": 0.35},
    {"id": 1, "name": "Blue Systems", "industry": "SaaS", "headcount": 30, "growth_rate": 0.50},
    {"id": 2, "name": "Arc Finance", "industry": "Fintech", "headcount": 200, "growth_rate": 0.40},
    {"id": 3, "name": "Peak Software", "industry": "SaaS", "headcount": 450, "growth_rate": 0.25},
    {"id": 4, "name": "Iron Works", "industry": "Manufacturing", "headcount": 800, "growth_rate": 0.05},
    {"id": 5, "name": "Luma Cloud", "industry": "SaaS", "headcount": 80, "growth_rate": 0.10},
    {"id": 6, "name": "Vertex Health", "industry": "Healthcare", "headcount": 300, "growth_rate": 0.30},
    {"id": 7, "name": "Orbit Analytics", "industry": "SaaS", "headcount": 600, "growth_rate": 0.45},
    {"id": 8, "name": "Cedar Retail", "industry": "Retail", "headcount": 150, "growth_rate": 0.15},
    {"id": 9, "name": "Pulse CRM", "industry": "SaaS", "headcount": 250, "growth_rate": 0.60},
    {"id": 10, "name": "Atlas Logistics", "industry": "Logistics", "headcount": 400, "growth_rate": 0.22},
    {"id": 11, "name": "Echo HR", "industry": "SaaS", "headcount": 50, "growth_rate": 0.21},
    {"id": 12, "name": "Summit Pay", "industry": "Fintech", "headcount": 90, "growth_rate": 0.70},
    {"id": 13, "name": "Drift Data", "industry": "SaaS", "headcount": 500, "growth_rate": 0.20},
    {"id": 14, "name": "Harbor Foods", "industry": "Retail", "headcount": 1200, "growth_rate": 0.08},
    {"id": 15, "name": "Quartz Security", "industry": "SaaS", "headcount": 180, "growth_rate": 0.32},
    {"id": 16, "name": "Maple Clinics", "industry": "Healthcare", "headcount": 60, "growth_rate": 0.12},
    {"id": 17, "name": "Zenith Ops", "industry": "SaaS", "headcount": 10, "growth_rate": 0.90},
    {"id": 18, "name": "Forge Robotics", "industry": "Manufacturing", "headcount": 350, "growth_rate": 0.28},
    {"id": 19, "name": "Signal Mail", "industry": "SaaS", "headcount": 320, "growth_rate": 0.18},
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


class ProspectingEnv:
    def reset(self):
        # Retourne les entreprises disponibles, sans headcount ni growth_rate
        self.budget = INITIAL_BUDGET
        return [
            {"id": c["id"], "name": c["name"], "industry": c["industry"]}
            for c in COMPANIES
        ]

    def step(self, action):
        company = COMPANIES[action["company_id"]]

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
            reward = 1 if qualified else -1
            observation = {"id": company["id"], "budget": self.budget}
            return observation, reward, True, {"qualified": qualified}

        raise ValueError(f"unknown action type: {action['type']!r}")
