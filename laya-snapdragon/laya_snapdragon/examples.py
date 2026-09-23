"""Inputs for verify and bench: short and long states, three languages, all three question types."""

EMAIL = {
    "from": "sam@example.invalid",
    "subject": "Question about monthly report export",
    "body": "Could you let us know whether CSV export is planned? Our team reviews this report each week.",
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}

GERMAN = ("Seit dem letzten Update stürzt die App beim Öffnen ab. Ich habe sie schon zweimal neu installiert, "
          "aber es hilft nichts. Bitte um schnelle Hilfe, ich brauche sie morgen für eine Präsentation.")
SPANISH = "¿Tienen descuentos para equipos de más de cincuenta personas? Estamos comparando proveedores este mes."

# Synthetic reliability scenario for local model testing.
LONG = " ".join([
    "Synthetic reliability exercise prepared only for local model testing.",
    "A routine software release changed a worker-pool setting, and background requests began to accumulate.",
    "The service remained available, but response times increased and some requests returned temporary errors.",
    "The on-call engineer paused the rollout, restored the previous configuration, and let queued work drain.",
    "Service health returned to normal after the rollback. This scenario describes no real system or people.",
    "The review found that the setting was duplicated across two configuration files and only one was checked.",
    "Follow-up actions are to keep one authoritative setting, add a latency alert, and include this path in",
    "the release checklist. The team will document rollback ownership for each support shift.",
    "The exercise asks reviewers to assess the temporary service disruption and decide whether to prioritize",
    "preventive work. These fictional details are included only as a repeatable local benchmark fixture.",
] * 4)

CASES = [
    ("email (en)", EMAIL, QUESTIONS),
    ("support (de)", GERMAN, QUESTIONS),
    ("sales (es)", SPANISH, QUESTIONS),
    ("incident, long", LONG, {
        "severity": {"type": "score", "instructions": "How severe was this incident for customers?",
                     "criteria": ["negligible", "minor", "major", "critical"]},
        "needs_leadership": {"type": "noul", "instructions": "Does leadership need to make a decision?"},
        "area": {"type": "choice", "instructions": "Which area caused the incident?",
                 "criteria": ["database configuration", "mobile app", "payment provider", "network"]},
    }),
]
