# RL Training Data Platform for LLM Agents

An advanced project that builds the data platform behind RLHF-style training of LLM agents — the pipeline from a task definition to a usable reward signal. You model RL tasks and the rollouts an agent produces, collect human preference judgments (the A-vs-B comparisons at the heart of RLHF), aggregate them into reliable labels, then train a tiny logistic reward model on those preferences and use it to score new rollouts. You add inter-annotator agreement to measure label quality, a data-quality monitor that flags degrading feedback, and a held-out eval suite that tracks whether the reward model actually improves agent ranking over time. Everything ships as a FastAPI service with a worker and docker-compose. By the end you own the platform machinery that turns human judgment into a training signal you can trust.

Built step-by-step with [KhwajaLabs Build](https://khwajalabs.com).

## Stack
- Python
- FastAPI
- NumPy
- Pydantic
- Docker
- docker-compose
