export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55360/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8360

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m qi.seed --if-empty
dev:            ## API with reload on :8360, jobs run by a local worker
	($(PY) -m core.jobs qi.api &) && $(PY) -m uvicorn qi.api:app --reload --port 8360
test:
	$(PY) -m pytest --cov=qi --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) plant-viewer-demo 32 10 "GET /v1/lines/161a8ee2-6cb8-5e23-a961-456168e3938f/quality" 
evaluate:       ## held-out lines, all three products -> docs/evaluation.md
	$(PY) -m qi.evaluate > docs/evaluation.md
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test evaluate reset
