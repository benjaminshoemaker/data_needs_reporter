.PHONY: fixtures test test-ci

fixtures:
	poetry run dnr-synth init fintech
	poetry run dnr-synth generate --config domains/fintech/config.yaml --seed 424242 --out data/fintech_golden
	poetry run dnr-synth evaluate --domain fintech --data data/fintech_golden --out artifacts/fintech_golden
	poetry run dnr-synth sample --domain fintech --out artifacts/fintech_golden --seed 424242
	@echo "Golden dataset in data/fintech_golden; artifacts in artifacts/fintech_golden"

test:
	poetry run pytest -q

test-ci:
	PYTHONWARNINGS="ignore:Passing a BlockManager to DataFrame is deprecated:DeprecationWarning:pandas.core.frame,error::DeprecationWarning,error::PendingDeprecationWarning" \
		poetry run pytest --cov=dnr_synth --cov-report=xml --cov-report=term -q
