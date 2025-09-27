.PHONY: fixtures

fixtures:
	@mkdir -p fixtures/packs docs/examples
	@echo "Generating golden fixture pack (seed=424242)..."
	datagap-synth gen --config config.yaml --out fixtures/packs/golden --seed 424242
	@echo "Generating example report (LLM off)..."
	datagap-report gen --pack fixtures/packs/golden --out docs/examples/golden --llm off
	@echo "Done. Fixtures in fixtures/packs/golden and docs/examples/golden"
