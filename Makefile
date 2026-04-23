.PHONY: setup test test-case

setup:
	pip install -r requirements.txt -r requirements-eval.txt

## Run the full test suite.
test:
	python -m eval.eval_cli run-all

## Run a single case: make test-case CASE=tc01_happy_path_voyager
test-case:
	@if [ -z "$(CASE)" ]; then \
		echo "Usage: make test-case CASE=<case_id>"; exit 1; \
	fi
	python -m eval.eval_cli run $(CASE)
