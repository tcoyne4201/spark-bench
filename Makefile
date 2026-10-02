.PHONY: quick run report
quick:  ## short shakedown against a running server
	uv run bench.py --quick
run:    ## full benchmark against a running server
	uv run bench.py
report: ## build report/ from results/
	uv run report.py results/
