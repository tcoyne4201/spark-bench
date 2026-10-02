# make fetch HOST=<ip> PORT=<ssh-port>   copy results off the Vast.ai box
# make report                            build report/report.html from fetched/
.PHONY: quick run fetch report
HOST ?=
PORT ?= 22
REMOTE_DIR ?= /root/spark-bench

quick:  ## short shakedown against a running server
	uv run bench.py --quick
run:    ## full benchmark against a running server
	uv run bench.py
fetch:  ## copy tarballs + server log from the remote machine into fetched/
	@test -n "$(HOST)" || (echo "usage: make fetch HOST=<ip> PORT=<ssh-port> [REMOTE_DIR=...]"; exit 1)
	mkdir -p fetched
	rsync -avz --progress -e "ssh -p $(PORT)" \
	  "root@$(HOST):$(REMOTE_DIR)/results/bench-results-*.tar.gz" \
	  "root@$(HOST):$(REMOTE_DIR)/results/server.log" fetched/
report: ## build report/report.html from fetched/*.tar.gz (or results/)
	uv run report.py $(or $(wildcard fetched/*.tar.gz),results/)
