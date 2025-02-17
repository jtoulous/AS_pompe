RED := \033[31m
GREEN := \033[32m
RESET := \033[0m

all: venv install

venv:
	@echo "$(GREEN) ===> creating virtual environnement..."
	@python -m venv virtualEnv
	@echo " ===> Done.$(RESET)"

install:
	@echo "$(GREEN) ===> installing dependencies..."
	@. virtualEnv/bin/activate && pip install -r requirements.txt
#	@. virtualEnv/bin/activate && pip install -e .
#	@rm -rf core/utils.egg-info
	@echo " ===> Done.$(RESET)"

reset:
	@echo "$(GREEN) ===> resetting models..."
	@rm -rf core/models/*
#	@rm -rf core/data/agents/scalers/*.pkl
	@echo " ===> Done.$(RESET)"

clean:
	@echo "$(GREEN) ===> removing virtual environnement..."
	@rm -rf virtualEnv
	@echo " ===> Done."

fclean: reset clean

.PHONY: all venv install clean fclean