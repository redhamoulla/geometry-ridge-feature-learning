.PHONY: install test quick public paper

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest

quick:
	python run_experiments.py --suite quick

public:
	python run_experiments.py --suite public

paper:
	$(MAKE) -C paper/src
