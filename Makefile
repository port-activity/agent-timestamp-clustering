SOURCEDIR = src
SOURCES := $(shell find $(SOURCEDIR) -name '*.py')

install:
	pipenv install --dev

run:
	pipenv run python3 src/lib/SMA/PAA/AGENT/TIMESTAMPCLUSTERING/timestamp_clustering.py

# TODO
test: ;

lint:
	pipenv run pylint $(SOURCES)

#fix:
#	TODO

ci-install-dependencies:
	pip3 install pipenv
	pipenv install --dev

ci: ci-install-dependencies lint test
