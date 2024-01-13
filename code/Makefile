PYTHON ?= python3 # Python 3 executable
PAGERANK_TESTS ?= $(wildcard *pagerank.expect)

all: graph pagerank graph_test pagerank-tests

graph:
	$(PYTHON) graph.py

pagerank:
	$(PYTHON) -m doctest pagerank.py

graph_test:
	$(PYTHON) graph_test.py > graph_test.out
	diff -q graph_test.out graph_test.expect

pagerank-tests: $(PAGERANK_TESTS:%.expect=%.test)

%.test:
	$(PYTHON) pagerank.py $(@:pagerank.test=nodes.csv) $(@:pagerank.test=edges.csv) > $(@:.test=.out)
	diff -q $(@:.test=.out) $(@:.test=.expect)

STYLE_SOURCES ?= $(wildcard *.py)
PYCODESTYLE ?= pycodestyle
PYDOCSTYLE ?= pydocstyle
PYLINT ?= pylint
PYLINT_FLAGS ?=

style: style-pycode style-pydoc style-pylint

style-pycode:
	$(PYCODESTYLE) $(STYLE_SOURCES)

style-pydoc:
	$(PYDOCSTYLE) $(STYLE_SOURCES)

style-pylint:
	$(PYLINT) $(PYLINT_FLAGS) $(STYLE_SOURCES)

clean:
	rm -f *.out
