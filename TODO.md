# TODO

## Problem with circular refs

* Tidy up the build method
* Get rid of tuples from object graph (see test_converging_tree)
* Fix tests
* Finish implementing test_converging_tree
* Add version of test_converging_tree with holder that uses list
* Tidy up prints etc in impl.py

## General

* Revisit tests in TestReferrerGraph and assert on complete (or larger portion of) graph.
* Look at "module level" tests. Why do they return so much stuff?
* Consider removing functions from exclusions now that untracked object searching is better.
* Add sampling for large graphs.
* Run MyPy and PyLint and fix any issues.
* Add more test cases. Maybe also some integration tests.
