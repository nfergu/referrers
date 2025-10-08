# TODO

## General

* Check whether _ReferrerGraphBuilder._is_excluded is correct. One thing it may be excluding
  that we don't want to exclude is references to closure functions.
* Revisit tests in TestReferrerGraph and assert on complete (or larger portion of) graph.
* Look at "module level" tests. Why do they return so much stuff?
* Consider removing functions from exclusions now that untracked object searching is better.
* Add sampling for large graphs.
* Run MyPy and PyLint and fix any issues.
* Add more test cases. Maybe also some integration tests.
