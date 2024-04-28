# TODO

* Should GC on each call to referrers?
* Make searching for untracked objects on by default (or on always?)
* Revisit tests in TestReferrerGraph and assert on complete (or larger portion of) graph.
* Revisit "module level" tests. Why do they return so much stuff?
* Consider removing functions from exclusions when untracked object searching is better.
* Add sampling for large graphs.
* Run MyPy and PyLint and fix any issues.
* Add more test cases. Maybe also some integration tests.
