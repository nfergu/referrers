# TODO

* Try to simplify searching for untracked objects (maybe by using gc.get_objects() and
  considering the fact that some untracked objects have referrers that are return from
  gc.get_referrers()).
* Consider removing functions from exclusions when untracked object searching is better.
* Add sampling for large graphs.
* Run MyPy and PyLint and fix any issues.
* Add more test cases. Maybe also some integration tests.
