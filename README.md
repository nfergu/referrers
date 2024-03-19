# Referrers

This library provides meaningful names for the graph of referrers of an object in Python.

For example:

```python
import dataclasses
import referrers


@dataclasses.dataclass
class A:
    instance_var: str

def f():
    local_var = "Hello World"
    A(local_var)
    print(referrers.get_referrer_graph(local_var))
    # Output:
    # {'local_var': {'f': {'<module>': {}}}}

```

pip install git+https://github.com/nfergu/referrers.git

TODO:

* Write test for module-level untracked objects
* Get rid of get_referrer_graph.target_object (local) when running
  test_untracked_object_within_object and test_untracked_object_within_container.
* Deal with concurrent modification when iterating through collections that could be modified
  concurrently.
* Remove requirement for get_module_prefixes
* Add more tests
* Add readme


Limitations:

* Untracked objects may sometimes be missing referrers. Try increasing `max_untracked_search_depth`
  if this happens.
* Sometimes internal references (from within referrers) may be included in the graph. I don't
  think I've managed to get rid of all of these yet.
  

