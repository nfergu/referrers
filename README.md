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

* Remove requirement for get_module_prefixes
* Deal with concurrent modification when iterating through collections that could be modified
  concurrently?
* Add test for get_referrer_graph_from_list
* Add readme


Limitations:

* Untracked objects may sometimes be missing referrers. Try increasing `max_untracked_search_depth`
  if this happens.
* Sometimes internal references (from within referrers) may be included in the graph. I don't
  think I've managed to get rid of all of these yet.
* There may be situations where references to untracked objects are not found if there are
  module-level references to them that are also not tracked. But I'm not sure how likely this is
  to happen in practice.

