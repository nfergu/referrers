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

* Think about what to do about target object being untracked (raise error?)
* Exclude all locals (and locals dict) from the frames within the impl module
* Remove requirement for get_module_prefixes
* Add more tests
* Add readme
