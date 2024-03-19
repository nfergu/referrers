# Referrers

This library helps to answer the question "what is holding a reference to this object?",
which is useful for debugging memory leaks and other issues.

It tries to assign a meaningful name to each reference to an object and returns a graph
of references (including indirect references).

Use the `referrers.get_referrer_graph` function to get a graph of references to a
specific object.

For example, to find references to an instance of `ChildClass`:

```python
import dataclasses
import referrers

class ChildClass:
    pass

@dataclasses.dataclass
class ContainerClass:
    instance_attribute: ChildClass

def my_function():
    local_variable = ContainerClass(ChildClass())
    print(referrers.get_referrer_graph(local_variable.instance_attribute))

my_function()
```

Will output something like:

```plaintext
╙── ChildClass (id=4386629440)
    └─╼ .instance_attribute (instance attribute) (id=4388628032)
        └─╼ ContainerClass (object) (id=4386632272)
            └─╼ my_function.local_variable (local) (id=4386632272)
```

Note: this library is experimental and may not work in all cases. It is also not very
efficient, so should not be used in performance-critical code.

## Installation

```bash
pip install git+https://github.com/nfergu/referrers.git
```

## Integration with memory analysis tools

This library can be used with other memory analysis tools to help identify the source
of memory leaks.

For example, to print the referrers of all lists using
[Pympler](https://pympler.readthedocs.io/en/latest/) (warning: this may produce a lot of output!):

```python
import referrers
from pympler import muppy

all_dicts = [obj for obj in muppy.get_objects() if isinstance(obj, list)]
for obj in all_dicts:
    print(referrers.get_referrer_graph(obj))
```

## Integration with NetworkX

The graph produced by `get_referrer_graph` can be converted to a NetworkX graph using
`referrers.to_networkx_graph`. This can be useful for visualizing the graph, or for
performing more complex analysis.

For example, to visualize a graph of references to an object using Matplotlib and NetworkX:

```python
import matplotlib.pyplot as plt
import networkx as nx
import dataclasses
import referrers

class ChildClass:
    pass

@dataclasses.dataclass
class ContainerClass:
    instance_attribute: ChildClass

def my_function():
    local_variable = ContainerClass(ChildClass())
    graph = referrers.get_referrer_graph(local_variable.instance_attribute)
    nx.draw(
        graph.to_networkx(),
        with_labels=True,
    )
    plt.show()
```

## Untracked Objects

By default, `get_referrer_graph` will only include objects that are tracked by the garbage
collector. However, the `search_for_untracked_objects` flag can be set to `True` to also
include objects that are not tracked by the garbage collector. This option is experimental
and may not work in all cases.

### Known limitations with untracked objects

* The depth of the search for untracked objects is limited by the `max_untracked_search_depth`
  parameter. If this is set too low, some untracked objects may be missing from the graph.
  Try setting this to a higher value if you think this is happening
* Sometimes internal references (from within `referrers`) may be included in the graph when
  finding untracked objects. It should be possible to get rid of these, but I haven't done
  so yet.
* There may be situations where references to untracked objects are not found if there are
  module-level references to them that are also not tracked.
* Finding untracked objects may be slow.

## TODO

* Run MyPy and PyLint and fix any issues
* Add more test cases
* Try to make handling of untracked objects more robust and faster
