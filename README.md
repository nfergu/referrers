# Referrers

Referrers is a Python package that helps to answer the question **"what is holding
a reference to this object?"**, which is useful for debugging memory leaks and other
issues. It tries to assign a meaningful name to each reference to an object and
returns a graph of referrers (including indirect referrers).

As a simple example, here is the graph of referrers for an instance of a Python `list`:

```plaintext
╙── list instance (id=4514970240)
    └─╼ ParentClass.member_variable (instance attribute) (id=4513308624)
        └─╼ my_func.local_variable (local) (id=4513308624)
```

In this case the list instance is referenced by a member variable of `ParentClass`, which
is in turn referenced by a local variable in the `my_func` function. For the code to produce
this graph see "Basic Example" below.

Note: this package is experimental and may not work in all cases. It may also be inefficient
in certain cases, and should not be used in performance-critical code.

## Installation

Install using pip:

```bash
pip3 install referrers
```

## Usage

Use the `referrers.get_referrer_graph` function to get a graph of references
to an object.

For example, to print all references to an object referenced by `my_variable`:

```python
import referrers

referrer_graph = referrers.get_referrer_graph(my_variable)
print(referrer_graph)
```

Alternatively, use the `get_referrer_graph_for_list` function to get a single graph
for multiple objects.

## Basic Example

In this example we find all references to a instance of `ChildClass`:

```python
import dataclasses
from typing import List

import referrers

@dataclasses.dataclass
class ParentClass:
    member_variable: List

def my_func():
    local_variable = ParentClass([1, 2])
    print(referrers.get_referrer_graph(local_variable.member_variable))

my_func()
```

This will output something like:

```plaintext
╙── list instance (id=4514970240)
    └─╼ ParentClass.member_variable (instance attribute) (id=4513308624)
        └─╼ my_func.local_variable (local) (id=4513308624)
```

Although the precise output will vary according to the Python version used.

In this case the instance of `ChildClass` that is passed to `referrers.get_referrer_graph`
is referenced directly by the `child_variable` local variable, and also indirectly
via `ContainerClass.instance_attribute`.

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
its `to_networkx` method. This can be useful for visualizing the graph, or for
performing more complex analysis.

The resulting NetworkX graph consists of nodes of type `ReferrerGraphNode`.

For example, to visualize a graph of references to an object using [NetworkX](https://networkx.org/)
and [Matplotlib](https://matplotlib.org/):

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

my_function()
```

## Untracked Objects

By default, `get_referrer_graph` will only include objects that are tracked by the garbage
collector. However, the `search_for_untracked_objects` flag can be set to `True` to also
include objects that are not tracked by the garbage collector. This option is experimental
and may not work well in all cases.

### Known limitations with untracked objects

* The depth of the search for untracked objects is limited by the `max_untracked_search_depth`
  parameter. If this is set too low, some untracked objects may be missing from the graph.
  Try setting this to a higher value if you think this is happening
* Sometimes internal references (from within `referrers`) may be included in the graph when
  finding untracked objects. It should be possible to get rid of these, but I haven't
  managed to track them all down yet.
* Finding untracked objects may be slow.

## Multi-threading

Referrers works well with multiple threads. For example, you can have a separate thread that
prints the referrers of objects that have references in other threads.

In the following example, there is a thread that prints the referrers of instances of `ChildClass`
every second:

```
import dataclasses
import threading
from time import sleep
from pympler import muppy

import referrers

class ChildClass:
    pass

@dataclasses.dataclass
class ContainerClass:
    instance_attribute: ChildClass

def print_referrers():
    while True:
        all_instances = [obj for obj in muppy.get_objects() if isinstance(obj, ChildClass)]
        print(referrers.get_referrer_graph_for_list(all_instances))
        sleep(1)

printing_thread = threading.Thread(target=print_referrers, daemon=True)
printing_thread.start()

def my_function():
    child_variable = ChildClass()
    container_variable = ContainerClass(child_variable)
    sleep(1000)

my_function()
```

## Source

See [https://github.com/nfergu/referrers](https://github.com/nfergu/referrers) for the Github repo.
