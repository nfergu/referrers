# Referrers

Referrers is a Python package that helps to answer the question **"what is holding
a reference to this object?"**, which is useful for debugging memory leaks and other
issues. It tries to assign a meaningful name to each reference to an object and
returns a graph of referrers (including indirect referrers).

For example, this code:

```python
import referrers

def my_function():
    a = [2, 4]
    d = dict(a=a)
    print(referrers.get_referrer_graph(a))

my_function()
```

Will produce output like:

```plaintext
╙── list instance (id=4346564736)
    ├─╼ dict[a] (id=4347073728)
    │   └─╼ my_function.d (local) (id=4347073728)
    └─╼ my_function.a (local) (id=4346564736)
```

In this case the list instance is referenced directly by a local variable called `a`, and also
via a dictionary, which is in turn referenced by a local variable called `d`.

Note: this package is experimental and may not work in all cases. It may also be inefficient
in certain cases, and should not be used in performance-critical code.

## Installation

Install using pip:

```bash
pip3 install referrers
```

## Usage

Use the `referrers.get_referrer_graph` function to get a graph of referrers for
an object.

For example, get the graph of referrers for the object referenced by `my_variable`:

```python
import referrers

referrer_graph = referrers.get_referrer_graph(my_variable)
print(referrer_graph)
```

Alternatively, use the `get_referrer_graph_for_list` function to get a single graph
for multiple objects.

## Basic Example

In this example we find all referrers for an instance of a `list`:

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

In this case the list instance is referenced by a member variable of `ParentClass`, which
is in turn referenced by a local variable in the `my_func` function.

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

The resulting NetworkX graph consists of nodes of type `ReferrerGraphNode`, with edges
directed from objects to their referrers.

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

By default, `get_referrer_graph` will raise an error if the object passed to it is not
tracked by the garbage collector. In CPython, for example, immutable objects and some
containers that contain only immutable objects (like dicts and tuples) are not tracked
by the garbage collector.

However, the `search_for_untracked_objects` flag can be set to `True` when calling
`get_referrer_graph` to try to find referrers for objects are not tracked by the garbage
collector. This option is experimental and may not work well in all cases.

For example, here we find the referrers of an untracked object (a `dict` containing only
immutable objects):

```python
import dataclasses
import gc
from typing import Dict

import referrers

@dataclasses.dataclass
class ParentClass:
    member_variable: Dict

def my_func():
    local_variable = ParentClass({"a": 1})
    assert not gc.is_tracked(local_variable.member_variable)
    print(referrers.get_referrer_graph(local_variable.member_variable, search_for_untracked_objects=True))

my_func()
```

This will output something like:

```plaintext
╙── dict instance (id=4483928640)
    └─╼ ParentClass.member_variable (instance attribute) (id=4482048576)
        └─╼ my_func.local_variable (local) (id=4482048576)
```

### Known limitations with untracked objects

* The depth of the search for untracked objects is limited by the `max_untracked_search_depth`
  parameter. If this is set too low, some untracked objects may be missing from the graph.
  Try setting this to a higher value if you think this is happening
* Sometimes internal references (from within `referrers`) may be included in the graph when
  finding untracked objects. It should be possible to get rid of these, but I'm not sure if I've
  found them all yet.
* Finding untracked objects may be slow.

## Multi-threading

Referrers works well with multiple threads. For example, you can have a separate thread that
prints the referrers of objects that have references in other threads.

In the following example, there is a thread that prints the referrers of all instances of
`ChildClass` every second (using [Pympler](https://pympler.readthedocs.io/en/latest/) to find
the instances):

```python
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
