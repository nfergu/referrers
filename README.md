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
╙── list (object) (id=4343582848) (target)
    ├── my_function.a (local) (id=4343582848) (root)
    └── dict key=a (id=4345092160)
        └── dict (object) (id=4345092160)
            └── my_function.d (local) (id=4345092160) (root)
```

In this case the list instance is referenced directly by a local variable called `a`, and also
via an entry in a dictionary, which is in turn referenced by a local variable called `d`.

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
╙── list (object) (id=4360191168) (target)
    └── ParentClass.member_variable (instance attribute) (id=4344740512)
        └── ParentClass (object) (id=4344740512)
            └── my_func.local_variable (local) (id=4344740512) (root)
```

In this case the list instance is referenced by a member variable of `ParentClass`, which
is in turn referenced by a local variable in the `my_func` function.

## Integration with memory analysis tools

This library can be used with other memory analysis tools to help identify the source
of memory leaks.

For example, to print the referrers of the top 10 largest objects using 
[Pympler](https://pympler.readthedocs.io/en/latest/):

```python
from pympler import muppy
import referrers

def main():
    top_10_objects = (muppy.sort(muppy.get_objects()))[-10:]
    top_10_objects.reverse()
    
    for obj in top_10_objects:
        print(
            referrers.get_referrer_graph(
                obj,
                exclude_object_ids=[id(top_10_objects)],
            )
        )
        
if __name__ == "__main__":
    main()
```

Note that this example  `exclude_object_ids` to exclude the `top_10_objects`
variable from the graph.

Here's how to find the referrers of all objects that have been created
between two points in time:

```python
import referrers
from pympler import summary, muppy

def main():
    o1 = muppy.get_objects()
    my_dict = {'a': [1]}
    o2 = muppy.get_objects()
    
    o1_ids = {id(obj) for obj in o1}
    o2_ids = {id(obj): obj for obj in o2}
    diff = [obj for obj_id, obj in o2_ids.items() if obj_id not in o1_ids]
    
    summary.print_(summary.get_diff(summary.summarize(o1), summary.summarize(o2)))
    
    for obj in diff:
        print(
            referrers.get_referrer_graph(
                obj,
                exclude_object_ids=[id(o1), id(o2), id(diff), id(o2_ids)],
            )
        )

if __name__ == '__main__':
    main()

```

This will print a summary of the objects that have been created between o1 and o2,
as well as the variable names that reference these objects. This can be useful for finding
memory leaks.

## Performance

Finding referrers can be slow for objects with many referrers. Use the following options
to control the performance of the search:

- `timeout`: The maximum time to spend searching for referrers. If this time is exceeded,
  a partial graph is returned. Note that this timeout is approximate, and may not be
  effective if the search is blocked by a long-running operation. The default is `None`
  which means no timeout. If the timeout is exceeded, the graph will contain a node
  containing the text "Timeout of N seconds exceeded".
- `max_depth`: The maximum depth to search for referrers. The default is 20. Specify
  `None` to search to unlimited depth (but be careful with this: it may take a long time).
  If the maximum depth is exceeded, the graph will contain a node containing the text
  "Maximum depth of N exceeded".
- `max_untracked_search_depth`: The maximum depth to search for referrers of untracked
  objects. This is the depth that referents will be searched from the roots (locals and
  globals). The default is 30. If you are missing referrers of untracked objects, you
  can increase this value.
- `single_object_referrer_limit`: The maximum number of referrers to include in the graph
  for an individual object instance. If the limit is exceeded, the graph will contain a
  node containing the text "Referrer limit of N exceeded". Note that this limit is
  approximate and does not apply to all referrer types. Specifically, it only applies to
  object references. Additionally, this limit does not apply to immortal objects.

## Integration with NetworkX

The graph produced by `get_referrer_graph` can be converted to a NetworkX graph using
its `to_networkx` method. This can be useful for visualizing the graph, or for
performing more complex analysis.

The resulting NetworkX graph consists of nodes of type `ReferrerGraphNode`, with edges
directed from referrers to their referents.

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

def node_label(node: referrers.ReferrerGraphNode) -> str:
    return f"{node.name.replace(" ", "\n", 1)}"

def my_function():
    local_variable = ContainerClass(ChildClass())
    other_variable = ContainerClass(local_variable.instance_attribute)
    graph = referrers.get_referrer_graph(local_variable.instance_attribute)
    
    nx_graph = graph.to_networkx()
    nx.draw_networkx(
        nx_graph,
        pos=nx.spectral_layout(nx_graph),
        with_labels=True,
        font_size=8,
        labels={node: node_label(node) for node in nx_graph.nodes},
        node_color="lightblue",
        node_size=1400,
    )
    plt.margins(x=0.2)
    plt.show()

my_function()
```

This will produce an image like this:

<img width="740" height="438" alt="image" src="https://github.com/user-attachments/assets/e1fd9220-e2c9-4a33-a508-3cc5da1876ab" />




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

## Untracked Objects

This library will try to find referrers for objects that are not tracked by the garbage
collector. For example, mutable objects, and collections containing only immutable objects in
CPython. However, this may be slower than for tracked objects, and is limited by the
`max_untracked_search_depth` parameter. Try setting this to a higher value if you think there
are referrers missing from the graph.

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
    print(referrers.get_referrer_graph(local_variable.member_variable))

my_func()
```

This will output something like:

```plaintext
╙── dict (object) (id=4374433472) (target)
    └── ParentClass.member_variable (instance attribute) (id=4381605040)
        └── ParentClass (object) (id=4381605040)
            └── my_func.local_variable (local) (id=4381605040) (root)
```

## Source

See [https://github.com/nfergu/referrers](https://github.com/nfergu/referrers) for the Github repo.
