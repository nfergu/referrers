"""
Functionality copied from networkx version 3.3. This is primarily to avoid depending on a specific
range of networkx versions, but the override_glyphs option has also been added.

See networkx-license for the networkx license.
"""
from collections import defaultdict
import networkx as nx


class BaseGlyphs:
    @classmethod
    def as_dict(cls):
        return {
            a: getattr(cls, a) for a in dir(cls) if not a.startswith("_") and a != "as_dict"
        }


class AsciiBaseGlyphs(BaseGlyphs):
    empty: str = "+"
    newtree_last: str = "+-- "
    newtree_mid: str = "+-- "
    endof_forest: str = "    "
    within_forest: str = ":   "
    within_tree: str = "|   "


class AsciiDirectedGlyphs(AsciiBaseGlyphs):
    last: str = "L-> "
    mid: str = "|-> "
    backedge: str = "<-"
    vertical_edge: str = "!"


class AsciiUndirectedGlyphs(AsciiBaseGlyphs):
    last: str = "L-- "
    mid: str = "|-- "
    backedge: str = "-"
    vertical_edge: str = "|"


class UtfBaseGlyphs(BaseGlyphs):
    # Notes on available box and arrow characters
    # https://en.wikipedia.org/wiki/Box-drawing_character
    # https://stackoverflow.com/questions/2701192/triangle-arrow
    empty: str = "╙"
    newtree_last: str = "╙── "
    newtree_mid: str = "╟── "
    endof_forest: str = "    "
    within_forest: str = "╎   "
    within_tree: str = "│   "


class UtfDirectedGlyphs(UtfBaseGlyphs):
    last: str = "└─╼ "
    mid: str = "├─╼ "
    backedge: str = "╾"
    vertical_edge: str = "╽"


class UtfUndirectedGlyphs(UtfBaseGlyphs):
    last: str = "└── "
    mid: str = "├── "
    backedge: str = "─"
    vertical_edge: str = "│"


def _find_sources(graph):
    """
    Determine a minimal set of nodes such that the entire graph is reachable
    """
    # For each connected part of the graph, choose at least
    # one node as a starting point, preferably without a parent
    if graph.is_directed():
        # Choose one node from each SCC with minimum in_degree
        sccs = list(nx.strongly_connected_components(graph))
        # condensing the SCCs forms a dag, the nodes in this graph with
        # 0 in-degree correspond to the SCCs from which the minimum set
        # of nodes from which all other nodes can be reached.
        scc_graph = nx.condensation(graph, sccs)
        supernode_to_nodes = {sn: [] for sn in scc_graph.nodes()}
        # Note: the order of mapping differs between pypy and cpython
        # so we have to loop over graph nodes for consistency
        mapping = scc_graph.graph["mapping"]
        for n in graph.nodes:
            sn = mapping[n]
            supernode_to_nodes[sn].append(n)
        sources = []
        for sn in scc_graph.nodes():
            if scc_graph.in_degree[sn] == 0:
                scc = supernode_to_nodes[sn]
                node = min(scc, key=lambda n: graph.in_degree[n])
                sources.append(node)
    else:
        # For undirected graph, the entire graph will be reachable as
        # long as we consider one node from every connected component
        sources = [
            min(cc, key=lambda n: graph.degree[n]) for cc in nx.connected_components(graph)
        ]
        sources = sorted(sources, key=lambda n: graph.degree[n])
    return sources


def generate_network_text(
    graph,
    with_labels=True,
    sources=None,
    max_depth=None,
    ascii_only=False,
    vertical_chains=False,
    override_glyphs=None,
):
    """Generate lines in the "network text" format

    This works via a depth-first traversal of the graph and writing a line for
    each unique node encountered. Non-tree edges are written to the right of
    each node, and connection to a non-tree edge is indicated with an ellipsis.
    This representation works best when the input graph is a forest, but any
    graph can be represented.

    This notation is original to networkx, although it is simple enough that it
    may be known in existing literature. See #5602 for details. The procedure
    is summarized as follows:

    1. Given a set of source nodes (which can be specified, or automatically
    discovered via finding the (strongly) connected components and choosing one
    node with minimum degree from each), we traverse the graph in depth first
    order.

    2. Each reachable node will be printed exactly once on it's own line.

    3. Edges are indicated in one of four ways:

        a. a parent "L-style" connection on the upper left. This corresponds to
        a traversal in the directed DFS tree.

        b. a backref "<-style" connection shown directly on the right. For
        directed graphs, these are drawn for any incoming edges to a node that
        is not a parent edge. For undirected graphs, these are drawn for only
        the non-parent edges that have already been represented (The edges that
        have not been represented will be handled in the recursive case).

        c. a child "L-style" connection on the lower right. Drawing of the
        children are handled recursively.

        d. if ``vertical_chains`` is true, and a parent node only has one child
        a "vertical-style" edge is drawn between them.

    4. The children of each node (wrt the directed DFS tree) are drawn
    underneath and to the right of it. In the case that a child node has already
    been drawn the connection is replaced with an ellipsis ("...") to indicate
    that there is one or more connections represented elsewhere.

    5. If a maximum depth is specified, an edge to nodes past this maximum
    depth will be represented by an ellipsis.

    6. If a node has a truthy "collapse" value, then we do not traverse past
    that node.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent

    with_labels : bool | str
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. If given as a
        string, then that attribute name will be used instead of "label".
        Defaults to True.

    sources : List
        Specifies which nodes to start traversal from. Note: nodes that are not
        reachable from one of these sources may not be shown. If unspecified,
        the minimal set of nodes needed to reach all others will be used.

    max_depth : int | None
        The maximum depth to traverse before stopping. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    vertical_chains : Boolean
        If True, chains of nodes will be drawn vertically when possible.

    override_glyphs : Boolean
        If specified these glyphs nodes will be used instead of the default.

    Yields
    ------
    str : a line of generated text

    Examples
    --------
    >>> graph = nx.path_graph(10)
    >>> graph.add_node("A")
    >>> graph.add_node("B")
    >>> graph.add_node("C")
    >>> graph.add_node("D")
    >>> graph.add_edge(9, "A")
    >>> graph.add_edge(9, "B")
    >>> graph.add_edge(9, "C")
    >>> graph.add_edge("C", "D")
    >>> graph.add_edge("C", "E")
    >>> graph.add_edge("C", "F")
    >>> nx.write_network_text(graph)
    ╙── 0
        └── 1
            └── 2
                └── 3
                    └── 4
                        └── 5
                            └── 6
                                └── 7
                                    └── 8
                                        └── 9
                                            ├── A
                                            ├── B
                                            └── C
                                                ├── D
                                                ├── E
                                                └── F
    >>> nx.write_network_text(graph, vertical_chains=True)
    ╙── 0
        │
        1
        │
        2
        │
        3
        │
        4
        │
        5
        │
        6
        │
        7
        │
        8
        │
        9
        ├── A
        ├── B
        └── C
            ├── D
            ├── E
            └── F
    """
    from typing import Any, NamedTuple

    class StackFrame(NamedTuple):
        parent: Any
        node: Any
        indents: list
        this_islast: bool
        this_vertical: bool

    collapse_attr = "collapse"

    is_directed = graph.is_directed()

    if is_directed:
        if override_glyphs:
            glyphs = override_glyphs
        else:
            glyphs = AsciiDirectedGlyphs if ascii_only else UtfDirectedGlyphs
        succ = graph.succ
        pred = graph.pred
    else:
        if override_glyphs:
            glyphs = override_glyphs
        else:
            glyphs = AsciiUndirectedGlyphs if ascii_only else UtfUndirectedGlyphs
        succ = graph.adj
        pred = graph.adj

    if isinstance(with_labels, str):
        label_attr = with_labels
    elif with_labels:
        label_attr = "label"
    else:
        label_attr = None

    if max_depth == 0:
        yield glyphs.empty + " ..."
    elif len(graph.nodes) == 0:
        yield glyphs.empty
    else:
        # If the nodes to traverse are unspecified, find the minimal set of
        # nodes that will reach the entire graph
        if sources is None:
            sources = _find_sources(graph)

        # Populate the stack with each:
        # 1. parent node in the DFS tree (or None for root nodes),
        # 2. the current node in the DFS tree
        # 2. a list of indentations indicating depth
        # 3. a flag indicating if the node is the final one to be written.
        # Reverse the stack so sources are popped in the correct order.
        last_idx = len(sources) - 1
        stack = [
            StackFrame(None, node, [], (idx == last_idx), False)
            for idx, node in enumerate(sources)
        ][::-1]

        num_skipped_children = defaultdict(lambda: 0)
        seen_nodes = set()
        while stack:
            parent, node, indents, this_islast, this_vertical = stack.pop()

            if node is not Ellipsis:
                skip = node in seen_nodes
                if skip:
                    # Mark that we skipped a parent's child
                    num_skipped_children[parent] += 1

                if this_islast:
                    # If we reached the last child of a parent, and we skipped
                    # any of that parents children, then we should emit an
                    # ellipsis at the end after this.
                    if num_skipped_children[parent] and parent is not None:
                        # Append the ellipsis to be emitted last
                        next_islast = True
                        try_frame = StackFrame(node, Ellipsis, indents, next_islast, False)
                        stack.append(try_frame)

                        # Redo this frame, but not as a last object
                        next_islast = False
                        try_frame = StackFrame(
                            parent, node, indents, next_islast, this_vertical
                        )
                        stack.append(try_frame)
                        continue

                if skip:
                    continue
                seen_nodes.add(node)

            if not indents:
                # Top level items (i.e. trees in the forest) get different
                # glyphs to indicate they are not actually connected
                if this_islast:
                    this_vertical = False
                    this_prefix = indents + [glyphs.newtree_last]
                    next_prefix = indents + [glyphs.endof_forest]
                else:
                    this_prefix = indents + [glyphs.newtree_mid]
                    next_prefix = indents + [glyphs.within_forest]

            else:
                # Non-top-level items
                if this_vertical:
                    this_prefix = indents
                    next_prefix = indents
                else:
                    if this_islast:
                        this_prefix = indents + [glyphs.last]
                        next_prefix = indents + [glyphs.endof_forest]
                    else:
                        this_prefix = indents + [glyphs.mid]
                        next_prefix = indents + [glyphs.within_tree]

            if node is Ellipsis:
                label = " ..."
                suffix = ""
                children = []
            else:
                if label_attr is not None:
                    label = str(graph.nodes[node].get(label_attr, node))
                else:
                    label = str(node)

                # Determine if we want to show the children of this node.
                if collapse_attr is not None:
                    collapse = graph.nodes[node].get(collapse_attr, False)
                else:
                    collapse = False

                # Determine:
                # (1) children to traverse into after showing this node.
                # (2) parents to immediately show to the right of this node.
                if is_directed:
                    # In the directed case we must show every successor node
                    # note: it may be skipped later, but we don't have that
                    # information here.
                    children = list(succ[node])
                    # In the directed case we must show every predecessor
                    # except for parent we directly traversed from.
                    handled_parents = {parent}
                else:
                    # Showing only the unseen children results in a more
                    # concise representation for the undirected case.
                    children = [child for child in succ[node] if child not in seen_nodes]

                    # In the undirected case, parents are also children, so we
                    # only need to immediately show the ones we can no longer
                    # traverse
                    handled_parents = {*children, parent}

                if max_depth is not None and len(indents) == max_depth - 1:
                    # Use ellipsis to indicate we have reached maximum depth
                    if children:
                        children = [Ellipsis]
                    handled_parents = {parent}

                if collapse:
                    # Collapsing a node is the same as reaching maximum depth
                    if children:
                        children = [Ellipsis]
                    handled_parents = {parent}

                # The other parents are other predecessors of this node that
                # are not handled elsewhere.
                other_parents = [p for p in pred[node] if p not in handled_parents]
                if other_parents:
                    if label_attr is not None:
                        other_parents_labels = ", ".join(
                            [str(graph.nodes[p].get(label_attr, p)) for p in other_parents]
                        )
                    else:
                        other_parents_labels = ", ".join([str(p) for p in other_parents])
                    suffix = " ".join(["", glyphs.backedge, other_parents_labels])
                else:
                    suffix = ""

            # Emit the line for this node, this will be called for each node
            # exactly once.
            if this_vertical:
                yield "".join(this_prefix + [glyphs.vertical_edge])

            yield "".join(this_prefix + [label, suffix])

            if vertical_chains:
                if is_directed:
                    num_children = len(set(children))
                else:
                    num_children = len(set(children) - {parent})
                # The next node can be drawn vertically if it is the only
                # remaining child of this node.
                next_is_vertical = num_children == 1
            else:
                next_is_vertical = False

            # Push children on the stack in reverse order so they are popped in
            # the original order.
            for idx, child in enumerate(children[::-1]):
                next_islast = idx == 0
                try_frame = StackFrame(node, child, next_prefix, next_islast, next_is_vertical)
                stack.append(try_frame)
