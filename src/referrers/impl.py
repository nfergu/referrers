import collections
import gc
import inspect
import sys
import traceback
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from types import FrameType
from typing import (
    Any,
    Collection,
    Deque,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
)

import networkx as nx

_TYPE_LOCAL = "local"
_TYPE_GLOBAL = "global"
_TYPE_OBJECT = "object"
_TYPE_MODULE_VARIABLE = "module variable"


@dataclass(frozen=True)
class ReferrerGraphNode:
    """
    Represents a node in a referrer graph.
    """

    name: str
    """
    A meaningful name for the referrer. For example, if the referrer is a local variable,
    the name would be the variable name, suffixed with "(local)".
    """

    id: int
    """
    A unique ID for the referrer object. If the referrer is not an object then this is the
    ID of the object it refers to.
    """

    type: str
    """
    A string representing the type of referrer. For example, if the referrer is a local
    variable, this would be "local".
    """

    is_cycle: bool = False
    """
    Whether the referrer is part of a cycle in the graph. If this is `True`, the referrer
    will be the last node in a branch of the graph.
    """

    def __str__(self):
        return f"{self.name} (id={self.id})" + (
            " (circular ref)" if self.is_cycle else ""
        )

    def __eq__(self, other):
        return (
            isinstance(other, ReferrerGraphNode)
            and self.name == other
            and self.id == other.id
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, self.id))


class ReferrerGraph(ABC):
    """
    Represents a graph of referrers.
    """

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def to_networkx(self) -> nx.DiGraph:
        """
        Converts the object graph to a NetworkX graph. The nodes of the graph are
        `ReferrerGraphNode`s and the edges are directed from the target object to the
        referrer object.
        """
        pass


def get_referrer_graph(
    target_object: Any,
    module_prefixes: Collection[str],
    max_depth: Optional[int] = 10,
    exclude_object_ids: Optional[Sequence[int]] = None,
) -> ReferrerGraph:
    """
    Gets a graph of referrers for the target object.

    To analyze a list of objects, use `get_referrer_graph_for_list` instead.

    :param target_object: The object to analyze.
    :param module_prefixes: The prefixes of the modules to search for module-level variables.
    :param max_depth: The maximum depth to search for referrers. The default is 10. Specify
        `None` to search to unlimited depth (but be careful with this: it may take a long time).
    :param exclude_object_ids: A list of object IDs to exclude from the referrer graph.
    :return: An ObjectGraph containing `ReferrerGraphNode`s, representing the referrers of
        `target_object`.
    """
    exclude_object_ids = exclude_object_ids or []
    exclude_object_ids = list(exclude_object_ids)

    # Always exclude the current frame's locals, as it's not interesting to report references
    # in our internal implementation.
    exclude_object_ids.append(id(locals()))

    return get_referrer_graph_for_list(
        [target_object],
        module_prefixes,
        max_depth=max_depth,
        exclude_object_ids=exclude_object_ids,
    )


def get_referrer_graph_for_list(
    target_objects: List[Any],
    module_prefixes: Collection[str],
    max_depth: Optional[int] = 10,
    exclude_object_ids: Optional[Sequence[int]] = None,
) -> ReferrerGraph:
    """
    Gets a graph of referrers for the list of target objects. All objects in the
    list are analyzed. To analyze a single object, use `get_referrer_graph`.

    The `target_objects` list is excluded from the referrer grap.

    :param target_objects: The objects to analyze. This must be a list.
    :param module_prefixes: The prefixes of the modules to search for module-level variables.
    :param max_depth: The maximum depth to search for referrers. The default is 10. Specify
        `None` to search to unlimited depth (but be careful with this: it may take a long time).
    :param exclude_object_ids: A list of object IDs to exclude from the referrer graph.
    :return: An ObjectGraph containing `ReferrerGraphNode`s, representing the referrers of
        the target objects.
    """
    # We don't allow any iterable, only lists. This is because it's easy to accidentally
    # pass a single big object (like a Pandas dataframe) that is iterable and would be
    # very slow to analyze.
    if not isinstance(target_objects, list):
        raise ValueError("target_objects must be a list")

    exclude_object_ids = exclude_object_ids or []
    exclude_object_ids = list(exclude_object_ids)

    # Always exclude the current frame's locals, as it's not interesting to report references
    # in our internal implementation.
    exclude_object_ids.append(id(locals()))

    # Always exclude all locals and globals dicts, otherwise we get effectively duplicated
    # references to the same objects (once as a local or global, and once as a referrer from
    # the locals or globals dict).
    for thread_frames in _get_frames_for_all_threads().values():
        for frame in thread_frames:
            exclude_object_ids.append(id(frame.f_locals))
            exclude_object_ids.append(id(frame.f_globals))

    builder = _ReferrerGraphBuilder(
        target_objects, module_prefixes, exclude_object_ids=exclude_object_ids
    )
    return builder.build(max_depth=max_depth)


class NameFinder(ABC):
    """
    Gets names for references to an object. This is independent of specific referrers.

    For example, if an object is referenced by a local variable, the name of the local
    variable is returned.
    """

    @abstractmethod
    def get_names(self, target_object: Any) -> Set[str]:
        """
        Returns names for references to `target_object`.

        :param target_object: The object to analyze.
        :return: A set of names.
        """
        pass

    @abstractmethod
    def get_type(self) -> str:
        """
        Returns a name representing the type of references that this class finds.
        For example, if this class finds local variables, this method would return
        "local".
        """
        pass


class LocalVariableNameFinder(NameFinder):
    """
    Gets the names of local variables that refer to the target object, across the
    stack frames of all threads.

    The `get_names` method returns the names of local variables for the
    top-most frame of each thread that contains references to the target object.

    Note that the target object must be in the set of locals *when an instance of this class
    is created*, otherwise it will not be found.
    """

    def __init__(self):
        self._local_var_ids: Set[int] = set()
        for thread_frames in _get_frames_for_all_threads().values():
            for frame in thread_frames:
                for var_value in frame.f_locals.values():
                    self._local_var_ids.add(id(var_value))

    def get_names(self, target_object: Any) -> Set[str]:
        names = set()
        if id(target_object) in self._local_var_ids:
            this_frame = inspect.currentframe()
            for thread_frames in _get_frames_for_all_threads().values():
                for frame in thread_frames:
                    # Exclude the frame of the call to get_names.
                    if frame is not this_frame:
                        frame_names = self._get_frame_names(frame, target_object)
                        # Don't go any further down the stack for this thread if we've found some
                        # names for the object in this frame.
                        if frame_names:
                            names.update(frame_names)
                            break
        return names

    def _get_frame_names(self, frame: FrameType, target_object: Any) -> Set[str]:
        """
        Gets the local_names of the target object in the local variables of the frame.
        :param frame: The frame to search.
        :param target_object: The object to search for.
        :return: A set of local_names of the target object in the frame.
        """
        return {
            f"{frame.f_code.co_name}.{var_name} ({_TYPE_LOCAL})"
            for var_name, var_value in frame.f_locals.items()
            if var_value is target_object
        }

    def get_type(self) -> str:
        return _TYPE_LOCAL


class GlobalVariableNameFinder(NameFinder):
    """
    Gets the names of global variables that refer to the target object, across the
    stack frames of all threads.

    The `get_names` method returns the names of global variables for the
    top-most frame of each thread that contains references to the target object.

    Note that the target object must be in the set of globals *when an instance of this class
    is created*, otherwise it will not be found.
    """

    def __init__(self):
        self._global_var_ids = {global_var.id for global_var in _get_global_vars()}

    def get_names(self, target_object: Any) -> Set[str]:
        names = set()
        if id(target_object) in self._global_var_ids:
            for thread_frames in _get_frames_for_all_threads().values():
                for frame in thread_frames:
                    frame_names = self._get_frame_names(frame, target_object)
                    # Don't go any further down the stack for this thread if we've found some
                    # names for the object in this frame.
                    if frame_names:
                        names.update(frame_names)
                        break
        return names

    def _get_frame_names(self, frame: FrameType, target_object: Any) -> Set[str]:
        """
        Gets the local_names of the target object in the local variables of the frame.
        :param frame: The frame to search.
        :param target_object: The object to search for.
        :return: A set of local_names of the target object in the frame.
        """
        names = set()
        for var_name, var_value in frame.f_globals.items():
            if var_value is target_object:
                module = inspect.getmodule(var_value)
                if module:
                    names.add(f"{module.__name__}.{var_name} ({_TYPE_GLOBAL})")
                else:
                    names.add(f"{var_name} ({_TYPE_GLOBAL})")
        return names

    def get_type(self) -> str:
        return _TYPE_GLOBAL


class ReferrerNameFinder(ABC):
    """
    Gets names for an object's referrer.

    For example, if an object is referenced by an instance attribute, the name of the
    instance attribute is returned.
    """

    @abstractmethod
    def get_names(self, target_object: Any, referrer_object: Any) -> Set[str]:
        """
        Returns names for `referrer_object`, where this references `target_object`.

        :param target_object: The object that is referenced by `referrer_object`.
        :param referrer_object: The object that refers to `target_object`.
        :return: A set of names.
        """
        pass

    @abstractmethod
    def get_type(self) -> str:
        """
        Returns a name representing the type of references that this class finds.
        For example, if this class finds object attributes, this method would return
        "object".
        """
        pass


class ObjectNameFinder(ReferrerNameFinder):
    """
    Gets the names of objects that refer to the target object.
    """

    def __init__(self, exclude_object_ids: Optional[Sequence[Any]] = None):
        self._exclude_object_ids = set(exclude_object_ids or [])

    def get_names(self, target_object: Any, parent_object: Any) -> Set[str]:
        if id(parent_object) in self._exclude_object_ids:
            return set()
        else:
            instance_attribute_names = self._get_instance_attribute_names(
                parent_object, target_object
            )
            # If the parent object contains instance attributes that refer to the target object,
            # return these. Otherwise, return a more general name for the parent object.
            if instance_attribute_names:
                return instance_attribute_names
            else:
                return self._get_container_names(target_object, parent_object)

    def _get_instance_attribute_names(self, parent_object: Any, target_object: Any):
        names = set()
        grandparents = gc.get_referrers(parent_object)
        # If the parent has referrers, we need to check if any of them are classes with
        # instance attributes that refer to the target object (via their __dict__).
        if grandparents:
            for grandparent in grandparents:
                # If the grandparent is a class, check if the parent is the class's dict.
                # If so the grandparent is referring to the target object via an instance
                # attribute. This affects the name that we give the target.
                if (
                    hasattr(grandparent, "__dict__")
                    and grandparent.__dict__ is parent_object
                ):
                    matching_keys = {
                        key
                        for key, value in parent_object.items()
                        if value is target_object
                    }
                    for key in matching_keys:
                        names.add(f".{key} (instance attribute)")
        return names

    def _get_container_names(self, target_object: Any, parent_object: Any) -> Set[str]:
        names = set()
        try:
            if isinstance(
                parent_object, (collections.abc.Mapping, collections.abc.MutableMapping)
            ):
                matching_keys = {
                    key
                    for key, value in parent_object.items()
                    if value is target_object
                }
                for key in matching_keys:
                    names.add(f"{type(parent_object).__name__}[{key}]")
            elif isinstance(
                parent_object,
                (collections.abc.Sequence, collections.abc.MutableSequence),
            ):
                matching_indices = {
                    index
                    for index, value in enumerate(parent_object)
                    if value is target_object
                }
                for index in matching_indices:
                    names.add(f"{type(parent_object).__name__}[{index}]")
        except Exception:
            # This catch-all isn't very nice, but we may get unexpected errors when trying
            # to iterate over containers. For example, they may be modified concurrently,
            # or they may not implement iteration properly (I've seen this in the wild).
            pass
        # If we couldn't find any more specific names, fall back to the parent's type name.
        if not names:
            names.add(f"{type(parent_object).__name__} (object)")
        return names

    def get_type(self) -> str:
        return _TYPE_OBJECT


class ModuleLevelNameFinder(NameFinder):
    """
    Gets all module-level variables that refer to the target object.

    The modules to search are specified when an instance of this class is created. The modules
    must have been imported before the instance is created.

    Global variables are not included in the search. Specifically, if a global variable exists
    with the same name as a module-level variable in the same module, and which refers to the
    target object, the global variable is not included in the results.
    """

    def __init__(self, module_prefix):
        self._modules = [
            module
            for name, module in sys.modules.items()
            if name.startswith(module_prefix)
        ]
        self._global_vars = _get_global_vars()

    def get_names(self, target_object: Any) -> Set[str]:
        names = set()
        for module in self._modules:
            if hasattr(module, "__dict__"):
                for var_name, var_value in module.__dict__.items():
                    if (
                        var_value is target_object
                        and _GlobalVariable(var_name, id(var_value), id(module))
                        not in self._global_vars
                    ):
                        names.add(
                            f"{module.__name__}.{var_name} ({_TYPE_MODULE_VARIABLE})"
                        )
        return names

    def get_type(self) -> str:
        return _TYPE_MODULE_VARIABLE


class _ReferrerGraph(ReferrerGraph):
    def __init__(self, graph: nx.DiGraph):
        self._graph = graph

    def __str__(self):
        return "\n" + "\n".join(line for line in nx.generate_network_text(self._graph))

    def to_networkx(self) -> nx.DiGraph:
        return self._graph


class _ReferrerGraphBuilder:
    """
    Builds a graph of referrers for a set of target objects.
    """

    def __init__(
        self,
        target_objects: Iterable[Any],
        module_prefixes: Collection[str],
        exclude_object_ids: Optional[Sequence[Any]] = None,
    ):
        self._target_objects = target_objects
        # Note: when we create the name finders is important because some implementations
        # start to track the objects that are in the environment when they are created.
        self._name_finders = _get_name_finders(module_prefixes)
        # Exclude the builder and its attributes from the referrer name finders, since we
        # store a reference to the target objects. Also exclude the target objects container.
        self._referrer_name_finders = _get_referrer_name_finders(
            exclude_object_ids=[id(self), id(self.__dict__), id(target_objects)]
            + list(exclude_object_ids)
        )

    def build(self, max_depth: Optional[int]) -> ReferrerGraph:
        graph = nx.DiGraph()

        stack: Deque[Tuple[ReferrerGraphNode, Any, int]] = collections.deque(
            self._get_initial_target_node(target_object)
            for target_object in self._target_objects
        )
        seen_ids = {id(target_object) for target_object in self._target_objects}

        # Do a depth-first search of the object graph, adding nodes and edges to the graph
        # as we go.
        while stack:
            target_graph_node, target_object, depth = stack.pop()

            if max_depth is None or depth < max_depth:
                # For each referrer of the target object, find the name(s) of the referrer and
                # add an edge to the graph for each
                for referrer_object in gc.get_referrers(target_object):
                    referrer_id = id(referrer_object)
                    seen = referrer_id in seen_ids
                    referrer_nodes = self._get_referrer_nodes(
                        target_object=target_object,
                        referrer=referrer_object,
                        seen=seen,
                    )
                    for referrer_graph_node in referrer_nodes:
                        graph.add_edge(target_graph_node, referrer_graph_node)
                        # Avoid an infinite loop by only adding referrers that we haven't seen
                        # before. We still add the relevant edge to the graph so we can see the
                        # relationship though.
                        if not seen:
                            seen_ids.add(referrer_id)
                            stack.append(
                                (referrer_graph_node, referrer_object, depth + 1)
                            )

            # For each non-referrer name pointing to the target object, add an edge to
            # the graph.
            non_referrer_nodes = self._get_non_referrer_nodes(target_object)
            for non_referrer_graph_node in non_referrer_nodes:
                graph.add_edge(target_graph_node, non_referrer_graph_node)

        return _ReferrerGraph(graph)

    def _get_initial_target_node(
        self, target_object: Any
    ) -> Tuple[ReferrerGraphNode, Any, int]:
        name = type(target_object).__name__
        return (
            ReferrerGraphNode(name=name, id=id(target_object), type="object"),
            target_object,
            0,
        )

    def _get_referrer_nodes(
        self, target_object: Any, referrer: Any, seen: bool
    ) -> Set[ReferrerGraphNode]:
        nodes = set()
        for finder in self._referrer_name_finders:
            for name in finder.get_names(target_object, referrer):
                nodes.add(
                    ReferrerGraphNode(
                        name=name,
                        id=id(referrer),
                        type=finder.get_type(),
                        is_cycle=seen,
                    )
                )
        return nodes

    def _get_non_referrer_nodes(self, target_object: Any) -> Set[ReferrerGraphNode]:
        nodes = set()
        for finder in self._name_finders:
            for name in finder.get_names(target_object):
                nodes.add(
                    # We use the target object's ID as the ID for the node, because we don't
                    # have a different unique ID for the reference name. However, this is
                    # fine because both the name and ID (and type) are used to uniquely
                    # identify the node.
                    ReferrerGraphNode(
                        name=name, id=id(target_object), type=finder.get_type()
                    )
                )
        return nodes


@dataclass(frozen=True)
class _GlobalVariable:
    name: str
    id: int
    module_id: Optional[int]


def _get_name_finders(module_prefixes: Collection[str]) -> Sequence[NameFinder]:
    finders = [
        LocalVariableNameFinder(),
        GlobalVariableNameFinder(),
    ]
    for module_prefix in module_prefixes:
        finders.append(ModuleLevelNameFinder(module_prefix))
    return finders


def _get_referrer_name_finders(
    exclude_object_ids: Sequence[int],
) -> Sequence[ReferrerNameFinder]:
    return [ObjectNameFinder(exclude_object_ids=exclude_object_ids)]


def _get_global_vars() -> Set[_GlobalVariable]:
    """
    Gets the names and IDs of all global variables in the current environment.
    :return: A set of `_GlobalVariable`s
    """
    global_vars: Set[_GlobalVariable] = set()
    for thread_frames in _get_frames_for_all_threads().values():
        for frame in thread_frames:
            for var_name, var_value in frame.f_globals.items():
                module = inspect.getmodule(var_value)
                if module:
                    global_vars.add(
                        _GlobalVariable(var_name, id(var_value), id(module))
                    )
                else:
                    global_vars.add(_GlobalVariable(var_name, id(var_value), None))
    return global_vars


def _get_frames_for_all_threads() -> Mapping[str, Iterable[FrameType]]:
    """
    Gets all frames for all threads. The keys are the thread IDs and the values are
    the frames for that thread, starting with the topmost frame.
    """
    return_dict = {}
    # According to the docs the _current_frames function "should be used for internal
    # and specialized purposes only", but what we're doing here seems to be a legitimate.
    for tid, top_frame in sys._current_frames().items():
        return_dict[str(tid)] = reversed(
            [frame for frame, _ in traceback.walk_stack(top_frame)]
        )
    return return_dict
