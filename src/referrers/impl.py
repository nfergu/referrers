import collections
import gc
import inspect
import sys
import traceback
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from itertools import chain
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
    Tuple,
    Dict,
    Callable,
    TypeVar,
    Hashable,
)

import networkx as nx

_PACKAGE_PREFIX = "referrers."

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
    max_depth: Optional[int] = 10,
    exclude_object_ids: Optional[Sequence[int]] = None,
    module_prefixes: Optional[Collection[str]] = None,
    search_for_untracked_objects: bool = False,
    max_untracked_search_depth: int = 10,
) -> ReferrerGraph:
    """
    Gets a graph of referrers for the target object.

    To analyze a list of objects, use `get_referrer_graph_for_list` instead.

    :param target_object: The object to analyze.
    :param max_depth: The maximum depth to search for referrers. The default is 10. Specify
        `None` to search to unlimited depth (but be careful with this: it may take a long time).
    :param exclude_object_ids: A list of object IDs to exclude from the referrer graph.
    :param module_prefixes: The prefixes of the modules to search for module-level variables.
        If this is not specified, the top-level package of the calling code is used.
    :param search_for_untracked_objects: Whether to search for referrers of objects that are
        not tracked by the garbage collector. By default, this is `False` because it can be
        slow to search for untracked objects, and it is not reliable in all cases. If this is
        set to `False` and `target_object` is untracked, an error will be raised. If this is
        set to `True` then `max_untracked_search_depth` is used to control the depth of the
        search.
    :param max_untracked_search_depth: The maximum depth to search for referrers of untracked
        objects. This is the depth that referents will be searched from the roots (locals and
        globals). The default is 10. If you are missing referrers of untracked objects, you
        can increase this value.

    :return: An ObjectGraph containing `ReferrerGraphNode`s, representing the referrers of
        `target_object`.
    """
    exclude_object_ids = exclude_object_ids or []
    exclude_object_ids = list(exclude_object_ids)

    return get_referrer_graph_for_list(
        [target_object],
        max_depth=max_depth,
        exclude_object_ids=exclude_object_ids,
        module_prefixes=module_prefixes,
        search_for_untracked_objects=search_for_untracked_objects,
        max_untracked_search_depth=max_untracked_search_depth,
    )


def get_referrer_graph_for_list(
    target_objects: List[Any],
    max_depth: Optional[int] = 10,
    exclude_object_ids: Optional[Sequence[int]] = None,
    module_prefixes: Optional[Collection[str]] = None,
    search_for_untracked_objects: bool = False,
    max_untracked_search_depth: int = 10,
) -> ReferrerGraph:
    """
    Gets a graph of referrers for the list of target objects. All objects in the
    list are analyzed. To analyze a single object, use `get_referrer_graph`.

    The `target_objects` list is excluded from the referrer grap.

    :param target_objects: The objects to analyze. This must be a list.
    :param max_depth: The maximum depth to search for referrers. The default is 10. Specify
        `None` to search to unlimited depth (but be careful with this: it may take a long time).
    :param exclude_object_ids: A list of object IDs to exclude from the referrer graph.
    :param module_prefixes: The prefixes of the modules to search for module-level variables.
        If this is `None`, the top-level package of the calling code is used.
    :param search_for_untracked_objects: Whether to search for referrers of objects that are
        not tracked by the garbage collector. By default, this is `False` because it can be
        slow to search for untracked objects, and it is not reliable in all cases. If this is
        set to `False` and any of `target_objects` are untracked, an error will be raised. If
        this is set to `True` then `max_untracked_search_depth` is used to control the depth of
        the search.
    :param max_untracked_search_depth: The maximum depth to search for referrers of untracked
        objects. This is the depth that referents will be searched from the roots (locals and
        globals). The default is 10. If you are missing referrers of untracked objects, you
        can increase this value.

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

    # Always exclude all locals and globals dicts, otherwise we get effectively duplicated
    # references to the same objects (once as a local or global, and once as a referrer from
    # the locals or globals dict).
    for thread_frames in _get_frames_for_all_threads().values():
        for frame in thread_frames:
            exclude_object_ids.append(id(frame.f_locals))
            exclude_object_ids.append(id(frame.f_globals))

    if not search_for_untracked_objects:
        untracked_objects = [obj for obj in target_objects if not gc.is_tracked(obj)]
        if untracked_objects:
            raise ValueError(
                "Some target objects are not tracked by the garbage collector. "
                "Set search_for_untracked_objects to True to search for referrers of "
                "these objects, but bear in mind that this can be slow and may not be "
                "reliable in all cases."
            )

    builder = _ReferrerGraphBuilder(
        target_objects,
        module_prefixes,
        max_untracked_search_depth=max_untracked_search_depth,
        exclude_object_ids=exclude_object_ids,
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
            for thread_frames in _get_frames_for_all_threads().values():
                for frame in thread_frames:
                    frame_module = inspect.getmodule(frame)
                    # Exclude all frames from the referrers package.
                    if frame_module is None or (
                        not frame_module.__name__.startswith(_PACKAGE_PREFIX)
                    ):
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
        return _filter_container(
            frame.f_locals,
            extractor_func=lambda x: x.items(),
            filter_func=lambda x: x[1] is target_object,
            selector_func=lambda x: f"{frame.f_code.co_name}.{x[0]} ({_TYPE_LOCAL})",
        )

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
        return _filter_container(
            frame.f_globals,
            extractor_func=lambda x: x.items(),
            filter_func=lambda x: x[1] is target_object,
            selector_func=self._selector_func,
        )

    def _selector_func(self, item: Tuple[str, Any]) -> str:
        var_name, var_value = item
        module = inspect.getmodule(var_value)
        if module:
            return f"{module.__name__}.{var_name} ({_TYPE_GLOBAL})"
        else:
            return f"{var_name} ({_TYPE_GLOBAL})"

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

    def __init__(self, excluded_id_set: Optional[Set[int]] = None):
        self._excluded_id_set = excluded_id_set or set()

    def get_names(self, target_object: Any, parent_object: Any) -> Set[str]:
        if id(parent_object) in self._excluded_id_set:
            return set()
        else:
            instance_attribute_names = self._get_instance_attribute_names(
                target_object, parent_object
            )
            # If the parent object contains instance attributes that refer to the target object,
            # return these. Otherwise, return a more general name for the parent object.
            if instance_attribute_names:
                return instance_attribute_names
            else:
                return self._get_container_names(target_object, parent_object)

    def _get_instance_attribute_names(self, target_object: Any, parent_object: Any):
        names = set()
        # The behaviour here is different between Python versions <=3.10, and > 3.10.
        # In versions <= 3.10, the parent of an object is a dict, which matches the
        # __dict__ attribute of the grandparent object (which is the actual referring object).
        # In versions > 3.10, the parent of an object is the referring object itself.
        if hasattr(parent_object, "__dict__"):
            # This is the logic for Python > 3.10
            matching_keys = {
                key
                for key, value in parent_object.__dict__.items()
                if value is target_object
            }
            for key in matching_keys:
                names.add(f"{type(parent_object).__name__}.{key} (instance attribute)")
        else:
            # This is the logic for Python <= 3.10
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
                            names.add(
                                f"{type(grandparent).__name__}.{key} (instance attribute)"
                            )
        return names

    def _get_container_names(self, target_object: Any, parent_object: Any) -> Set[str]:
        names = set()
        try:
            if isinstance(
                parent_object, (collections.abc.Mapping, collections.abc.MutableMapping)
            ):
                names.update(
                    _filter_container(
                        parent_object,
                        extractor_func=lambda x: x.items(),
                        filter_func=lambda x: x[1] is target_object,
                        selector_func=lambda x: f"{type(parent_object).__name__}[{x[0]}]",
                    )
                )
            elif isinstance(
                parent_object,
                (collections.abc.Sequence, collections.abc.MutableSequence),
            ):
                names.update(
                    _filter_container(
                        parent_object,
                        extractor_func=lambda x: enumerate(x),
                        filter_func=lambda x: x[1] is target_object,
                        selector_func=lambda x: f"{type(parent_object).__name__}[{x[0]}]",
                    )
                )
        except Exception:
            # Certain containers don't support iteration. We can't do anything about that,
            # so we just fall back to the more general name for the parent object.
            # The catch-all exception isn't ideal, but we don't know what exceptions
            # the container types might raise.
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

    def __init__(self, module_prefix: str):
        """
        Initializes the name finder.

        :param module_prefix: The prefix to use when searching for modules. If `None`, the
            prefix is determined by the top-level package of the first frame in the call stack
            that is not part of the `referrers` package.
        """
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
        module_prefixes: Optional[Collection[str]],
        max_untracked_search_depth: int,
        exclude_object_ids: Optional[Sequence[int]] = None,
    ):
        if not module_prefixes:
            stack_frames = inspect.stack()
            for frame_info in stack_frames:
                frame_module = inspect.getmodule(frame_info.frame).__name__
                if not frame_module.startswith(_PACKAGE_PREFIX):
                    # Use the top-level package of the calling code as the module prefix
                    # (with a trailing dot). For example, if the calling code is in a module
                    # called my_module.do_thing, the module prefix would be "my_module.".
                    module_prefixes = [f"{frame_module.split('.')[0]}."]
                    break
        if not module_prefixes:
            raise ValueError(
                "Could not determine the top-level package of the calling code. "
                "Please specify the module_prefixes parameter to set this explicitly."
            )
        excluded_id_set = {id(self), id(self.__dict__), id(target_objects)} | set(
            exclude_object_ids
        )
        self._target_objects = target_objects
        (
            self._untracked_objects_referrers,
            extra_exclusions,
        ) = self._get_untracked_object_referrers(
            target_objects,
            excluded_id_set=excluded_id_set,
            max_depth=max_untracked_search_depth,
            module_prefixes=module_prefixes,
        )
        # Note: when we create the name finders is important because some implementations
        # start to track the objects that are in the environment when they are created.
        self._name_finders = _get_name_finders(module_prefixes)
        # Exclude the builder and its attributes from the referrer name finders, since we
        # store a reference to the target objects. Also exclude the target objects container.
        self._referrer_name_finders = _get_referrer_name_finders(
            excluded_id_set | extra_exclusions
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
                for referrer_object in self._get_referrers(target_object):
                    if not self._is_excluded(referrer_object):
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

    def _is_excluded(self, obj: Any) -> bool:
        # We exclude these objects because anything referenced by them should be picked-up
        # elsewhere (locals global etc), and excluding them speeds things up a lot.
        return inspect.isframe(obj) or inspect.isroutine(obj) or inspect.ismodule(obj)

    def _get_referrers(self, target_object: Any) -> Iterable[Any]:
        if gc.is_tracked(target_object):
            return gc.get_referrers(target_object)
        else:
            return self._untracked_objects_referrers.get(id(target_object), [])

    def _get_initial_target_node(
        self, target_object: Any
    ) -> Tuple[ReferrerGraphNode, Any, int]:
        name = f"{type(target_object).__name__} instance"
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

    def _get_untracked_object_referrers(
        self,
        target_objects: Iterable[Any],
        excluded_id_set: Set[int],
        max_depth: int,
        module_prefixes: Collection[str],
    ) -> Tuple[Mapping[id, List[Any]], Set[int]]:
        """
        Builds a mapping of object IDs to referrers for objects that are not tracked by the
        garbage collector, and returns this along with extra IDs to exclude.
        """
        return_dict: Dict[int, List[Any]] = collections.defaultdict(list)

        extra_exclusions = set()
        excluded_id_set.add(id(return_dict))

        do_not_visit = copy(excluded_id_set)
        do_not_visit.add(id(return_dict))

        untracked_target_object_ids = {
            id(obj) for obj in target_objects if not gc.is_tracked(obj)
        }

        # Look through the referents of all locals and globals to try and find
        # untracked objects.
        if len(untracked_target_object_ids) > 0:

            roots = self._get_roots_from_locals_globals_and_modules(
                do_not_visit=do_not_visit,
                module_prefixes=module_prefixes,
            )
            # Make sure we don't visit the roots list, or very strange things will happen!
            do_not_visit.add(id(roots))
            # Also add the roots to the excluded set. It's not clear why this is necessary,
            # but it seems to be.
            extra_exclusions.add(id(roots))

            for root in roots:
                untracked_stack = collections.deque()
                do_not_visit.add(id(untracked_stack))
                extra_exclusions.add(id(untracked_stack))
                self._populate_untracked_object_referrers(
                    obj=root,
                    do_not_visit=do_not_visit,
                    untracked_object_referrers=return_dict,
                    untracked_target_object_ids=untracked_target_object_ids,
                    untracked_stack=untracked_stack,
                    depth=0,
                    max_depth=max_depth,
                )

        for value in return_dict.values():
            extra_exclusions.add(id(value))

        return return_dict, extra_exclusions

    def _populate_untracked_object_referrers(
        self,
        obj: Any,
        do_not_visit: Set[int],
        untracked_object_referrers: Dict[int, List],
        untracked_target_object_ids: Set[int],
        untracked_stack: Deque[Any],
        depth: int,
        max_depth: int,
    ):
        # This method is pretty horrible. It can maybe be made less complex.
        obj_id = id(obj)
        if inspect.isframe(obj) or obj_id in do_not_visit or depth >= max_depth:
            return
        else:
            do_not_visit.add(obj_id)
            for referent in gc.get_referents(obj):
                referent_id = id(referent)
                is_tracked = gc.is_tracked(referent)
                pop = False
                # Push untracked objects or objects that are referred to by untracked objects
                # onto the stack (though I'm not sure if this is possible).
                if (not is_tracked) or len(untracked_stack) > 0:
                    untracked_stack.append(obj)
                    pop = True
                # If we find one of the target objects, add the current object to the
                # referrers list for the target object. We also walk back up the untracked
                # object stack and add any other untracked objects that refer indirectly
                # to the target object.
                if referent_id in untracked_target_object_ids:
                    id_to_add = referent_id
                    for untracked_obj in reversed(untracked_stack):
                        untracked_object_referrers[id_to_add].append(untracked_obj)
                        id_to_add = id(untracked_obj)
                # Recurse into the referent. The exit condition is when we find a referent
                # that is in the do_not_visit set, or when we reach the maximum depth.
                self._populate_untracked_object_referrers(
                    obj=referent,
                    do_not_visit=do_not_visit,
                    untracked_object_referrers=untracked_object_referrers,
                    untracked_target_object_ids=untracked_target_object_ids,
                    untracked_stack=untracked_stack,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                if pop:
                    untracked_stack.pop()

    def _get_roots_from_locals_globals_and_modules(
        self, do_not_visit: Set[int], module_prefixes: Optional[Collection[str]] = None
    ) -> List[Any]:
        roots = []
        for thread_frames in _get_frames_for_all_threads().values():
            for frame in thread_frames:
                # Exclude the locals and globals themselves from the search.
                do_not_visit.add(id(frame.f_locals))
                do_not_visit.add(id(frame.f_globals))
                for var_value in chain(
                    frame.f_locals.values(), frame.f_globals.values()
                ):
                    roots.append(var_value)
        self._modules = [
            module
            for name, module in sys.modules.items()
            if self._matches_prefixes(name, module_prefixes)
        ]
        self._global_vars = _get_global_vars()
        for module in self._modules:
            if hasattr(module, "__dict__"):
                for var_name, var_value in module.__dict__.items():
                    if (
                        _GlobalVariable(var_name, id(var_value), id(module))
                        not in self._global_vars
                    ):
                        roots.append(var_value)
        return roots

    def _matches_prefixes(self, module_name: str, module_prefixes: Collection[str]):
        return any(module_name.startswith(prefix) for prefix in module_prefixes)


@dataclass(frozen=True)
class _GlobalVariable:
    name: str
    id: int
    module_id: Optional[int]


def _get_name_finders(
    module_prefixes: Collection[str],
) -> Sequence[NameFinder]:
    finders = [
        LocalVariableNameFinder(),
        GlobalVariableNameFinder(),
    ]
    for module_prefix in module_prefixes:
        finders.append(ModuleLevelNameFinder(module_prefix))
    return finders


def _get_referrer_name_finders(
    excluded_id_set: Set[int],
) -> Sequence[ReferrerNameFinder]:
    return [ObjectNameFinder(excluded_id_set=excluded_id_set)]


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


_T = TypeVar("_T")
_V = TypeVar("_V", bound=Hashable)


def _filter_container(
    container: Any,
    extractor_func: Callable[[Any], Iterable[_T]],
    filter_func: Callable[[_T], bool],
    selector_func: Callable,
) -> Set[_V]:
    """
    Filters a container using the given functions.

    This function is robust to containers being modified while we're iterating over them.
    If a RuntimeError is raised, we make a copy of the container and try again.

    :param container: The container to filter.
    :param extractor_func: A function that creates an iterable from the container
    :param filter_func: A function that filters items from the iterable
    :param selector_func: A function that selects items from the iterable
    :return: A set of selected items.
    """
    try:
        return {
            selector_func(item)
            for item in extractor_func(container)
            if filter_func(item)
        }
    except RuntimeError:
        # If we get a RuntimeError, it's likely that the iterable is being modified while
        # we're iterating over it. In this case we make a copy of the container and try again.
        return {
            selector_func(item)
            for item in extractor_func(copy(container))
            if filter_func(item)
        }
