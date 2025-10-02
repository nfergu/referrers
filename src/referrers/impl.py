import collections
import enum
import gc
import inspect
import logging
import sys
import traceback
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from itertools import chain
from timeit import default_timer
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

from referrers import networkx_copy

IMMORTAL_OBJECT_REFCOUNT = 1000000000

_PACKAGE_PREFIX = "referrers."

_TYPE_LOCAL = "local"
_TYPE_CLOSURE = "closure"
_TYPE_GLOBAL = "global"
_TYPE_OBJECT = "object"
_TYPE_MODULE_VARIABLE = "module variable"

_MAX_MAPPING_KEY_LENGTH = 50

logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferrerGraphNode:
    """
    Represents a node in a referrer graph.

    Nodes are uniquely identified by their name, id, and type (*not* by the id alone).
    """

    name: str
    """
    A meaningful name for the referrer. For example, if the referrer is a local variable,
    the name would be the variable name, suffixed with "(local)".
    """

    id: int
    """
    The ID of the referrer object. If the referrer is not an object then this is the
    ID of the object it refers to.
    
    Note: multiple nodes in the graph may have the same ID. For example, an object's instance
    attribute has the same ID as the object itself, and a local variable that refers to an object
    has the same ID as the object itself.
    """

    type: str
    """
    A string representing the type of referrer. For example, if the referrer is a local
    variable, this would be "local".
    """

    is_target: bool = False
    """
    Whether this node is a target (the object for which we are trying to find referrers).
    This attribute is not include in equality comparisons and the hash.
    """

    def __str__(self):
        return f"{self.name} (id={self.id})" + (" (target)" if self.is_target else "")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ReferrerGraphNode):
            return False
        return self.name == other.name and self.id == other.id and self.type == other.type

    def __hash__(self) -> int:
        return hash((self.name, self.id, self.type))


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
    max_depth: Optional[int] = 20,
    exclude_object_ids: Optional[Sequence[int]] = None,
    module_prefixes: Optional[Collection[str]] = None,
    max_untracked_search_depth: int = 30,
    timeout: Optional[float] = None,
    single_object_referrer_limit: Optional[int] = 100,
) -> ReferrerGraph:
    """
    Gets a graph of referrers for the target object.

    To analyze a list of objects, use `get_referrer_graph_for_list` instead.

    Note: This function forces a garbage collection.

    :param target_object: The object to analyze.
    :param max_depth: The maximum depth to search for referrers. The default is 10. Specify
        `None` to search to unlimited depth (but be careful with this: it may take a long time).
    :param exclude_object_ids: A list of object IDs to exclude from the referrer graph.
    :param module_prefixes: The prefixes of the modules to search for module-level variables.
        If this is not specified, the top-level package of the calling code is used.
    :param max_untracked_search_depth: The maximum depth to search for referrers of untracked
        objects. This is the depth that referents will be searched from the roots (locals and
        globals). The default is 30. If you are missing referrers of untracked objects, you
        can increase this value.
    :param timeout: The maximum time to spend searching for referrers. If this time is exceeded,
        a partial graph is returned. Note that this timeout is approximate, and may not be
        effective if the search is blocked by a long-running operation. The default is `None`
        which means no timeout.
    :param single_object_referrer_limit: The maximum number of referrers to include in the graph
        for an individual object instance. If the limit is exceeded, the graph will contain a
        node containing the text "Referrer limit of N exceeded". Note that this limit is
        approximate and does not apply to all referrers types. Specifically, it only applies to
        object references. Additionally, this limit does not apply to immortal objects.

    :return: An ObjectGraph containing `ReferrerGraphNode`s, representing the referrers of
        `target_object`.
    """
    gc.collect()

    exclude_object_ids = exclude_object_ids or []
    exclude_object_ids = list(exclude_object_ids)

    return get_referrer_graph_for_list(
        [target_object],
        max_depth=max_depth,
        exclude_object_ids=exclude_object_ids,
        module_prefixes=module_prefixes,
        max_untracked_search_depth=max_untracked_search_depth,
        timeout=timeout,
        single_object_referrer_limit=single_object_referrer_limit,
    )


def get_referrer_graph_for_list(
    target_objects: List[Any],
    max_depth: Optional[int] = 20,
    exclude_object_ids: Optional[Sequence[int]] = None,
    module_prefixes: Optional[Collection[str]] = None,
    max_untracked_search_depth: int = 30,
    timeout: Optional[float] = None,
    single_object_referrer_limit: Optional[int] = 100,
) -> ReferrerGraph:
    """
    Gets a graph of referrers for the list of target objects. All objects in the
    list are analyzed. To analyze a single object, use `get_referrer_graph`.

    The `target_objects` list is excluded from the referrer graph.

    Note: This function forces a garbage collection.

    :param target_objects: The objects to analyze. This must be a list.
    :param max_depth: The maximum depth to search for referrers. The default is 10. Specify
        `None` to search to unlimited depth (but be careful with this: it may take a long time).
    :param exclude_object_ids: A list of object IDs to exclude from the referrer graph.
    :param module_prefixes: The prefixes of the modules to search for module-level variables.
        If this is `None`, the top-level package of the calling code is used.
    :param max_untracked_search_depth: The maximum depth to search for referrers of untracked
        objects. This is the depth that referents will be searched from the roots (locals and
        globals). The default is 30. If you are missing referrers of untracked objects, you
        can increase this value.
    :param timeout: The maximum time to spend searching for referrers. If this time is exceeded,
        a partial graph is returned. Note that this timeout is approximate, and may not be
        effective if the search is blocked by a long-running operation. The default is `None`
        which means no timeout.
    :param single_object_referrer_limit: The maximum number of referrers to include in the graph
        for an individual object instance. If the limit is exceeded, the graph will contain a
        node containing the text "Referrer limit of N excedded". Note that this limit is
        approximate and does not apply to all referrers types. Specifically, it only applies to
        object references. Additionally, this limit does not apply to immortal objects.

    :return: An ObjectGraph containing `ReferrerGraphNode`s, representing the referrers of
        the target objects.
    """
    # Garbage collect before we start, otherwise repeated calls to this function may identify
    # referrers that are internal to this module.
    gc.collect()

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

    builder = _ReferrerGraphBuilder(
        target_objects,
        module_prefixes,
        max_untracked_search_depth=max_untracked_search_depth,
        exclude_object_ids=exclude_object_ids,
        single_object_referrer_limit=single_object_referrer_limit,
    )
    return builder.build(
        max_depth=max_depth,
        timeout=timeout,
    )


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


class _InternalReferrer(ABC):
    """
    Interface for referrers that we have created internally (normally to wrap
    other objects.

    These objects have a special string representation.
    """

    @abstractmethod
    def unpack(self):
        """
        Unpacks the referrer to the object that it wraps.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


@dataclass(frozen=True)
class _ClosureDetails(_InternalReferrer):
    function: Any
    variable_name: str

    def unpack(self):
        return self.function

    def __str__(self):
        return f"{self.function.__qualname__}.{self.variable_name} ({_TYPE_CLOSURE})"


@dataclass(frozen=True)
class _ReferrerLimitReached(_InternalReferrer):
    num: int
    limit: Optional[int]

    def unpack(self):
        return str(self)

    def __str__(self):
        return f"Referrer limit of {self.limit} exceeded with {self.num} referrers."


@dataclass(frozen=True)
class _DepthLimitReached(_InternalReferrer):
    limit: Optional[int]

    def unpack(self):
        return str(self)

    def __str__(self):
        return f"Maximum depth of {self.limit} exceeded"


@dataclass(frozen=True)
class _Timeout(_InternalReferrer):
    time_taken: float
    timeout: Optional[float]

    def unpack(self):
        return str(self)

    def __str__(self):
        return (
            f"Timeout of {self.timeout:.2f} seconds exceeded "
            f"(after {self.time_taken:.2f} seconds)"
        )


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

    def _get_frame_names(self, frame: FrameType, target_object: Any) -> Iterable[str]:
        """
        Gets the local_names of the target object in the local variables of the frame.
        :param frame: The frame to search.
        :param target_object: The object to search for.
        :return: The local names of the target object in the frame.
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

    def _get_frame_names(self, frame: FrameType, target_object: Any) -> Iterable[str]:
        """
        Gets the local_names of the target object in the local variables of the frame.
        :param frame: The frame to search.
        :param target_object: The object to search for.
        :return: The global names of the target object in the frame.
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


class ReferrerNameType(enum.Enum):
    """
    The type of a referrer name.
    """

    INTERNAL = "INTERNAL"
    """
    An internal referrer name (not a real one from the object graph).
    """

    OBJECT = "OBJECT"
    """
    An object. This generally means that we couldn't find a more specific container and so
    we couldn't determine a more specific referrer name.
    """

    COLLECTION_MEMBER = "COLLECTION"
    """
    A member of a collection (dict, list etc).
    """

    INSTANCE_ATTRIBUTE_IN_OBJECT = "INSTANCE_ATTRIBUTE_IN_OBJECT"
    """
    An instance attribute within an object. This is the way that Python >= 3.11 sometimes
    presents instance attributes.  
    """

    INSTANCE_ATTRIBUTE_IN_DICT = "INSTANCE_ATTRIBUTE_IN_DICT"
    """
    An instance attribute within a dict that is referred to by an object. This is the way that 
    Python < 3.11 always presents instance attributes.
    """

    UNKNOWN = "UNKNOWN"
    """
    An unknown type.
    """


@dataclass
class ReferrerName:
    """
    Represents a referrer name, along with some additional information.

    Note: this class is deliberately not frozen (and therefore not hashable) because
    it contains a reference to the referrer object which we don't want to try and
    hash.
    """

    name: str
    """
    A friendly name for the referrer.
    """

    is_container: bool
    """
    Whether the referrer is a container. Containers include everything inheriting from the
    Python Container class, and also all objects that contain a referent as an instance
    attribute.
    
    Containers can be treated specially because there are potentially multiple interesting
    things about a collection: the collection itself, and an object's membership of the
    collection.
    """

    referrer: Any
    """
    The referrer object. This isn't necessarily the parent of the target object, since
    in some cases targets are referenced indirectly by their grandparents for example.
    This is the case for some instance attributes, which are referenced indirectly via dicts.
    """


class ReferrerNameFinder(ABC):
    """
    Gets names for an object's referrer.

    For example, if an object is referenced by an instance attribute, the name of the
    instance attribute is returned.
    """

    @abstractmethod
    def get_names(self, target_object: Any, referrer_object: Any) -> List[ReferrerName]:
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

    def __init__(
        self,
        single_object_referrer_limit: Optional[float],
        excluded_id_set: Optional[Set[int]] = None,
    ):
        self._single_object_referrer_limit = single_object_referrer_limit
        self._excluded_id_set = excluded_id_set or set()

    def get_names(self, target_object: Any, parent_object: Any) -> List[ReferrerName]:
        if id(parent_object) in self._excluded_id_set:
            return []
        else:
            # Deal with internal objects as a special case. These generally have a nice
            # string representation that we can use.
            if isinstance(parent_object, _InternalReferrer):
                return [
                    ReferrerName(
                        name=str(parent_object),
                        is_container=False,
                        referrer=parent_object,
                    )
                ]
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

    def _get_instance_attribute_names(
        self, target_object: Any, parent_object: Any
    ) -> List[ReferrerName]:
        names: List[ReferrerName] = []
        # The behaviour here is different between Python versions <=3.10, and > 3.10,
        # and also between multiple calls to the referrers of an object's instance
        # attributes within a single Python version.
        # Sometimes the parent of an object is a dict, which matches the
        # __dict__ attribute of the grandparent object (which is the actual referring object).
        # Sometimes the parent of an object is the referring object itself.
        if _safe_hasattr(parent_object, "__dict__"):
            # This is the logic there the parent of an object is the referring object itself
            matching_keys = {
                key for key, value in parent_object.__dict__.items() if value is target_object
            }
            for key in matching_keys:
                names.append(
                    ReferrerName(
                        name=f"{type(parent_object).__name__}.{key} (instance attribute)",
                        is_container=True,
                        referrer=parent_object,
                    )
                )
        else:
            num_referrers = sys.getrefcount(target_object) - 1
            if _reached_referrer_limit(num_referrers, self._single_object_referrer_limit):
                names.append(
                    ReferrerName(
                        name=str(
                            _ReferrerLimitReached(
                                num_referrers, self._single_object_referrer_limit
                            )
                        ),
                        is_container=False,
                        referrer=parent_object,
                    )
                )
            else:
                # This is the logic where the parent of an object is a dict, which matches the
                # __dict__ attribute of the grandparent object (which is the actual referring
                # object)
                grandparents = gc.get_referrers(parent_object)
                # If the parent has referrers, we need to check if any of them are classes with
                # instance attributes that refer to the target object (via their __dict__).
                if grandparents:
                    for grandparent in grandparents:
                        # If the grandparent is a class, check if the parent is the class's dict.
                        # If so the grandparent is referring to the target object via an instance
                        # attribute. This affects the name that we give the target.
                        if (
                            _safe_hasattr(grandparent, "__dict__")
                            and grandparent.__dict__ is parent_object
                        ):
                            matching_keys = {
                                key
                                for key, value in parent_object.items()
                                if value is target_object
                            }
                            for key in matching_keys:
                                names.append(
                                    ReferrerName(
                                        name=f"{type(grandparent).__name__}.{key} (instance "
                                        "attribute)",
                                        is_container=True,
                                        referrer=grandparent,
                                    )
                                )
        return names

    def _get_container_names(
        self, target_object: Any, parent_object: Any
    ) -> Set[ReferrerName]:
        names: List[ReferrerName] = []
        try:
            if isinstance(
                parent_object, (collections.abc.Mapping, collections.abc.MutableMapping)
            ):
                names.extend(
                    _filter_container(
                        parent_object,
                        extractor_func=lambda x: x.items(),
                        filter_func=lambda x: x[1] is target_object,
                        selector_func=lambda x: ReferrerName(
                            name=f"{type(parent_object).__name__} key="
                            f"{_safe_str(x[0], truncate_at=_MAX_MAPPING_KEY_LENGTH)}",
                            is_container=True,
                            referrer=parent_object,
                        ),
                    )
                )
            elif isinstance(
                parent_object,
                (collections.abc.Sequence, collections.abc.MutableSequence),
            ):
                names.extend(
                    _filter_container(
                        parent_object,
                        extractor_func=lambda x: enumerate(x),
                        filter_func=lambda x: x[1] is target_object,
                        selector_func=lambda x: ReferrerName(
                            name=f"{type(parent_object).__name__} index={x[0]}",
                            is_container=True,
                            referrer=parent_object,
                        ),
                    )
                )
            # Note: the order is important here, since mappings and sequences are
            # also containers.
            elif isinstance(parent_object, collections.abc.Container):
                # This doesn't add a lot of extra information, but it's nice to be
                # consistent in the fact that we add an "extra" node for all containers.
                names.append(
                    ReferrerName(
                        name=f"{type(parent_object).__name__} member",
                        is_container=True,
                        referrer=parent_object,
                    )
                )
        except Exception as e:
            # Certain containers don't support iteration. We can't do anything about that,
            # so we just fall back to the more general name for the parent object.
            # The catch-all exception isn't ideal, but we don't know what exceptions
            # the container types might raise.
            logger.warning(
                f"Error encountered while iterating over a container: {e}. "
                f"Falling-back to the parent object's type name."
            )
            pass
        # If we couldn't find any more specific names, fall back to the parent's type name.
        if not names:
            names.append(
                ReferrerName(
                    name=f"{type(parent_object).__name__} (object)",
                    is_container=False,
                    referrer=parent_object,
                )
            )
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
            module for name, module in sys.modules.items() if name.startswith(module_prefix)
        ]
        self._global_vars = _get_global_vars()

    def get_names(self, target_object: Any) -> Set[str]:
        names = set()
        for module in self._modules:
            if _safe_hasattr(module, "__dict__"):
                for var_name, var_value in module.__dict__.items():
                    if (
                        var_value is target_object
                        and _GlobalVariable(var_name, id(var_value), id(module))
                        not in self._global_vars
                    ):
                        names.add(f"{module.__name__}.{var_name} ({_TYPE_MODULE_VARIABLE})")
        return names

    def get_type(self) -> str:
        return _TYPE_MODULE_VARIABLE


@dataclass(frozen=True)
class _PrintableGraphNode:
    unique_id: int
    display_name: str

    def __str__(self) -> str:
        return self.display_name


class _ReferrerGraph(ReferrerGraph):
    def __init__(self, graph: nx.DiGraph):
        self._graph = graph

    def __str__(self):
        # Use a copy of the networkx generate_network_text function to avoid depending on
        # a narrow range of networkx versions.
        printable_graph = self._to_printable_graph()
        # Use UtfUndirectedGlyphs here. Although the graph is directed, it's a bit confusing
        # to use the directed glyphs here as the direction is from referents to referrers,
        # which might be confusing.
        return "\n" + "\n".join(
            line
            for line in networkx_copy.generate_network_text(
                printable_graph, override_glyphs=networkx_copy.UtfUndirectedGlyphs
            )
        )

    def to_networkx(self) -> nx.DiGraph:
        # Reverse the graph so that the direction of edges is from referrer to referent.
        # This is probably what most people would expect.
        return self._graph.reverse(copy=True)

    def _to_printable_graph(self) -> nx.DiGraph:
        new_graph = nx.DiGraph()
        seen: Set[ReferrerGraphNode] = set()
        unique_id = 0
        # Make a string representation of the graph, breaking any cycles.
        for u, v in self._graph.edges():
            if isinstance(u, ReferrerGraphNode) and isinstance(v, ReferrerGraphNode):
                u_str = str(u)
                v_str = str(v)
                # We deal with cycles and root nodes specially. In particular, we
                # suffix them with some special text, and we ensure that they are unique
                # within the graph to break any cycles.
                if v in seen:
                    v_str = f"{v_str} (cycle member)"
                    printable_v = _PrintableGraphNode(unique_id=unique_id, display_name=v_str)
                    unique_id += 1
                elif self._graph.out_degree(v) == 0:
                    v_str = v_str + " (root)"
                    printable_v = _PrintableGraphNode(unique_id=unique_id, display_name=v_str)
                    unique_id += 1
                else:
                    printable_v = _PrintableGraphNode(unique_id=0, display_name=v_str)
                printable_u = _PrintableGraphNode(unique_id=0, display_name=u_str)
                new_graph.add_edge(printable_u, printable_v)
                seen.add(u)
                seen.add(v)
            else:
                raise ValueError(f"Unexpected type: {type(u)} or {type(v)}")
        return new_graph


class _ReferrerGraphBuilder:
    """
    Builds a graph of referrers for a set of target objects.
    """

    def __init__(
        self,
        target_objects: Iterable[Any],
        module_prefixes: Optional[Collection[str]],
        max_untracked_search_depth: int,
        single_object_referrer_limit: Optional[int],
        exclude_object_ids: Optional[Sequence[int]] = None,
    ):
        if not module_prefixes:
            stack_frames = inspect.stack()
            for frame_info in stack_frames:
                frame_module = inspect.getmodule(frame_info.frame)
                if frame_module and not frame_module.__name__.startswith(_PACKAGE_PREFIX):
                    # Use the top-level package of the calling code as the module prefix
                    # (with a trailing dot). For example, if the calling code is in a module
                    # called my_module.do_thing, the module prefix would be "my_module.".
                    # In some cases (like Jupyter notebooks), there may not be a top-level
                    # package, in which there won't be any module prefixes. We log a warning
                    # in this case.
                    module_prefixes = [f"{frame_module.__name__.split('.')[0]}."]
                    break
        if not module_prefixes:
            logger.warning(
                "Could not determine the top-level package of the calling code. "
                "You can specify the module_prefixes parameter to set this explicitly."
            )

        # Populate a dict of object IDs to the closures that enclose them.
        # This is used for finding referrers that are closures, and identifying
        # the name of the variable that is enclosed.
        self._id_to_enclosing_closure = self._get_closure_functions()

        self._target_objects = target_objects

        excluded_id_set = {
            id(self),
            id(self.__dict__),
            id(target_objects),
            id(self._id_to_enclosing_closure),
        } | set(exclude_object_ids)

        # Get the referrers of the target objects that are not tracked by the garbage collector.
        (
            self._untracked_objects_referrers,
            extra_exclusions,
        ) = self._get_untracked_object_referrers(
            target_objects,
            excluded_id_set=excluded_id_set,
            max_depth=max_untracked_search_depth,
            module_prefixes=module_prefixes,
        )

        self._single_object_referrer_limit = single_object_referrer_limit

        # Note: when we create the name finders is important because some implementations
        # start to track the objects that are in the environment when they are created.
        self._name_finders = _get_name_finders(module_prefixes)

        # Exclude the builder and its attributes from the referrer name finders, since we
        # store a reference to the target objects. Also exclude the target objects container.
        self._referrer_name_finders = _get_referrer_name_finders(
            excluded_id_set=excluded_id_set | extra_exclusions,
            single_object_referrer_limit=self._single_object_referrer_limit,
        )

    def build(
        self,
        max_depth: Optional[int],
        timeout: Optional[float],
    ) -> ReferrerGraph:
        start_time = default_timer()
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

            if self._reached_timeout(start_time=start_time, timeout=timeout):
                time_taken = default_timer() - start_time
                timeout_object = _Timeout(time_taken=time_taken, timeout=timeout)
                referrer_nodes = self._get_referrer_nodes(
                    target_object=target_object,
                    referrer=timeout_object,
                )
                for referrer_graph_node, _ in referrer_nodes:
                    graph.add_edge(target_graph_node, referrer_graph_node)
            elif max_depth is None or depth < max_depth:

                # For each non-referrer name pointing to the target object, add an edge to
                # the graph. Also process any additional referrers that are returned.
                non_referrer_nodes = self._get_non_referrer_nodes(target_object)
                for non_referrer_graph_node in non_referrer_nodes:
                    graph.add_edge(target_graph_node, non_referrer_graph_node)

                # For each referrer of the target object, find the name(s) of the referrer and
                # add an edge to the graph for each
                for referrer_object in self._get_referrers(
                    target_object,
                    single_object_referrer_limit=self._single_object_referrer_limit,
                ):
                    if not self._is_excluded(referrer_object):
                        self._process_referrer_object(
                            referrer_object=referrer_object,
                            target_object=target_object,
                            target_graph_node=target_graph_node,
                            graph=graph,
                            stack=stack,
                            seen_ids=seen_ids,
                            current_depth=depth,
                        )
            else:
                limit_exceeded_object = _DepthLimitReached(limit=max_depth)
                referrer_nodes = self._get_referrer_nodes(
                    target_object=target_object,
                    referrer=limit_exceeded_object,
                )
                for referrer_graph_node, _ in referrer_nodes:
                    graph.add_edge(target_graph_node, referrer_graph_node)

        return _ReferrerGraph(graph)

    def _process_referrer_object(
        self,
        referrer_object: Any,
        target_object: Any,
        target_graph_node: ReferrerGraphNode,
        graph: nx.DiGraph,
        stack: Deque[Tuple[ReferrerGraphNode, Any, int]],
        seen_ids: Set[int],
        current_depth: int,
    ):
        """
        Processes a referrer object. This involves finding the names of the referrer object
        (in relation to the target object) and adding the appropriate edges to the graph based
        on the names found.

        The stack is also updated to include the next referrer(s) to process, if they haven't
        been seen before (based on their presence in `seen_ids`).
        """
        referrer_nodes = self._get_referrer_nodes(
            target_object=target_object,
            referrer=referrer_object,
        )
        for (
            referrer_graph_node,
            name_obj,
        ) in referrer_nodes:
            actual_referrer = name_obj.referrer
            is_container = name_obj.is_container
            graph.add_edge(target_graph_node, referrer_graph_node)
            # If the referrer is a container, we also add the container itself to
            # the graph. This makes the graph more readable when, for example,
            # instance attributes refer to objects that are referenced in other
            # instance attributes.
            if is_container:
                next_node = self._get_object_node(
                    actual_referrer,
                    object_id=id(actual_referrer),
                    is_target=False,
                )
                graph.add_edge(referrer_graph_node, next_node)
                next_depth = current_depth + 2
            else:
                next_node = referrer_graph_node
                next_depth = current_depth + 1
            actual_referrer_id = id(actual_referrer)
            # Avoid an infinite loop by only adding referrers that we haven't seen
            # before. We still add the relevant edge to the graph so we can see the
            # relationship though.
            if actual_referrer_id not in seen_ids:
                seen_ids.add(actual_referrer_id)
                # Exclude the actual referrer. The original referrer object would have
                # already been excluded if necessary, but the actual referrer
                # may be different.
                if not self._is_excluded(actual_referrer):
                    stack.append((next_node, actual_referrer, next_depth))

    def _reached_timeout(self, start_time: float, timeout: Optional[float]) -> bool:
        if timeout is None:
            return False
        else:
            return default_timer() - start_time > timeout

    def _is_excluded(self, obj: Any) -> bool:
        # We exclude these objects because anything referenced by them should be picked-up
        # elsewhere (locals global etc), and excluding them speeds things up a lot.
        # We don't want to exclude closures, but they are wrapped in a _ClosureDetails object
        # so they won't be excluded.
        # We also exclude certain internal objects that aren't real referrers even though they
        # (temporarily) hold a reference to objects.
        return (
            inspect.isframe(obj)
            or inspect.isroutine(obj)
            or inspect.ismodule(obj)
            or isinstance(obj, ReferrerName)
        )

    def _get_referrers(
        self, target_object: Any, single_object_referrer_limit: Optional[int]
    ) -> Iterable[Any]:
        # If this is an internal object, we need to unpack it to get the real object.
        if isinstance(target_object, _InternalReferrer):
            target_object = target_object.unpack()
        num_referrers = sys.getrefcount(target_object) - 1
        if _reached_referrer_limit(num_referrers, single_object_referrer_limit):
            refs = [_ReferrerLimitReached(num_referrers, single_object_referrer_limit)]
        else:
            # This might be empty if the object is not tracked. However, in some cases untracked
            # objects have referrers, so we need to eliminate duplicates.
            refs = gc.get_referrers(target_object)
        ref_ids = {id(ref) for ref in refs}
        for untracked_referrer in self._untracked_objects_referrers.get(id(target_object), []):
            if id(untracked_referrer) not in ref_ids:
                refs.append(untracked_referrer)
        closures = self._id_to_enclosing_closure.get(id(target_object), [])
        for closure in closures:
            if id(closure) not in ref_ids:
                refs.append(closure)
        return refs

    def _get_initial_target_node(
        self, target_object: Any
    ) -> Tuple[ReferrerGraphNode, Any, int]:
        return (
            self._get_object_node(target_object, object_id=id(target_object), is_target=True),
            target_object,
            0,
        )

    def _get_object_node(self, target_object: Any, object_id: int, is_target: bool):
        name = f"{type(target_object).__name__} (object)"
        return ReferrerGraphNode(name=name, id=object_id, type="object", is_target=is_target)

    def _get_referrer_nodes(
        self, target_object: Any, referrer: Any
    ) -> Set[Tuple[ReferrerGraphNode, ReferrerName]]:
        nodes = list()
        seen_names: Set[str] = set()
        for finder in self._referrer_name_finders:
            for name_obj in finder.get_names(target_object, referrer):
                # Filter out duplicate names.
                if name_obj.name not in seen_names:
                    nodes.append(
                        (
                            ReferrerGraphNode(
                                name=name_obj.name,
                                id=id(name_obj.referrer),
                                type=finder.get_type(),
                            ),
                            name_obj,
                        )
                    )
                    seen_names.add(name_obj.name)
        return nodes

    def _get_non_referrer_nodes(self, target_object: Any) -> Set[ReferrerGraphNode]:
        nodes = set()
        for finder in self._name_finders:
            names = finder.get_names(target_object)
            # We just use the target object ID as the object ID as we don't have anything else
            # (this is maybe a bit weird?).
            for name in names:
                nodes.add(
                    ReferrerGraphNode(name=name, id=id(target_object), type=finder.get_type())
                )
        return nodes

    def _get_untracked_object_referrers(
        self,
        target_objects: Iterable[Any],
        excluded_id_set: Set[int],
        max_depth: int,
        module_prefixes: Collection[str],
    ) -> Tuple[Mapping[int, List[Any]], Set[int]]:
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

        if len(untracked_target_object_ids) > 0:

            roots = self._get_untracked_search_roots(
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
        extra_exclusions.add(id(return_dict))

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
        """
        Populates the referrers of untracked objects, where the object chain leads to one of
        the target objects we are looking for. This is a recursive function that walks the
        object graph, starting from the roots.
        """
        # This method is a bit horrible. It can maybe be made less complex.
        # The fact that we're using a stack *and* recursing is a bit weird, so
        # I'm pretty sure that can be fixed.
        obj_id = id(obj)
        if inspect.isframe(obj) or obj_id in do_not_visit or depth >= max_depth:
            return
        else:
            do_not_visit.add(obj_id)
            for referent in gc.get_referents(obj):
                if not gc.is_tracked(referent):
                    referent_id = id(referent)
                    # Push the referrer of the untracked object on to the stack.
                    # We will pop this off when we return from the recursion.
                    untracked_stack.append(obj)
                    # If we find one of the target objects, add the current object to the
                    # referrers list for the target object. We also walk back up the untracked
                    # object stack and add any other objects that refer indirectly
                    # to the target object.
                    if referent_id in untracked_target_object_ids:
                        id_to_add = referent_id
                        for untracked_obj in reversed(untracked_stack):
                            untracked_object_referrers[id_to_add].append(untracked_obj)
                            id_to_add = id(untracked_obj)
                    # Recurse into the referent. The exit condition is when we encounter an
                    # object in the do_not_visit list, we hit the max depth, or we find a
                    # tracked object (but I'm not sure that untracked objects can refer
                    # to tracked objects, so this last case may not happen in practice).
                    self._populate_untracked_object_referrers(
                        obj=referent,
                        do_not_visit=do_not_visit,
                        untracked_object_referrers=untracked_object_referrers,
                        untracked_target_object_ids=untracked_target_object_ids,
                        untracked_stack=untracked_stack,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                    untracked_stack.pop()

    def _contains_untracked_objects(self, obj: Any):
        return any(not gc.is_tracked(referent) for referent in gc.get_referents(obj))

    def _get_untracked_search_roots(
        self, do_not_visit: Set[int], module_prefixes: Optional[Collection[str]] = None
    ) -> List[Any]:
        """
        Gets "root" objects from which to search for untracked objects. We search in all
        objects that have untracked referents, from within:

         * The result of gc.get_objects()
         * Local variables
         * Global variables
         * Module-level variables that are not global variables.
        """
        roots = []

        for obj in gc.get_objects():
            if self._contains_untracked_objects(obj):
                roots.append(obj)

        for thread_frames in _get_frames_for_all_threads().values():
            for frame in thread_frames:
                # Exclude the locals and globals themselves from the search.
                do_not_visit.add(id(frame.f_locals))
                do_not_visit.add(id(frame.f_globals))
                roots.extend(
                    _filter_container(
                        frame.f_locals,
                        extractor_func=lambda x: x.values(),
                        filter_func=lambda x: self._contains_untracked_objects(x),
                        selector_func=lambda x: x,
                    )
                )
                roots.extend(
                    _filter_container(
                        frame.f_globals,
                        extractor_func=lambda x: x.values(),
                        filter_func=lambda x: self._contains_untracked_objects(x),
                        selector_func=lambda x: x,
                    )
                )

        self._global_vars = _get_global_vars()
        self._modules = [
            module
            for name, module in sys.modules.items()
            if self._matches_prefixes(name, module_prefixes)
        ]
        for module in self._modules:
            if _safe_hasattr(module, "__dict__"):
                for var_name, var_value in module.__dict__.items():
                    if (
                        self._contains_untracked_objects(var_value)
                        and _GlobalVariable(var_name, id(var_value), id(module))
                        not in self._global_vars
                    ):
                        roots.append(var_value)
        return roots

    def _matches_prefixes(self, module_name: str, module_prefixes: Collection[str]):
        return any(module_name.startswith(prefix) for prefix in module_prefixes)

    def _get_closure_functions(
        self,
    ) -> Dict[int, List[_ClosureDetails]]:
        id_to_enclosing_closure: Dict[int, List[_ClosureDetails]] = collections.defaultdict(
            list
        )
        all_closure_ids = set()
        for possible_function in gc.get_objects():
            try:
                if inspect.isfunction(possible_function) or inspect.ismethod(
                    possible_function
                ):
                    try:
                        closure_vars = inspect.getclosurevars(possible_function)
                    except TypeError:
                        # It's not clear why, but some things that claim to be functions
                        # return a TypeError with "is not a Python function" here, so we
                        # just skip them.
                        continue
                    except ValueError:
                        # The inspect.getclosurevars function raises a ValueError with
                        # "Cell is empty" in some cases. it's not clear how to avoid this, so
                        # we just skip these cases.
                        continue
                    for var_name, var_value in chain(
                        closure_vars.nonlocals.items(), closure_vars.globals.items()
                    ):
                        id_to_enclosing_closure[id(var_value)].append(
                            _ClosureDetails(possible_function, var_name)
                        )
                        all_closure_ids.add(id(possible_function))
            except ReferenceError as e:
                # This can happen if the object is a weak reference proxy where the underlying
                # object has been garbage collected. We just skip these cases.
                continue
        return id_to_enclosing_closure


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
    single_object_referrer_limit: Optional[float],
) -> Sequence[ReferrerNameFinder]:
    return [
        ObjectNameFinder(
            single_object_referrer_limit=single_object_referrer_limit,
            excluded_id_set=excluded_id_set,
        )
    ]


def _get_global_vars() -> Set[_GlobalVariable]:
    """
    Gets the names and IDs of all global variables in the current environment.
    :return: A set of `_GlobalVariable`s
    """
    global_vars: Set[_GlobalVariable] = set()
    for thread_frames in _get_frames_for_all_threads().values():
        for frame in thread_frames:
            # We use filter_container here just to be more robust to modification of
            # f_globals during iteration. I've never seen this in practice, but I
            # *think* it's a possibility.
            global_tuples = _filter_container(
                frame.f_globals,
                extractor_func=lambda x: x.items(),
                filter_func=lambda x: True,
                selector_func=lambda x: x,
            )
            for var_name, var_value in global_tuples:
                module = inspect.getmodule(var_value)
                if module:
                    global_vars.add(_GlobalVariable(var_name, id(var_value), id(module)))
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
) -> Iterable[_V]:
    """
    Filters a container using the given functions.

    This function is robust to containers being modified while we're iterating over them.
    If a RuntimeError is raised, we make a copy of the container and try again.

    :param container: The container to filter.
    :param extractor_func: A function that creates an iterable from the container
    :param filter_func: A function that filters items from the iterable
    :param selector_func: A function that selects items from the iterable
    :return: The selected items.
    """
    try:
        return [selector_func(item) for item in extractor_func(container) if filter_func(item)]
    except RuntimeError as e:
        # If we get a RuntimeError, it's likely that the iterable is being modified while
        # we're iterating over it. In this case we make a copy of the container and try again.
        logger.warning(
            f"Runtime error encountered while iterating over a container: {e}"
            f"Retrying with a copy."
        )
        return [
            selector_func(item)
            for item in extractor_func(copy(container))
            if filter_func(item)
        ]


def _reached_referrer_limit(
    num_referrers: int, single_object_referrer_limit: Optional[int]
) -> bool:
    # Immortal objects don't return the real number of referrers in Python >=3.12.
    # So can't apply the limit to these objects.
    if _is_probably_immortal(num_referrers) or single_object_referrer_limit is None:
        return False
    else:
        return num_referrers > single_object_referrer_limit


def _is_probably_immortal(referrer_count: int):
    """
    Guesses whether an object is immortal based on its referrer count.

    According to the Python documentation "Immortal objects have very large refcounts that do
    not match the actual number of references to the object". However, it's not clear
    how many references are returned, so we just guess at one billion.

    There does not seem to be any better way to determine if an object is immortal at the moment.
    """
    return referrer_count > IMMORTAL_OBJECT_REFCOUNT


def _safe_str(obj: Any, truncate_at: int) -> str:
    try:
        str_repr = str(obj)
        if len(str_repr) > truncate_at:  # pragma: no cover
            str_repr = (
                str_repr[:truncate_at] + f" … ({len(str_repr) - truncate_at} more chars)"
            )
        return str_repr
    except Exception as e:
        # Some things don't like their string representation being obtained.
        return f"<Error when getting string representation: {str(e)}>"


def _safe_hasattr(obj: Any, attr_name: str) -> bool:
    try:
        return hasattr(obj, attr_name)
    except Exception:
        # Some things don't like hasattr being called on them.
        return False
