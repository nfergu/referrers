import dataclasses
import gc
import io
import random
import re
import sys
import weakref
from functools import lru_cache
from time import sleep
import threading
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, List, Set

import pytest
from networkx import bfs_edges

import referrers
from referrers.impl import (
    GlobalVariableNameFinder,
    LocalVariableNameFinder,
    ModuleLevelNameFinder,
    ObjectNameFinder,
    _ReferrerGraphBuilder,
    IMMORTAL_OBJECT_REFCOUNT,
    ReferrerGraphNode,
    ReferrerNameType,
)
from tests.testing_modules.module2 import imported_module_variable


class TestClass1:
    pass


CONSTANT = TestClass1()


class TestClass2:
    def __init__(self, my_attribute: TestClass1):
        self.my_attribute = my_attribute


@dataclasses.dataclass(frozen=True)
class TestClass2Frozen:
    let_it_go: TestClass1


class TestClass3:
    def __init__(self, my_attribute: TestClass1, my_attribute2: TestClass1):
        self.my_attribute = my_attribute
        self._my_attribute2 = my_attribute2


class DictContainer:
    def __init__(self, my_dict: Dict[Any, Any]):
        self.my_dict = my_dict


class Link:
    def __init__(self, next_link: Optional["Link"] = None):
        self.next_link = next_link


class TestClass2Container:
    def __init__(self, contained_attribute: TestClass2):
        self.contained_attribute = contained_attribute


def stack_frames_namer_assert_in_function(passed_ref: TestClass1, name: str):
    names = LocalVariableNameFinder().get_names(passed_ref)
    assert names == {name}


def get_print_input_closure(input_val: TestClass1) -> Callable[[], None]:
    def print_input():
        print(input_val)

    return print_input


module_level_variable = TestClass1()
module_level_closure = get_print_input_closure(module_level_variable)


class TargetClass:
    def __init__(self):
        self._running = True

    def target_function(self, passed_ref_to_target_function: TestClass1):
        while self._running:
            sleep(0.0000001)

    def stop(self):
        self._running = False


class DictWithoutItems(dict):
    def items(self):
        raise NotImplementedError("This dict is not iterable")


def _one(iterable: Iterable):
    iterator = iter(iterable)
    try:
        result = next(iterator)
    except StopIteration:
        raise ValueError("No items in iterable")
    try:
        next(iterator)
        raise ValueError("More than one item in iterable")
    except StopIteration:
        return result


MY_GLOBAL = TestClass1()


class TreeNode:
    def __init__(self, parent: Optional["TreeNode"]):
        self.parent = parent


class TreeNodeHolder:
    def __init__(self, node1: TreeNode, node2: TreeNode, node3: TreeNode, node4: TreeNode):
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4


class TestLocalVariableNameFinder:
    def test_calling_frame(self):
        ref1 = TestClass1()
        names = LocalVariableNameFinder().get_names(ref1)
        assert names == {"test_calling_frame.ref1 (local)"}

    def test_calling_frame_multiple(self):
        ref1 = TestClass1()
        ref2 = ref1
        names = LocalVariableNameFinder().get_names(ref1)
        assert names == {
            "test_calling_frame_multiple.ref1 (local)",
            "test_calling_frame_multiple.ref2 (local)",
        }
        assert ref1 is ref2

    def test_calling_frame_two_classes(self):
        ref1 = TestClass1()
        ref2 = TestClass1()
        names = LocalVariableNameFinder().get_names(ref1)
        assert names == {"test_calling_frame_two_classes.ref1 (local)"}
        assert ref1 is not ref2

    def test_calling_frame_assert_in_function(self):
        top_level_ref = TestClass1()
        stack_frames_namer_assert_in_function(
            top_level_ref,
            name="test_calling_frame_assert_in_function.top_level_ref (local)",
        )

    def test_nothing_in_stack_frames(self):
        # This should return an empty set because the object is not in the stack frames
        # (it will be in the stack frame of the call to get_names, but this is excluded).
        names = LocalVariableNameFinder().get_names(TestClass1())
        assert names == set()

    def test_calling_frame_in_separate_thread(self):
        target = TargetClass()
        my_ref = TestClass1()
        thread = threading.Thread(target=target.target_function, args=(my_ref,))
        thread.start()
        names = LocalVariableNameFinder().get_names(my_ref)
        target.stop()
        thread.join()
        # We should get names from all threads, including the main thread.
        assert names == {
            "test_calling_frame_in_separate_thread.my_ref (local)",
            "target_function.passed_ref_to_target_function (local)",
        }

    def test_also_in_instance_attribute(self):
        ref1 = TestClass1()
        containing_class = TestClass2(ref1)
        names = LocalVariableNameFinder().get_names(ref1)
        # The StackFramesNamer should not find the instance attribute, because it only looks at
        # local variables.
        assert names == {"test_also_in_instance_attribute.ref1 (local)"}
        assert containing_class.my_attribute is ref1

    def test_interned_ints(self):
        ref1 = 255
        ref2 = 255
        ref3 = 257
        names = LocalVariableNameFinder().get_names(ref1)
        # Since 255 is interned, ref1 and ref2 will be the same object, but ref3 will not.
        # This isn't particularly desirable behaviour, since it's a bit unintuitive, but
        # there's not much we can do about it.
        assert names == {
            "test_interned_ints.ref1 (local)",
            "test_interned_ints.ref2 (local)",
        }
        assert ref1 is ref2
        assert ref1 is not ref3


class TestObjectNameFinder:
    def test_instance_attribute(self):
        local_ref = TestClass1()
        containing_class = TestClass2(local_ref)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        name = _one(names)
        assert name.name == "TestClass2.my_attribute (instance attribute)"
        assert name.is_container
        assert name.referrer is containing_class
        assert containing_class.my_attribute is local_ref

    def test_instance_attribute_frozen_dataclass(self):
        local_ref = TestClass1()
        containing_class = TestClass2Frozen(local_ref)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        name = _one(names)
        assert name.name == "TestClass2Frozen.let_it_go (instance attribute)"
        assert name.is_container
        assert name.referrer is containing_class
        assert containing_class.let_it_go is local_ref

    def test_instance_attribute_changed(self):
        local_ref = TestClass1()
        # Construct the containing class with a different object and then change it.
        containing_class = TestClass2(TestClass1())
        containing_class.my_attribute = local_ref
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        name = _one(names)
        assert name.name == "TestClass2.my_attribute (instance attribute)"
        assert name.is_container
        assert name.referrer is containing_class
        assert containing_class.my_attribute is local_ref

    def test_multiple_instance_attributes_in_same_class(self):
        local_ref = TestClass1()
        containing_class = TestClass3(local_ref, local_ref)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        assert len(names) == 2
        name_strings = {name.name for name in names}
        assert name_strings == {
            "TestClass3.my_attribute (instance attribute)",
            "TestClass3._my_attribute2 (instance attribute)",
        }
        for name in names:
            assert name.is_container
            assert name.referrer is containing_class
        assert containing_class.my_attribute is local_ref

    def test_instance_attribute_in_different_classes(self):
        local_ref = TestClass1()
        containing_class = TestClass2(local_ref)
        containing_class2 = TestClass2(local_ref)
        for referrer in gc.get_referrers(local_ref):
            names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
                local_ref, referrer
            )
            if referrer is containing_class.__dict__ or referrer is containing_class:
                name = _one(names)
                assert name.name == "TestClass2.my_attribute (instance attribute)"
                assert name.is_container
                assert name.referrer is containing_class
            elif referrer is containing_class2.__dict__ or referrer is containing_class2:
                name = _one(names)
                assert name.name == "TestClass2.my_attribute (instance attribute)"
                assert name.is_container
                assert name.referrer is containing_class2
            else:
                raise AssertionError(f"Unexpected referrer: {referrer}")
        assert containing_class.my_attribute is local_ref
        assert containing_class2.my_attribute is local_ref

    def test_dict(self):
        local_ref = TestClass1()
        my_dict = {"mykey": local_ref}
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        name = _one(names)
        assert name.name == "dict key=mykey"
        assert name.is_container
        assert name.referrer is my_dict
        assert my_dict["mykey"] is local_ref

    def test_instance_attribute_and_dict(self):
        local_ref = TestClass1()
        containing_class = TestClass2(local_ref)
        my_dict = {"mykey": local_ref}
        for referrer in gc.get_referrers(local_ref):
            names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
                local_ref, referrer
            )
            if referrer is my_dict:
                name = _one(names)
                assert name.name == "dict key=mykey"
                assert name.is_container
                assert name.referrer is my_dict
            elif referrer is containing_class.__dict__ or referrer is containing_class:
                name = _one(names)
                assert name.name == "TestClass2.my_attribute (instance attribute)"
                assert name.is_container
                assert name.referrer is containing_class
            else:
                raise AssertionError(f"Unexpected referrer: {referrer}")
        assert containing_class.my_attribute is local_ref
        assert my_dict["mykey"] is local_ref

    def test_instance_attribute_references_dict_of_other_object(self):
        local_ref = TestClass1()
        containing_class = TestClass2(local_ref)
        # This is a bit of an odd situation, but it's possible. The __dict__ of one class
        # is referenced by an instance attribute of another class.
        dict_container = DictContainer(containing_class.__dict__)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        # This is perhaps a bit confusing, but we only report the instance attribute
        # of the containing class, not the separate dict.
        name = _one(names)
        assert name.name == "TestClass2.my_attribute (instance attribute)"
        assert name.is_container
        assert name.referrer is containing_class
        assert containing_class.my_attribute is local_ref
        assert dict_container.my_dict is containing_class.__dict__

    def test_list(self):
        local_ref = TestClass1()
        my_list = [1, local_ref, 3]
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        name = _one(names)
        assert name.name == "list index=1"
        assert name.is_container
        assert name.referrer is my_list
        assert my_list[1] is local_ref

    def test_instance_attribute_and_list(self):
        local_ref = TestClass1()
        containing_class = TestClass2(local_ref)
        my_list = [1, local_ref, 3]
        for referrer in gc.get_referrers(local_ref):
            names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
                local_ref, referrer
            )
            if referrer is my_list:
                name = _one(names)
                assert name.name == "list index=1"
                assert name.is_container
                assert name.referrer is my_list
            elif referrer is containing_class.__dict__ or referrer is containing_class:
                name = _one(names)
                assert name.name == "TestClass2.my_attribute (instance attribute)"
                assert name.is_container
                assert name.referrer is containing_class
            else:
                raise AssertionError(f"Unexpected referrer: {referrer}")
        assert containing_class.my_attribute is local_ref
        assert my_list[1] is local_ref

    def test_set(self):
        local_ref = TestClass1()
        my_set = {local_ref}
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        name = _one(names)
        assert name.name == "set member"
        assert name.is_container
        assert name.referrer is my_set
        assert local_ref in my_set

    @pytest.mark.skipif(sys.version_info >= (3, 13), reason="Requires Python 3.13+")
    def test_class_dict(self):
        # This test doesn't work in Python 3.13+ because gc.get_referrers returns
        # nothing for the class dict.
        local_ref = TestClass1()
        containing_class = TestClass2(local_ref)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            containing_class.__dict__, _one(gc.get_referrers(containing_class.__dict__))
        )
        name = _one(names)
        assert name.name == "TestClass2 (object)"
        assert not name.is_container
        assert name.referrer is containing_class
        assert containing_class.my_attribute is local_ref

    def test_outer_container(self):
        local_ref = TestClass1()
        containing_class = TestClass2(local_ref)
        outer_class = TestClass2Container(containing_class)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            containing_class, _one(gc.get_referrers(containing_class))
        )
        name = _one(names)
        assert name.name == "TestClass2Container.contained_attribute (instance attribute)"
        assert name.is_container
        assert name.referrer is outer_class
        assert containing_class.my_attribute is local_ref
        assert outer_class.contained_attribute is containing_class

    def test_with_dict_that_does_not_support_getting_items(self):
        # Tests what happens when we have a dict where we cannot get items from it.
        # These seem to exist in the wild.
        local_ref = TestClass1()
        my_dict = DictWithoutItems(mykey=local_ref)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        name = _one(names)
        assert name.name == "DictWithoutItems (object)"
        assert not name.is_container
        assert name.referrer is my_dict
        assert my_dict["mykey"] is local_ref

    def test_tree_node(self):
        tree_node1 = TreeNode(parent=None)
        tree_node2 = TreeNode(parent=tree_node1)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            tree_node1, _one(gc.get_referrers(tree_node1))
        )
        name = _one(names)
        assert name.name == "TreeNode.parent (instance attribute)"
        assert name.is_container
        assert name.referrer is tree_node2

    def test_tree_node_holder(self):
        """
        Tests a tree holder that has multiple references to nodes in a tree.
        """
        (_, leaf_holder, *_) = _construct_mini_tree()
        # The behaviour of gc.get_referrers is kind of strange. In Python > 3.12 the first
        # time the referrers of one of the member attributes are obtained, the referrer
        # is the LeafNodeHolder object, but subsequently the referrer is the
        # LeafNodeHolder object's dict. I don't know this happens, but we try to ensure the
        # result of the name finder is consistent.
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            leaf_holder.node1, _one(gc.get_referrers(leaf_holder.node1))
        )
        name = _one(names)
        assert name.name == "TreeNodeHolder.node1 (instance attribute)"
        assert name.is_container
        assert name.referrer is leaf_holder
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            leaf_holder.node1, _one(gc.get_referrers(leaf_holder.node1))
        )
        name = _one(names)
        assert name.name == "TreeNodeHolder.node1 (instance attribute)"
        assert name.is_container
        assert name.referrer is leaf_holder
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            leaf_holder.node2, _one(gc.get_referrers(leaf_holder.node2))
        )
        name = _one(names)
        assert name.name == "TreeNodeHolder.node2 (instance attribute)"
        assert name.is_container
        assert name.referrer is leaf_holder
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            leaf_holder.node3, _one(gc.get_referrers(leaf_holder.node3))
        )
        name = _one(names)
        assert name.name == "TreeNodeHolder.node3 (instance attribute)"
        assert name.is_container
        assert name.referrer is leaf_holder
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            leaf_holder.node4, _one(gc.get_referrers(leaf_holder.node4))
        )
        name = _one(names)
        assert name.name == "TreeNodeHolder.node4 (instance attribute)"
        assert name.is_container
        assert name.referrer is leaf_holder

    def _name_type_from_version(self) -> ReferrerNameType:
        # Instance attribute behaviour differs between Python versions.
        # See ReferrerNameType for more details.
        if sys.version_info < (3, 11):
            return ReferrerNameType.INSTANCE_ATTRIBUTE_IN_DICT
        else:
            return ReferrerNameType.INSTANCE_ATTRIBUTE_IN_OBJECT


class TestModuleLevelNameFinder:
    def test_variable_from_non_imported_module(self):
        # Here we're testing the case where the module containing the variable has not been
        # imported directly, and is therefore not in globals. We rely on conftest.py to import
        # the module (pytest does this automatically). We also rely on the fact that 178 is
        # interned, so we can provide exactly the same object as module_variable refers to
        # without importing it (we also rely on the fact that there are no other module-level
        # references to 178 in this program).
        names = ModuleLevelNameFinder("tests.testing_modules").get_names(178)
        assert names == {"tests.testing_modules.module1.module_variable (module variable)"}

    def test_non_imported_variable(self):
        # Here we're testing the case where we have imported a module, but we have not
        # imported a variable from this module. In this case, the variable is not in globals.
        # Specifically, module2.imported_module_variable2 points to exactly the same object as
        # module2.imported_module_variable, but it's not been imported
        names = ModuleLevelNameFinder("tests.testing_modules").get_names(
            imported_module_variable
        )
        assert names == {
            "tests.testing_modules.module2.imported_module_variable2 (module variable)"
        }

    def test_scope_does_not_match(self):
        # Although the variable is a module-level variable, it's not in the module prefix
        # that we're searching for.
        names = ModuleLevelNameFinder("tests.not_matching").get_names(178)
        assert names == set()

    def test_global_variable_not_found(self):
        # The module-level name finder should not find global variables.
        names = ModuleLevelNameFinder("tests").get_names(CONSTANT)
        assert names == set()


class TestGlobalVariableNameFinder:
    def test_global_variable_from_class_import(self):
        names = GlobalVariableNameFinder().get_names(GlobalVariableNameFinder)
        assert names == {"referrers.impl.GlobalVariableNameFinder (global)"}

    def test_global_variable_module_level(self):
        names = GlobalVariableNameFinder().get_names(CONSTANT)
        assert names == {"tests.test_impl.CONSTANT (global)"}

    def test_global_variable_in_function(self):
        global my_global
        my_global = TestClass1()
        names = GlobalVariableNameFinder().get_names(my_global)
        assert names == {"tests.test_impl.my_global (global)"}

    def test_imported_module_variable(self):
        # Here we're testing the case where the module containing the variables we're
        # interested in has been imported.
        names = GlobalVariableNameFinder().get_names(imported_module_variable)
        # There are two module-level variables referring to the same object in module2
        # but only one of them is imported, so this is the only one that's picked up in
        # globals.
        assert names == {
            "tests.testing_modules.module2.imported_module_variable (global)",
        }


class A:
    def __init__(self, instance_var: str):
        self.instance_var = instance_var


@dataclasses.dataclass
class ClosureHolder:
    closure: Callable[[], None]


def _get_nested_tuples() -> Tuple[Tuple[Tuple[Tuple[str]]]]:
    a = ("tuples all the way down",)
    b = (a,)
    c = (b,)
    d = (c,)
    # CPython stops tracking tuples if they contain only immutable objects, but
    # only when they are first seen by the garbage collector, so we need to collect
    # here to trigger this.
    gc.collect()
    assert not gc.is_tracked(a)
    assert not gc.is_tracked(b)
    assert not gc.is_tracked(c)
    assert not gc.is_tracked(d)
    return d


class HeldClass:
    def __init__(self, a: int):
        self._a = a


class ClassAttributeHolder:
    class_attr: HeldClass = None


class InstanceAttributeHolder:
    def __init__(self, instance_attr: HeldClass):
        self.instance_attr = instance_attr


class StringHolder:
    def __init__(self, the_string: str):
        self.the_string = the_string


class ListHolder:
    def __init__(self, the_list: List[Any]):
        self.the_list = the_list


class SetHolder:
    def __init__(self, the_set: Set[Any]):
        self.the_set = the_set


class MultiAttributeHolder:
    def __init__(self, attr1: Any, attr2: Any):
        self.attr1 = attr1
        self.attr2 = attr2


@dataclasses.dataclass
class PythonDataclass:
    id: int
    name: str = "John Doe"


class ClassWithFunction:
    def my_func(self):
        pass


class ClassWithDataAndMethod:
    def __init__(self, data: int):
        self.data = data

    def my_method(self):
        pass


class ClassWithMethodReference:
    def __init__(self, method: Callable[[], None]):
        self.method = method


class TestGetReferrerGraph:
    def test_get_referrer_graph(self):
        the_reference = TestClass1()
        graph = referrers.get_referrer_graph(the_reference)
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["TestClass1 (object)"] == [root.name for root in roots]
        bfs_names = [
            (edge[0].name, edge[1].name) for edge in bfs_edges(nx_graph, source=_one(roots))
        ]
        assert bfs_names == [
            ("TestClass1 (object)", "test_get_referrer_graph.the_reference (local)"),
        ]
        # Check that (parts of) the graph are printed correctly
        string_buffer = io.StringIO()
        print(graph, file=string_buffer)
        output_string = string_buffer.getvalue()
        assert_in("╙── TestClass1 (object) (id=<ANY>) (target)", output_string)
        assert_in(
            "    └── test_get_referrer_graph.the_reference (local) (id=<ANY>)", output_string
        )

    def test_get_referrer_graph_with_cycle(self):
        # Create a cycle
        link1 = Link()
        link1_id = id(link1)
        link2 = Link()
        link1.next_link = link2
        link2.next_link = link1
        # Create a link from outside the cycle
        link3 = Link(link1)
        graph = referrers.get_referrer_graph(link1)

        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        # There should be no roots since the target node is in a cycle
        assert len(roots) == 0
        # Find the target object and check it is as expected
        targets = [
            node
            for node in nx_graph.nodes
            if isinstance(node, ReferrerGraphNode) and node.is_target
        ]
        assert len(targets) == 1
        target = _one(targets)
        assert target.name == "Link (object)"
        assert target.id == link1_id
        # Check that there is an edge from a next_link instance attribute back to the target
        edges_to_target = [(u, v) for u, v in nx_graph.edges if v == target]
        assert len(edges_to_target) == 1
        from_node, to_node = _one(edges_to_target)
        assert from_node.name == "Link.next_link (instance attribute)"
        assert to_node.name == "Link (object)"

        # Check that (parts of) the graph are printed correctly
        string_buffer = io.StringIO()
        print(graph, file=string_buffer)
        output_string = string_buffer.getvalue()
        assert_in("╙── Link (object) (id=<ANY>) (target)", output_string)
        assert_in(
            "    ├── test_get_referrer_graph_with_cycle.link1 (local) (id=<ANY>)",
            output_string,
        )
        assert_in("    ├── Link.next_link (instance attribute) (id=<ANY>)", output_string)
        assert_in("    │   └── Link (object) (id=<ANY>)", output_string)
        assert_in(
            "    │       ├── test_get_referrer_graph_with_cycle.link2 (local) (id=<ANY>)",
            output_string,
        )
        assert_in(
            "    │       └── Link.next_link (instance attribute) (id=<ANY>)", output_string
        )
        assert_in("    │           └── Link (object) (id=<ANY>) (cycle member)", output_string)
        assert_in("    └── Link.next_link (instance attribute) (id=<ANY>)", output_string)
        assert_in("        └── Link (object) (id=<ANY>)", output_string)
        assert_in(
            "            └── test_get_referrer_graph_with_cycle.link3 (local) (id=<ANY>)",
            output_string,
        )

    def test_get_referrer_graph_for_list(self):
        the_reference = TestClass1()
        the_reference2 = TestClass2(my_attribute=the_reference)
        graph = referrers.get_referrer_graph_for_list([the_reference, the_reference2])
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert {"TestClass1 (object)"} == set(root.name for root in roots)
        targets = [node for node in nx_graph.nodes if node.is_target]
        assert len(targets) == 2
        for target in targets:
            bfs_names = [
                (edge[0].name, edge[1].name) for edge in bfs_edges(nx_graph, source=target)
            ]
            if target.name == "TestClass1 (object)":
                expected_names = [
                    (
                        "TestClass1 (object)",
                        "TestClass2.my_attribute (instance attribute)",
                    ),
                    # For instance attributes we also include the containing object
                    # in the graph. This happens without doing anything special in Python
                    # versions < 3.11, and we do it explicitly in later Python versions.
                    (
                        "TestClass2.my_attribute (instance attribute)",
                        "TestClass2 (object)",
                    ),
                    (
                        "TestClass1 (object)",
                        "test_get_referrer_graph_for_list.the_reference (local)",
                    ),
                    # We can also find TestClass2 from the TestClass1 target, as there
                    # is a cycle in the graph.
                    (
                        "TestClass2 (object)",
                        "test_get_referrer_graph_for_list.the_reference2 (local)",
                    ),
                ]
                assert set(bfs_names) == set(expected_names)
            elif target.name == "TestClass2 (object)":
                assert bfs_names == [
                    (
                        "TestClass2 (object)",
                        "test_get_referrer_graph_for_list.the_reference2 (local)",
                    ),
                ]
            else:
                raise AssertionError(f"Unexpected target: {target}")

    def test_passed_object_excluded(self):
        # Check that we exclude our internal reference to the target object from
        # the graph.
        the_reference = TestClass1()
        graph = referrers.get_referrer_graph(the_reference)
        for node in graph.to_networkx().reverse().nodes:
            assert "target_object" not in node.name

    def test_graph_builder_excluded(self):
        the_reference = TestClass1()
        graph = referrers.get_referrer_graph(the_reference)
        for node in graph.to_networkx().reverse().nodes:
            assert "_ReferrerGraphBuilder" not in node.name

    def test_max_depth(self):
        link = Link(None)
        original_link = link
        for i in range(30):
            link = Link(link)
        graph = referrers.get_referrer_graph(
            original_link,
            max_depth=12,
        )
        nx_graph = graph.to_networkx().reverse()
        # There are two levels in the graph for each link: the Link object and the instance
        # attribute that points at the next link.
        instance_attribute_nodes = [
            node for node in nx_graph.nodes if "instance attribute" in node.name
        ]
        link_nodes = [node for node in nx_graph.nodes if "Link (object)" in node.name]
        assert len(instance_attribute_nodes) + len(link_nodes) == 13
        assert "Maximum depth of 12 exceeded" in str(graph)

    def test_untracked_container_object(self):
        # In this case the_dict is not tracked by the garbage collector because it
        # is a container containing only immutable objects (this is a CPython implementation
        # detail I think). However, the implementation should still be able to find the
        # reference to the_dict in the graph as a local variable.
        the_dict = {"a": "hello"}
        assert not gc.is_tracked(the_dict)
        assert len(gc.get_referrers(the_dict)) == 0
        graph = referrers.get_referrer_graph(the_dict)
        assert the_dict["a"] == "hello"
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any(
            "test_untracked_container_object.the_dict" in node_name for node_name in node_names
        ), graph

    def test_untracked_object_within_container(self):
        # In this case "hello" is not tracked by the garbage collector and the_dict
        # is not tracked because it is a container containing only immutable objects.
        # However, the implementation should still be able to find the reference to "hello"
        # in the graph because it treats untracked objects as a special case, and searches
        # for them in the referrents of locals and globals.
        the_dict = {"a": "hello"}
        assert not gc.is_tracked(the_dict)
        assert len(gc.get_referrers(the_dict)) == 0
        assert not gc.is_tracked(the_dict["a"])
        assert len(gc.get_referrers(the_dict["a"])) == 0
        graph = referrers.get_referrer_graph(the_dict["a"])
        assert the_dict["a"] == "hello"
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any(
            "test_untracked_object_within_container.the_dict" in node_name
            for node_name in node_names
        ), str(graph)

    def test_untracked_object_within_tracked_tuple(self):
        my_var = "hello"
        hello_tuple = (my_var,)
        assert not gc.is_tracked(my_var)
        assert gc.is_tracked(hello_tuple)
        graph = referrers.get_referrer_graph(my_var)
        assert hello_tuple[0] == "hello"
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any(
            "test_untracked_object_within_tracked_tuple.my_var" in node_name
            for node_name in node_names
        ), str(graph)

    def test_untracked_object_within_object(self):
        # In this case "hello" is not tracked by the garbage collector but the object that
        # references it is. However, the object's dict is not tracked, so the implementation
        # needs to search the object's referrents to find the reference to "hello".
        gc.collect()
        builders = [o for o in gc.get_objects() if isinstance(o, _ReferrerGraphBuilder)]
        assert len(builders) == 0, builders
        the_obj = A("hello")
        assert gc.is_tracked(the_obj)
        assert not gc.is_tracked(the_obj.__dict__)
        assert not gc.is_tracked(the_obj.instance_var)
        graph = referrers.get_referrer_graph(
            the_obj.instance_var,
        )
        assert the_obj.instance_var == "hello"
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any(
            "test_untracked_object_within_object.the_obj" in node_name
            for node_name in node_names
        ), str(graph)
        assert any("A (object)" in node_name for node_name in node_names), str(graph)
        assert any(".instance_var" in node_name for node_name in node_names), str(graph)

    def test_get_referrer_graph_for_unimported_module_with_explicit_module_prefix(self):
        # The module module1 is not imported, but has been loaded by the test runner.
        # It defines module_variable = 178.
        graph = referrers.get_referrer_graph(178, module_prefixes=["tests"])
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["int (object)"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any(
            "tests.testing_modules.module1.module_variable (module variable)" in node_name
            for node_name in node_names
        ), str(graph)
        # We don't attempt to find referrers from the module object itself, so
        # there should be no outgoing edges from the module node.
        module_nodes = [
            node
            for node in graph.to_networkx().reverse().nodes
            if node.name == "module (object)"
        ]
        for module_node in module_nodes:
            assert nx_graph.out_degree(module_node) == 0

    def test_get_referrer_graph_for_unimported_module_without_explicit_module_prefix(
        self,
    ):
        # The module module1 is not imported, but has been loaded by the test runner.
        # It defines module_variable = 178.
        graph = referrers.get_referrer_graph(178)
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["int (object)"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any(
            "tests.testing_modules.module1.module_variable (module variable)" in node_name
            for node_name in node_names
        ), str(graph)
        # We don't attempt to find referrers from the module object itself, so
        # there should be no outgoing edges from the module node.
        module_nodes = [
            node
            for node in graph.to_networkx().reverse().nodes
            if node.name == "module (object)"
        ]
        for module_node in module_nodes:
            assert nx_graph.out_degree(module_node) == 0

    def test_get_referrer_graph_for_unimported_module_referencing_untracked_object(
        self,
    ):
        # The module module1 is not imported, but has been loaded by the test runner.
        # It defines a class called IntContainer with an instance attribute that references
        # 145.
        graph = referrers.get_referrer_graph(145)
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["int (object)"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any(
            ".int_container_value (instance attribute)" in node_name
            for node_name in node_names
        ), str(graph)
        # We don't attempt to find referrers from the module object itself, so
        # there should be no outgoing edges from the module node.
        module_nodes = [
            node
            for node in graph.to_networkx().reverse().nodes
            if node.name == "module (object)"
        ]
        for module_node in module_nodes:
            assert nx_graph.out_degree(module_node) == 0

    def test_get_referrer_graph_for_closure(
        self,
    ):
        the_obj = TestClass1()
        closure = get_print_input_closure(the_obj)
        graph = referrers.get_referrer_graph(the_obj)
        assert closure is not None
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["TestClass1 (object)"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        # The closure name should be in the graph
        assert any(
            "get_print_input_closure.<locals>.print_input.input_val (closure)" in node_name
            for node_name in node_names
        ), str(graph)

    def test_get_referrer_graph_for_closure_referenced_by_object(
        self,
    ):
        the_obj = TestClass1()
        closure = get_print_input_closure(the_obj)
        closure_holder = ClosureHolder(closure)
        graph = referrers.get_referrer_graph(the_obj)
        assert closure is not None
        assert closure_holder.closure is closure
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["TestClass1 (object)"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        # The closure_holder variable should be in the graph
        assert any(
            "test_get_referrer_graph_for_closure_referenced_by_object.closure_holder"
            in node_name
            for node_name in node_names
        ), str(graph)

    def test_get_referrer_graph_for_module_level_closure(
        self,
    ):
        graph = referrers.get_referrer_graph(module_level_variable)
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["TestClass1 (object)"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        # The closure name should be in the graph
        assert any(
            "get_print_input_closure.<locals>.print_input.input_val (closure)" in node_name
            for node_name in node_names
        ), str(graph)

    def test_with_nested_tuples(
        self,
    ):
        nested_tuples = _get_nested_tuples()
        graph = referrers.get_referrer_graph(nested_tuples[0][0][0][0])
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["str (object)"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert sum("tuple index=0" in node_name for node_name in node_names) == 4, str(graph)

    def test_get_referrer_graph_with_timeout(self):
        the_reference = TestClass1()
        graph = referrers.get_referrer_graph(the_reference, timeout=0.0)
        # The graph should have two nodes: one for the target object, and one
        # telling us we've timed-out.
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert len(node_names) == 2
        assert any("TestClass1 (object)" in node_name for node_name in node_names)
        assert any("Timeout of 0.00 seconds exceeded" in node_name for node_name in node_names)

    def test_single_object_referrer_limit(self):
        myobj = TestClass1()
        otherobjs = [TestClass2(myobj) for _ in range(100)]
        assert otherobjs is not None
        graph = referrers.get_referrer_graph(
            myobj, single_object_referrer_limit=90, exclude_object_ids=[id(otherobjs)]
        )
        assert "Referrer limit of 90 exceeded" in str(graph)

    def test_single_object_referrer_limit_with_default(self):
        myobj = TestClass1()
        otherobjs = [TestClass2(myobj) for _ in range(120)]
        assert otherobjs is not None
        graph = referrers.get_referrer_graph(myobj, exclude_object_ids=[id(otherobjs)])
        assert "Referrer limit of 100 exceeded" in str(graph)

    def test_single_object_referrer_limit_with_no_limit(self):
        myobj = TestClass1()
        otherobjs = [TestClass2(myobj) for _ in range(120)]
        assert otherobjs is not None
        graph = referrers.get_referrer_graph(
            myobj, single_object_referrer_limit=None, exclude_object_ids=[id(otherobjs)]
        )
        assert str(graph).count("TestClass2.my_attribute") == 120

    @pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python >= 3.12")
    @pytest.mark.skipif(sys.version_info > (3, 12, 3), reason="requires python <= 3.12.3")
    def test_single_object_referrer_limit_with_immortal_object(self):
        # In early versions of Python 3.12 immortal objects have a very high reference
        # count so we need to deal with this somehow.
        myvalue = "hello"
        assert sys.getrefcount(myvalue) > IMMORTAL_OBJECT_REFCOUNT
        graph = referrers.get_referrer_graph(myvalue)
        assert "Referrer limit of 100 exceeded" not in str(graph)
        assert "myvalue" in str(graph)

    def test_weakref_proxy_with_deleted_ref(self):
        def my_function():
            return 1

        _ = weakref.proxy(my_function)
        del my_function
        the_reference = TestClass1()
        graph = referrers.get_referrer_graph(the_reference)
        nx_graph = graph.to_networkx().reverse()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["TestClass1 (object)"] == [root.name for root in roots]
        bfs_names = [
            (edge[0].name, edge[1].name) for edge in bfs_edges(nx_graph, source=_one(roots))
        ]
        assert bfs_names == [
            (
                "TestClass1 (object)",
                "test_weakref_proxy_with_deleted_ref.the_reference (local)",
            ),
        ]

    def test_class_attribute(self):
        held_instance = HeldClass(a=23)
        ClassAttributeHolder.class_attr = held_instance
        graph = referrers.get_referrer_graph(held_instance)
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any("ClassAttributeHolder" in node_name for node_name in node_names), str(graph)
        assert any("dict key=class_attr" in node_name for node_name in node_names), str(graph)

    def test_class_attribute_in_instance(self):
        held_instance = HeldClass(a=23)
        holder = ClassAttributeHolder()
        holder.class_attr = held_instance
        graph = referrers.get_referrer_graph(held_instance)
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any(
            "ClassAttributeHolder.class_attr" in node_name for node_name in node_names
        ), str(graph)
        assert any(
            "test_class_attribute_in_instance.holder (local)" in node_name
            for node_name in node_names
        ), str(graph)

    def test_multi_attribute_holder(self):
        my_int = 42238423948239842934
        held_instance = HeldClass(my_int)
        holder1 = InstanceAttributeHolder(held_instance)
        holder2 = InstanceAttributeHolder(held_instance)
        multi_holder = MultiAttributeHolder(holder1, holder2)
        graph = referrers.get_referrer_graph(my_int)
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        # Check that both attributes of the multi-holder are in the graph
        assert any("MultiAttributeHolder.attr1" in node_name for node_name in node_names), str(
            graph
        )
        assert any("MultiAttributeHolder.attr2" in node_name for node_name in node_names), str(
            graph
        )

    def test_regular_dataclass(self):
        my_dataclass = PythonDataclass(id=98237948729348)
        graph = referrers.get_referrer_graph(98237948729348)
        print(graph)
        node_names = [node.name for node in graph.to_networkx().reverse().nodes]
        assert any("int (object)" in node_name for node_name in node_names), str(graph)
        assert any("PythonDataclass.id" in node_name for node_name in node_names), str(graph)
        assert any(
            "test_regular_dataclass.my_dataclass" in node_name for node_name in node_names
        ), str(graph)
        assert any("PythonDataclass (object)" in node_name for node_name in node_names), str(
            graph
        )

    def test_list_holder(self):
        my_int = 4223842938472938479
        held_instance1 = HeldClass(my_int)
        held_instance2 = HeldClass(my_int)
        held_instance3 = HeldClass(my_int)
        list_holder = ListHolder(the_list=[held_instance1, held_instance2, held_instance3])
        graph = referrers.get_referrer_graph(my_int)

        nx_graph = graph.to_networkx().reverse()

        targets = [node for node in nx_graph.nodes if node.is_target]
        assert len(targets) == 1
        target = _one(targets)
        assert "int (object)" == target.name

        bfs_names = [
            (f"{edge[0].name}: {edge[0].id}", f"{edge[1].name}: {edge[1].id}")
            for edge in bfs_edges(nx_graph, source=target)
        ]

        assert bfs_names == [
            (f"int (object): {id(my_int)}", f"test_list_holder.my_int (local): {id(my_int)}"),
            (
                f"int (object): {id(my_int)}",
                f"HeldClass._a (instance attribute): {id(held_instance1)}",
            ),
            (
                f"int (object): {id(my_int)}",
                f"HeldClass._a (instance attribute): {id(held_instance2)}",
            ),
            (
                f"int (object): {id(my_int)}",
                f"HeldClass._a (instance attribute): {id(held_instance3)}",
            ),
            (
                f"HeldClass._a (instance attribute): {id(held_instance1)}",
                f"HeldClass (object): {id(held_instance1)}",
            ),
            (
                f"HeldClass._a (instance attribute): {id(held_instance2)}",
                f"HeldClass (object): {id(held_instance2)}",
            ),
            (
                f"HeldClass._a (instance attribute): {id(held_instance3)}",
                f"HeldClass (object): {id(held_instance3)}",
            ),
            (
                f"HeldClass (object): {id(held_instance1)}",
                f"test_list_holder.held_instance1 (local): {id(held_instance1)}",
            ),
            (
                f"HeldClass (object): {id(held_instance1)}",
                f"list index=0: {id(list_holder.the_list)}",
            ),
            (
                f"HeldClass (object): {id(held_instance2)}",
                f"test_list_holder.held_instance2 (local): {id(held_instance2)}",
            ),
            (
                f"HeldClass (object): {id(held_instance2)}",
                f"list index=1: {id(list_holder.the_list)}",
            ),
            (
                f"HeldClass (object): {id(held_instance3)}",
                f"test_list_holder.held_instance3 (local): {id(held_instance3)}",
            ),
            (
                f"HeldClass (object): {id(held_instance3)}",
                f"list index=2: {id(list_holder.the_list)}",
            ),
            (
                f"list index=0: {id(list_holder.the_list)}",
                f"list (object): {id(list_holder.the_list)}",
            ),
            (
                f"list (object): {id(list_holder.the_list)}",
                f"ListHolder.the_list (instance attribute): {id(list_holder)}",
            ),
            (
                f"ListHolder.the_list (instance attribute): {id(list_holder)}",
                f"ListHolder (object): {id(list_holder)}",
            ),
            (
                f"ListHolder (object): {id(list_holder)}",
                f"test_list_holder.list_holder (local): {id(list_holder)}",
            ),
        ]

    def test_set_holder(self):
        my_int1 = 4223842938472938479
        my_int2 = 9834779238479823749
        set_holder = SetHolder(the_set={my_int1, my_int2})
        graph = referrers.get_referrer_graph(my_int1)

        nx_graph = graph.to_networkx().reverse()

        targets = [node for node in nx_graph.nodes if node.is_target]
        assert len(targets) == 1
        target = _one(targets)
        assert "int (object)" == target.name

        node_names = [node.name for node in graph.to_networkx().reverse().nodes]

        # Both the set itself, and the ints membership of the set should be represented
        # as nodes in the graph.
        assert any("set member" in node_name for node_name in node_names), str(graph)
        assert any("set (object)" in node_name for node_name in node_names), str(graph)

    def test_get_referrer_graph_with_html_entity(self):
        # This is an HTML entity. A module that lists HTML entities used to cause us problems
        # before we ignored modules.
        a = "zwnj;"
        graph = referrers.get_referrer_graph(a)
        assert (
            str(graph)
            == f"""
╙── str (object) (id={id(a)}) (target)
    └── test_get_referrer_graph_with_html_entity.a (local) (id={id(a)}) (root)"""
        )

    def test_get_referrer_graph_with_lru_cache(self):
        all_results = []
        for _ in range(16):
            all_results.append(_function_with_lru_cache(random.randint(0, 10000000000000000)))
        graph = referrers.get_referrer_graph(all_results[10])
        # Just check that we don't get loads of extra stuff when we use an LRU cache
        assert len(graph.to_networkx().nodes) == 4

    def test_referenced_method(self):
        with_data = ClassWithDataAndMethod(23948293487293847923)
        with_ref = ClassWithMethodReference(with_data.my_method)
        graph = referrers.get_referrer_graph(with_data.data)

        print(graph)

        nx_graph = graph.to_networkx().reverse()

        targets = [node for node in nx_graph.nodes if node.is_target]
        assert len(targets) == 1
        target = _one(targets)
        assert "int (object)" == target.name

        bfs_names = [
            (f"{edge[0].name}", f"{edge[1].name}")
            for edge in bfs_edges(nx_graph, source=target)
        ]

        assert bfs_names == [
            ("int (object)", "ClassWithDataAndMethod.data (instance attribute)"),
            (
                "ClassWithDataAndMethod.data (instance attribute)",
                "ClassWithDataAndMethod (object)",
            ),
            (
                "ClassWithDataAndMethod (object)",
                "test_referenced_method.with_data (local)",
            ),
            ("ClassWithDataAndMethod (object)", "my_method method"),
            ("my_method method", "ClassWithMethodReference.method (instance attribute)"),
            (
                "ClassWithMethodReference.method (instance attribute)",
                "ClassWithMethodReference (object)",
            ),
            (
                "ClassWithMethodReference (object)",
                "test_referenced_method.with_ref (local)",
            ),
        ]

    def test_regression_on_converging_tree(self):
        """
        Test for a tree-like structure.

        This test checks the exact structure of the referrer graph, so is sensitive to
        small changes, so it's more of a regression test than a unit test.
        """
        (
            root,
            leaf_holder,
            id_root,
            id_level1_1,
            id_level1_2,
            id_level2_1,
            id_level2_2,
            id_level2_3,
            id_level2_4,
            id_leaf_holder,
        ) = _construct_mini_tree()
        graph = referrers.get_referrer_graph(root)

        assert (
            str(graph)
            == f"""
╙── TreeNode (object) (id={id_root}) (target)
    ├── test_regression_on_converging_tree.root (local) (id={id_root}) (root)
    ├── TreeNode.parent (instance attribute) (id={id_level1_1})
    │   └── TreeNode (object) (id={id_level1_1})
    │       ├── TreeNode.parent (instance attribute) (id={id_level2_1})
    │       │   └── TreeNode (object) (id={id_level2_1})
    │       │       └── TreeNodeHolder.node1 (instance attribute) (id={id_leaf_holder})
    │       │           └── TreeNodeHolder (object) (id={id_leaf_holder}) (cycle member)
    │       └── TreeNode.parent (instance attribute) (id={id_level2_2})
    │           └── TreeNode (object) (id={id_level2_2})
    │               └── TreeNodeHolder.node2 (instance attribute) (id={id_leaf_holder})
    │                   └── TreeNodeHolder (object) (id={id_leaf_holder}) (cycle member)
    └── TreeNode.parent (instance attribute) (id={id_level1_2})
        └── TreeNode (object) (id={id_level1_2})
            ├── TreeNode.parent (instance attribute) (id={id_level2_3})
            │   └── TreeNode (object) (id={id_level2_3})
            │       └── TreeNodeHolder.node3 (instance attribute) (id={id_leaf_holder})
            │           └── TreeNodeHolder (object) (id={id_leaf_holder}) (cycle member)
            └── TreeNode.parent (instance attribute) (id={id_level2_4})
                └── TreeNode (object) (id={id_level2_4})
                    └── TreeNodeHolder.node4 (instance attribute) (id={id_leaf_holder})
                        └── TreeNodeHolder (object) (id={id_leaf_holder})
                            └── test_regression_on_converging_tree.leaf_holder (local) (id={id_leaf_holder}) (root)"""
        )

        nx_graph = graph.to_networkx().reverse()

        targets = [node for node in nx_graph.nodes if node.is_target]
        assert len(targets) == 1
        target = _one(targets)
        assert "TreeNode (object)" == target.name

        bfs_names = [
            (f"{edge[0].name}: {edge[0].id}", f"{edge[1].name}: {edge[1].id}")
            for edge in bfs_edges(nx_graph, source=target)
        ]

        assert bfs_names == [
            (
                f"TreeNode (object): {id_root}",
                f"test_regression_on_converging_tree.root (local): {id_root}",
            ),
            (
                f"TreeNode (object): {id_root}",
                f"TreeNode.parent (instance attribute): {id_level1_1}",
            ),
            (
                f"TreeNode (object): {id_root}",
                f"TreeNode.parent (instance attribute): {id_level1_2}",
            ),
            (
                f"TreeNode.parent (instance attribute): {id_level1_1}",
                f"TreeNode (object): {id_level1_1}",
            ),
            (
                f"TreeNode.parent (instance attribute): {id_level1_2}",
                f"TreeNode (object): {id_level1_2}",
            ),
            (
                f"TreeNode (object): {id_level1_1}",
                f"TreeNode.parent (instance attribute): {id_level2_1}",
            ),
            (
                f"TreeNode (object): {id_level1_1}",
                f"TreeNode.parent (instance attribute): {id_level2_2}",
            ),
            (
                f"TreeNode (object): {id_level1_2}",
                f"TreeNode.parent (instance attribute): {id_level2_3}",
            ),
            (
                f"TreeNode (object): {id_level1_2}",
                f"TreeNode.parent (instance attribute): {id_level2_4}",
            ),
            (
                f"TreeNode.parent (instance attribute): {id_level2_1}",
                f"TreeNode (object): {id_level2_1}",
            ),
            (
                f"TreeNode.parent (instance attribute): {id_level2_2}",
                f"TreeNode (object): {id_level2_2}",
            ),
            (
                f"TreeNode.parent (instance attribute): {id_level2_3}",
                f"TreeNode (object): {id_level2_3}",
            ),
            (
                f"TreeNode.parent (instance attribute): {id_level2_4}",
                f"TreeNode (object): {id_level2_4}",
            ),
            (
                f"TreeNode (object): {id_level2_1}",
                f"TreeNodeHolder.node1 (instance attribute): {id_leaf_holder}",
            ),
            (
                f"TreeNode (object): {id_level2_2}",
                f"TreeNodeHolder.node2 (instance attribute): {id_leaf_holder}",
            ),
            (
                f"TreeNode (object): {id_level2_3}",
                f"TreeNodeHolder.node3 (instance attribute): {id_leaf_holder}",
            ),
            (
                f"TreeNode (object): {id_level2_4}",
                f"TreeNodeHolder.node4 (instance attribute): {id_leaf_holder}",
            ),
            (
                f"TreeNodeHolder.node1 (instance attribute): {id_leaf_holder}",
                f"TreeNodeHolder (object): {id_leaf_holder}",
            ),
            (
                f"TreeNodeHolder (object): {id_leaf_holder}",
                f"test_regression_on_converging_tree.leaf_holder (local): {id_leaf_holder}",
            ),
        ]


@lru_cache(maxsize=10)
def _function_with_lru_cache(input_val: int):
    return str(input_val)


def _construct_mini_tree() -> Tuple[
    TreeNode,
    TreeNodeHolder,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    root = TreeNode(parent=None)
    level1_1 = TreeNode(parent=root)
    level1_2 = TreeNode(parent=root)
    level2_1 = TreeNode(parent=level1_1)
    level2_2 = TreeNode(parent=level1_1)
    level2_3 = TreeNode(parent=level1_2)
    level2_4 = TreeNode(parent=level1_2)
    leaf_holder = TreeNodeHolder(
        node1=level2_1, node2=level2_2, node3=level2_3, node4=level2_4
    )
    return (
        root,
        leaf_holder,
        id(root),
        id(level1_1),
        id(level1_2),
        id(level2_1),
        id(level2_2),
        id(level2_3),
        id(level2_4),
        id(leaf_holder),
    )


def assert_in(substring: str, full_string: str):
    """
    Asserts that a substring is contained within another string, allowing for
    a wildcard token <ANY> in the substring.

    The <ANY> token matches any sequence of characters (including an empty one).
    """
    if not substring:
        return
    parts = [re.escape(part) for part in substring.split("<ANY>")]
    regex_pattern = ".*?".join(parts)
    match = re.search(regex_pattern, full_string, re.DOTALL)
    if not match:
        raise AssertionError(
            f"Pattern not found.\n"
            f"  Substring: '{substring}'\n"
            f"  Full Text: '{full_string}'"
        )
