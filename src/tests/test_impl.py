import dataclasses
import gc
import sys
from time import sleep
import threading
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

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
        assert names == {f"TestClass2.my_attribute (instance attribute)"}
        assert containing_class.my_attribute is local_ref

    def test_instance_attribute_frozen_dataclass(self):
        local_ref = TestClass1()
        containing_class = TestClass2Frozen(local_ref)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        assert names == {f"TestClass2Frozen.let_it_go (instance attribute)"}
        assert containing_class.let_it_go is local_ref

    def test_instance_attribute_changed(self):
        local_ref = TestClass1()
        # Construct the containing class with a different object and then change it.
        containing_class = TestClass2(TestClass1())
        containing_class.my_attribute = local_ref
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        assert names == {f"TestClass2.my_attribute (instance attribute)"}
        assert containing_class.my_attribute is local_ref

    def test_multiple_instance_attributes_in_same_class(self):
        local_ref = TestClass1()
        containing_class = TestClass3(local_ref, local_ref)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        assert names == {
            f"TestClass3.my_attribute (instance attribute)",
            f"TestClass3._my_attribute2 (instance attribute)",
        }
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
                assert names == {f"TestClass2.my_attribute (instance attribute)"}
            elif (
                referrer is containing_class2.__dict__ or referrer is containing_class2
            ):
                assert names == {f"TestClass2.my_attribute (instance attribute)"}
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
        assert names == {f"dict[mykey]"}
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
                assert names == {f"dict[mykey]"}
            elif referrer is containing_class.__dict__ or referrer is containing_class:
                assert names == {f"TestClass2.my_attribute (instance attribute)"}
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
        assert names == {
            f"TestClass2.my_attribute (instance attribute)",
        }
        assert containing_class.my_attribute is local_ref
        assert dict_container.my_dict is containing_class.__dict__

    def test_list(self):
        local_ref = TestClass1()
        my_list = [1, local_ref, 3]
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            local_ref, _one(gc.get_referrers(local_ref))
        )
        assert names == {f"list[1]"}
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
                assert names == {f"list[1]"}
            elif referrer is containing_class.__dict__ or referrer is containing_class:
                assert names == {f"TestClass2.my_attribute (instance attribute)"}
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
        assert names == {f"set (object)"}
        assert local_ref in my_set

    def test_class_dict(self):
        local_ref = TestClass1()
        containing_class = TestClass2(local_ref)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            containing_class.__dict__, _one(gc.get_referrers(containing_class.__dict__))
        )
        assert names == {f"TestClass2 (object)"}
        assert containing_class.my_attribute is local_ref

    def test_outer_container(self):
        local_ref = TestClass1()
        containing_class = TestClass2(local_ref)
        outer_class = TestClass2Container(containing_class)
        names = ObjectNameFinder(single_object_referrer_limit=None).get_names(
            containing_class, _one(gc.get_referrers(containing_class))
        )
        assert names == {
            f"TestClass2Container.contained_attribute (instance attribute)"
        }
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
        assert names == {f"DictWithoutItems (object)"}
        assert my_dict["mykey"] is local_ref

    # This test is only valid for Python versions <=3.10 as newer versions use
    # a different way of finding object references (avoiding the referrers of grandparents).
    @pytest.mark.skipif(sys.version_info > (3, 10), reason="requires python <= 3.10")
    def test_single_object_referrer_limit(self):
        myobj = TestClass1()
        otherobjs = [TestClass2(myobj) for _ in range(100)]
        assert otherobjs is not None
        names = ObjectNameFinder(single_object_referrer_limit=50).get_names(
            myobj, next(iter(gc.get_referrers(myobj)))
        )
        assert any("Referrer limit of 50 exceeded" in name for name in list(names))


class TestModuleLevelNameFinder:
    def test_variable_from_non_imported_module(self):
        # Here we're testing the case where the module containing the variable has not been
        # imported directly, and is therefore not in globals. We rely on conftest.py to import
        # the module (pytest does this automatically). We also rely on the fact that 178 is
        # interned, so we can provide exactly the same object as module_variable refers to
        # without importing it (we also rely on the fact that there are no other module-level
        # references to 178 in this program).
        names = ModuleLevelNameFinder("tests.testing_modules").get_names(178)
        assert names == {
            "tests.testing_modules.module1.module_variable (module variable)"
        }

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


class TestGetReferrerGraph:
    def test_get_referrer_graph(self):
        the_reference = TestClass1()
        graph = referrers.get_referrer_graph(the_reference)
        nx_graph = graph.to_networkx()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["TestClass1 instance"] == [root.name for root in roots]
        bfs_names = [
            (edge[0].name, edge[1].name)
            for edge in bfs_edges(nx_graph, source=_one(roots))
        ]
        assert bfs_names == [
            ("TestClass1 instance", "test_get_referrer_graph.the_reference (local)"),
        ]

    def test_get_referrer_graph_for_list(self):
        the_reference = TestClass1()
        the_reference2 = TestClass2(my_attribute=the_reference)
        graph = referrers.get_referrer_graph_for_list([the_reference, the_reference2])
        nx_graph = graph.to_networkx()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert {"TestClass1 instance", "TestClass2 instance"} == set(
            root.name for root in roots
        )
        for root in roots:
            bfs_names = [
                (edge[0].name, edge[1].name)
                for edge in bfs_edges(nx_graph, source=root)
            ]
            if root.name == "TestClass1 instance":
                expected_names = [
                    (
                        "TestClass1 instance",
                        "TestClass2.my_attribute (instance attribute)",
                    ),
                    (
                        "TestClass1 instance",
                        "test_get_referrer_graph_for_list.the_reference (local)",
                    ),
                ]
                # In Python 3.10 and earlier there will be an extra node in the graph.
                # This is the object that contains the instance attribute. In Python 3.11
                # this is not included in the graph.
                if sys.version_info[1] <= 10:
                    expected_names.append(
                        (
                            "TestClass2.my_attribute (instance attribute)",
                            "TestClass2 (object)",
                        )
                    )
                assert set(bfs_names) == set(expected_names)
            elif root.name == "TestClass2 instance":
                assert bfs_names == [
                    (
                        "TestClass2 instance",
                        "test_get_referrer_graph_for_list.the_reference2 (local)",
                    )
                ]
            else:
                raise AssertionError(f"Unexpected root: {root}")

    def test_passed_object_excluded(self):
        # Check that we exclude our internal reference to the target object from
        # the graph.
        the_reference = TestClass1()
        graph = referrers.get_referrer_graph(the_reference)
        for node in graph.to_networkx().nodes:
            assert "target_object" not in node.name

    def test_graph_builder_excluded(self):
        the_reference = TestClass1()
        graph = referrers.get_referrer_graph(the_reference)
        for node in graph.to_networkx().nodes:
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
        nx_graph = graph.to_networkx()
        # There are two levels in the graph for each link: the Link object and the instance
        # attribute that points at the next link.
        instance_attribute_nodes = [
            node for node in nx_graph.nodes if "instance attribute" in node.name
        ]
        link_nodes = [node for node in nx_graph.nodes if "Link (object)" in node.name]
        assert len(instance_attribute_nodes) + len(link_nodes) == 12
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
        node_names = [node.name for node in graph.to_networkx().nodes]
        assert any(
            "test_untracked_container_object.the_dict" in node_name
            for node_name in node_names
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
        node_names = [node.name for node in graph.to_networkx().nodes]
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
        node_names = [node.name for node in graph.to_networkx().nodes]
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
        node_names = [node.name for node in graph.to_networkx().nodes]
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
        nx_graph = graph.to_networkx()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["int instance"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().nodes]
        assert any(
            "tests.testing_modules.module1.module_variable (module variable)"
            in node_name
            for node_name in node_names
        ), str(graph)

    def test_get_referrer_graph_for_unimported_module_without_explicit_module_prefix(
        self,
    ):
        # The module module1 is not imported, but has been loaded by the test runner.
        # It defines module_variable = 178.
        graph = referrers.get_referrer_graph(178)
        nx_graph = graph.to_networkx()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["int instance"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().nodes]
        assert any(
            "tests.testing_modules.module1.module_variable (module variable)"
            in node_name
            for node_name in node_names
        ), str(graph)

    def test_get_referrer_graph_for_unimported_module_referencing_untracked_object(
        self,
    ):
        # The module module1 is not imported, but has been loaded by the test runner.
        # It defines a class called IntContainer with an instance attribute that references
        # 145.
        graph = referrers.get_referrer_graph(145)
        nx_graph = graph.to_networkx()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["int instance"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().nodes]
        assert any(
            ".int_container_value (instance attribute)" in node_name
            for node_name in node_names
        ), str(graph)

    def test_get_referrer_graph_for_closure(
        self,
    ):
        the_obj = TestClass1()
        closure = get_print_input_closure(the_obj)
        graph = referrers.get_referrer_graph(the_obj)
        assert closure is not None
        nx_graph = graph.to_networkx()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["TestClass1 instance"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().nodes]
        # The closure name should be in the graph
        assert any(
            "get_print_input_closure.<locals>.print_input.input_val (closure)"
            in node_name
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
        nx_graph = graph.to_networkx()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["TestClass1 instance"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().nodes]
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
        nx_graph = graph.to_networkx()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["TestClass1 instance"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().nodes]
        # The closure name should be in the graph
        assert any(
            "get_print_input_closure.<locals>.print_input.input_val (closure)"
            in node_name
            for node_name in node_names
        ), str(graph)

    def test_with_nested_tuples(
        self,
    ):
        nested_tuples = _get_nested_tuples()
        graph = referrers.get_referrer_graph(nested_tuples[0][0][0][0])
        nx_graph = graph.to_networkx()
        roots = [node for node in nx_graph.nodes if nx_graph.in_degree(node) == 0]
        assert ["str instance"] == [root.name for root in roots]
        node_names = [node.name for node in graph.to_networkx().nodes]
        assert sum("tuple[0]" in node_name for node_name in node_names) == 4, str(graph)

    def test_get_referrer_graph_with_timeout(self):
        the_reference = TestClass1()
        graph = referrers.get_referrer_graph(the_reference, timeout=0.0)
        # The graph should have two nodes: one for the target object, and one
        # telling us we've timed-out.
        node_names = [node.name for node in graph.to_networkx().nodes]
        assert len(node_names) == 2
        assert any("TestClass1 instance" in node_name for node_name in node_names)
        assert any(
            "Timeout of 0.00 seconds exceeded" in node_name for node_name in node_names
        )

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
    def test_single_object_referrer_limit_with_immortal_object(self):
        # In Python >= 3.12 immortal objects have a very high reference count so
        # we need to deal with this somehow.
        myvalue = "hello"
        assert sys.getrefcount(myvalue) > IMMORTAL_OBJECT_REFCOUNT
        graph = referrers.get_referrer_graph(myvalue)
        assert "Referrer limit of 100 exceeded" not in str(graph)
        assert "myvalue" in str(graph)
