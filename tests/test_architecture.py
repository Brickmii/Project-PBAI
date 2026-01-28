"""
Test the v2 architecture with unified Axes and Frames.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nodes import Node, SelfNode, Axis, Frame, Order, Element, assert_self_valid, reset_birth_for_testing
from core.manifold import Manifold, create_manifold


def test_basic_node():
    """Test basic node creation with frame."""
    print("\n=== Test: Basic Node ===")
    
    node = Node(concept="apple", position="nne")
    
    # Check 6 motion functions
    assert node.heat == 1.0
    assert node.polarity == 1
    assert node.existence == "actual"
    assert node.righteousness == 1.0
    assert node.order == 0
    
    # Check frame
    assert node.frame.origin == "apple"
    assert node.frame.axes == {}
    
    print(f"Created: {node}")
    print(f"Frame: {node.frame}")
    print("✓ Basic node works")


def test_unified_axes():
    """Test that spatial and semantic are both axes."""
    print("\n=== Test: Unified Axes ===")
    
    node = Node(concept="length", position="n")
    
    # Add spatial axis
    axis_n = node.add_axis("n", "north_neighbor_id")
    assert axis_n.is_spatial == True
    assert axis_n.is_semantic == False
    
    # Add semantic axis
    axis_measure = node.add_axis("measurement", "measurement_node_id")
    assert axis_measure.is_spatial == False
    assert axis_measure.is_semantic == True
    
    # Both in same dict
    assert len(node.frame.axes) == 2
    assert "n" in node.frame.axes
    assert "measurement" in node.frame.axes
    
    # Convenience accessors
    assert len(node.spatial_axes) == 1
    assert len(node.semantic_axes) == 1
    
    # Auto x/y assignment
    assert node.frame.x_axis == "n"
    assert node.frame.y_axis == "measurement"
    
    print(f"Node: {node}")
    print(f"Spatial axes: {list(node.spatial_axes.keys())}")
    print(f"Semantic axes: {list(node.semantic_axes.keys())}")
    print("✓ Unified axes work")


def test_order_on_axis():
    """Test Robinson arithmetic on axis (proper frame)."""
    print("\n=== Test: Order on Axis ===")
    
    node = Node(concept="numbers", position="u")
    
    # Create axis with order
    axis = node.add_axis("elements", "zero_node_id")
    assert axis.is_proper == False  # Not proper yet
    assert axis.capability == "righteous"
    
    # Make it proper (add Robinson arithmetic)
    order = axis.make_proper()
    assert axis.is_proper == True
    assert axis.capability == "ordered"
    
    # Add elements: 0, 1, 2
    elem0 = order.add_element("zero_id")
    elem1 = order.add_element("one_id")
    elem2 = order.add_element("two_id")
    
    assert elem0.index == 0
    assert elem1.index == 1
    assert elem2.index == 2
    
    # Robinson successor
    assert order.successor(0) == 1
    assert order.successor(1) == 2
    
    # Lookup
    assert order.get_index("one_id") == 1
    
    print(f"Order: {len(order)} elements")
    print(f"Elements: {[e.index for e in order.elements]}")
    print(f"Proper axes: {list(node.proper_axes.keys())}")
    print("✓ Order on axis works")


def test_nested_capability_hierarchy():
    """Test the full R→O→M→G capability hierarchy."""
    print("\n=== Test: Nested Capability Hierarchy ===")
    
    node = Node(concept="spectrum", position="n")
    axis = node.add_axis("colors", "colors_id")
    
    # Stage 1: RIGHTEOUS (just exists)
    assert axis.capability == "righteous"
    assert not axis.is_ordered
    assert not axis.is_movable
    assert not axis.is_graphic
    print(f"1. Righteous: {axis.capability}")
    
    # Stage 2: ORDERED (add Robinson arithmetic)
    order = axis.make_ordered()
    order.add_element("red_id")
    order.add_element("orange_id")
    order.add_element("yellow_id")
    
    assert axis.capability == "ordered"
    assert axis.is_ordered
    assert not axis.is_movable
    assert not axis.is_graphic
    print(f"2. Ordered: {axis.capability} ({len(order)} elements)")
    
    # Stage 3: MOVABLE (add transitions)
    movement = axis.make_movable()
    movement.add_transition(0, 1)  # red → orange
    movement.add_transition(1, 2)  # orange → yellow
    movement.add_transition(1, 0)  # orange → red (reverse)
    
    assert axis.capability == "movable"
    assert axis.is_ordered
    assert axis.is_movable
    assert not axis.is_graphic
    assert movement.can_move(0, 1)
    assert movement.can_move(1, 2)
    assert not movement.can_move(0, 2)  # No direct transition
    print(f"3. Movable: {axis.capability} ({len(movement.transitions)} transitions)")
    
    # Stage 4: GRAPHIC (add coordinates)
    graphic = axis.make_graphic()
    graphic.set_coordinates(0, 1.0, 0.0, 0.0)   # red
    graphic.set_coordinates(1, 0.5, 0.5, 0.0)   # orange
    graphic.set_coordinates(2, 1.0, 1.0, 0.0)   # yellow
    graphic.set_bounds((0, 1), (0, 1))
    
    assert axis.capability == "graphic"
    assert axis.is_ordered
    assert axis.is_movable
    assert axis.is_graphic
    
    coords = graphic.get_coordinates(1)
    assert coords == (0.5, 0.5, 0.0)
    print(f"4. Graphic: {axis.capability} ({len(graphic.coordinates)} coordinates)")
    
    # Verify nesting structure
    assert axis.order is not None
    assert axis.order.movement is not None
    assert axis.order.movement.graphic is not None
    
    print("✓ Nested capability hierarchy works")


def test_frame_plane():
    """Test that frame defines 2D plane with x and y axes."""
    print("\n=== Test: Frame Plane ===")
    
    node = Node(concept="temperature", position="ne")
    
    # Create x axis (identity)
    node.add_axis("self", "self_node_id")
    
    # Create y axis (values)
    values_axis = node.add_axis("values", "values_id")
    values_axis.make_proper()
    values_axis.order.add_element("cold_id")
    values_axis.order.add_element("warm_id")
    values_axis.order.add_element("hot_id")
    
    # Frame has 2D plane
    assert node.frame.x_axis == "self"
    assert node.frame.y_axis == "values"
    
    # Values axis is proper with order
    values = node.get_axis("values")
    assert values.is_proper
    assert len(values.order) == 3
    
    print(f"Frame: origin={node.frame.origin}")
    print(f"  x_axis: {node.frame.x_axis}")
    print(f"  y_axis: {node.frame.y_axis}")
    print(f"  Values order: cold(0) → warm(1) → hot(2)")
    print("✓ Frame plane works")


def test_axis_polarity():
    """Test polarity on axes (+/- on each axis)."""
    print("\n=== Test: Axis Polarity ===")
    
    node = Node(concept="temperature", position="n")
    
    # Positive pole
    hot_axis = node.add_axis("hot", "hot_id", polarity=1)
    assert hot_axis.polarity == 1
    
    # Negative pole
    cold_axis = node.add_axis("cold", "cold_id", polarity=-1)
    assert cold_axis.polarity == -1
    
    print(f"Hot axis polarity: {hot_axis.polarity}")
    print(f"Cold axis polarity: {cold_axis.polarity}")
    print("✓ Axis polarity works")


def test_axis_strengthening():
    """Test that repeated traversal strengthens axis (warping)."""
    print("\n=== Test: Axis Strengthening ===")
    
    node = Node(concept="test", position="n")
    
    # First creation
    axis = node.add_axis("link", "target_id")
    assert axis.traversal_count == 1
    assert axis.strength <= 0.5  # count=1 gives exactly 0.5
    
    # Strengthen (same direction + target)
    axis2 = node.add_axis("link", "target_id")
    assert axis2.traversal_count == 2
    assert axis2 is axis  # Same object
    
    # Strengthen more
    for _ in range(10):
        node.add_axis("link", "target_id")
    
    assert axis.traversal_count == 12
    assert axis.strength > 0.9
    
    print(f"Traversal count: {axis.traversal_count}")
    print(f"Strength: {axis.strength:.3f}")
    print("✓ Axis strengthening works")


def test_manifold_bootstrap():
    """Test manifold bootstrap with v2 structure."""
    print("\n=== Test: Manifold Bootstrap ===")
    
    # Reset birth state for this test
    reset_birth_for_testing()
    
    manifold = Manifold()
    manifold.bootstrap()
    
    # Self exists
    assert manifold.self_node is not None
    assert_self_valid(manifold.self_node)
    
    # 9 nodes total: 6 physical (n,s,e,w,u,d) + 3 psychology (identity, ego, conscience)
    assert len(manifold.nodes) == 9, f"Expected 9 nodes, got {len(manifold.nodes)}"
    
    # Self has 9 axes: 6 spatial + 3 semantic (identity, ego, conscience)
    assert len(manifold.self_node.frame.axes) == 9, f"Expected 9 axes, got {len(manifold.self_node.frame.axes)}"
    
    # Self's righteous frame is set
    assert manifold.self_node.frame.x_axis == "identity"
    assert manifold.self_node.frame.y_axis == "ego"
    assert manifold.self_node.frame.z_axis == "conscience"
    
    # Psychology nodes exist with trig positions
    assert manifold.identity_node is not None
    assert manifold.ego_node is not None
    assert manifold.conscience_node is not None
    
    assert manifold.identity_node.trig_position is not None
    assert manifold.ego_node.trig_position is not None
    assert manifold.conscience_node.trig_position is not None
    
    # Check Freudian heat distribution (approximately)
    total_psy_heat = manifold.identity_node.heat + manifold.ego_node.heat + manifold.conscience_node.heat
    identity_ratio = manifold.identity_node.heat / total_psy_heat
    ego_ratio = manifold.ego_node.heat / total_psy_heat
    conscience_ratio = manifold.conscience_node.heat / total_psy_heat
    
    assert abs(identity_ratio - 0.70) < 0.01, f"Identity ratio should be ~70%, got {identity_ratio*100:.1f}%"
    assert abs(ego_ratio - 0.10) < 0.01, f"Ego ratio should be ~10%, got {ego_ratio*100:.1f}%"
    assert abs(conscience_ratio - 0.20) < 0.01, f"Conscience ratio should be ~20%, got {conscience_ratio*100:.1f}%"
    
    # Each bootstrap node has axis back to Self
    for node in manifold.nodes.values():
        if node.concept.startswith("bootstrap_"):
            # Should have at least one axis (back to Self)
            assert len(node.frame.axes) >= 1
            # One of them should point to Self
            back_to_self = any(a.target_id == manifold.self_node.id for a in node.frame.axes.values())
            assert back_to_self, f"Node {node.concept} doesn't connect back to Self"
    
    print(manifold.visualize())
    print("✓ Manifold bootstrap works")


def test_warp_factor():
    """Test that axis strength affects warping."""
    print("\n=== Test: Warp Factor ===")
    
    # Reset birth state for this test
    reset_birth_for_testing()
    
    manifold = Manifold()
    manifold.bootstrap()
    
    # Get a bootstrap node and strengthen its connection
    north_node = manifold.get_node_by_concept("bootstrap_n")
    assert north_node is not None
    
    # Get axis back to self
    back_axis = None
    for axis in north_node.frame.axes.values():
        if axis.target_id == manifold.self_node.id:
            back_axis = axis
            break
    
    assert back_axis is not None
    
    # Initial warp factor
    warp1 = manifold.calculate_warp_factor(back_axis)
    print(f"Initial warp factor: {warp1:.3f}")
    
    # Strengthen the axis
    for _ in range(20):
        back_axis.strengthen()
    
    # New warp factor should be higher
    warp2 = manifold.calculate_warp_factor(back_axis)
    print(f"After strengthening: {warp2:.3f}")
    
    assert warp2 > warp1
    print("✓ Warp factor works")


def test_backward_compatibility():
    """Test conceive() and connect() aliases."""
    print("\n=== Test: Backward Compatibility ===")
    
    node = Node(concept="test", position="n")
    
    # Old style: conceive (semantic)
    axis1 = node.conceive("name", "name_id", polarity=1)
    assert axis1.is_semantic
    
    # Old style: connect (spatial)
    axis2 = node.connect("e", "east_id")
    assert axis2.is_spatial
    
    # Old style: get_conception
    conc = node.get_conception("name")
    assert conc is not None
    assert conc.target_id == "name_id"
    
    # Old style: get_connection
    conn = node.get_connection("e")
    assert conn is not None
    assert conn.target_id == "east_id"
    
    print("✓ Backward compatibility works")


def test_serialization():
    """Test save/load of v2 structure."""
    print("\n=== Test: Serialization ===")
    
    # Create node with complex frame
    node = Node(concept="complex", position="nne")
    node.add_axis("n", "north_id")
    axis = node.add_axis("ordered_thing", "thing_id")
    axis.make_proper()
    axis.order.add_element("elem1")
    axis.order.add_element("elem2")
    
    # Serialize
    data = node.to_dict()
    
    # Deserialize
    node2 = Node.from_dict(data)
    
    # Verify
    assert node2.concept == "complex"
    assert node2.position == "nne"
    assert len(node2.frame.axes) == 2
    
    restored_axis = node2.get_axis("ordered_thing")
    assert restored_axis.is_proper
    assert len(restored_axis.order) == 2
    
    print(f"Original: {node}")
    print(f"Restored: {node2}")
    print("✓ Serialization works")


def test_serialization_full_hierarchy():
    """Test serialization of full R→O→M→G hierarchy."""
    print("\n=== Test: Serialization Full Hierarchy ===")
    
    node = Node(concept="color_wheel", position="u")
    axis = node.add_axis("spectrum", "spectrum_id")
    
    # Build full hierarchy
    order = axis.make_ordered()
    order.add_element("red")
    order.add_element("green")
    order.add_element("blue")
    
    movement = axis.make_movable()
    movement.add_transition(0, 1, cost=0.5)
    movement.add_transition(1, 2, cost=0.5)
    movement.add_transition(2, 0, cost=0.5)
    
    graphic = axis.make_graphic()
    graphic.set_coordinates(0, 1.0, 0.0, 0.0)
    graphic.set_coordinates(1, 0.0, 1.0, 0.0)
    graphic.set_coordinates(2, 0.0, 0.0, 1.0)
    graphic.set_bounds((0, 1), (0, 1), (0, 1))
    
    assert axis.capability == "graphic"
    
    # Serialize
    data = node.to_dict()
    
    # Deserialize
    node2 = Node.from_dict(data)
    axis2 = node2.get_axis("spectrum")
    
    # Verify full hierarchy restored
    assert axis2.capability == "graphic"
    assert axis2.is_ordered
    assert axis2.is_movable
    assert axis2.is_graphic
    
    assert len(axis2.order) == 3
    assert len(axis2.order.movement.transitions) == 3
    assert len(axis2.order.movement.graphic.coordinates) == 3
    
    # Check specific values
    assert axis2.order.movement.can_move(0, 1)
    assert axis2.order.movement.get_transition(0, 1).cost == 0.5
    assert axis2.order.movement.graphic.get_coordinates(0) == (1.0, 0.0, 0.0)
    
    print(f"Axis capability: {axis2.capability}")
    print(f"Elements: {len(axis2.order)}")
    print(f"Transitions: {len(axis2.order.movement.transitions)}")
    print(f"Coordinates: {len(axis2.order.movement.graphic.coordinates)}")
    print("✓ Full hierarchy serialization works")


def test_full_scenario():
    """Test a realistic scenario: temperature with measurements."""
    print("\n=== Test: Full Scenario (Temperature) ===")
    
    # Reset birth state for this test
    reset_birth_for_testing()
    
    manifold = Manifold()
    manifold.bootstrap()
    
    # Create "temperature" concept
    temp_node = Node(
        concept="temperature",
        position="nnu",  # Abstract concept
        heat=5.0,
        righteousness=0.0
    )
    manifold.add_node(temp_node)
    
    # Connect to Self
    manifold.self_node.add_axis("temperature", temp_node.id)
    temp_node.add_axis("self", manifold.self_node.id)
    
    # Create semantic axis for "hot/cold" (bounded, semantic)
    semantic_axis = temp_node.add_axis("feeling", "feeling_frame_id", polarity=1)
    # Not proper - just semantic (short/medium/long style)
    
    # Create proper axis for degrees (with Robinson arithmetic)
    degrees_axis = temp_node.add_axis("degrees", "degrees_frame_id")
    degrees_axis.make_proper()
    
    # Add some temperature values
    for i in range(101):  # 0-100 degrees
        degrees_axis.order.add_element(f"degree_{i}")
    
    # Verify structure
    # x_axis = "self" (first added - connection back to Self)
    # y_axis = "feeling" (second added)
    assert temp_node.frame.x_axis == "self"
    assert temp_node.frame.y_axis == "feeling"
    
    assert len(temp_node.semantic_axes) == 3  # self + feeling + degrees
    assert len(temp_node.proper_axes) == 1    # degrees only
    
    print(f"Temperature node: {temp_node}")
    print(f"Frame: {temp_node.frame}")
    print(f"Degrees axis has {len(degrees_axis.order)} ordered elements")
    print("✓ Full scenario works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("PBAI Thermal Manifold v2 Architecture Tests")
    print("=" * 60)
    
    test_basic_node()
    test_unified_axes()
    test_order_on_axis()
    test_nested_capability_hierarchy()
    test_frame_plane()
    test_axis_polarity()
    test_axis_strengthening()
    test_manifold_bootstrap()
    test_warp_factor()
    test_backward_compatibility()
    test_serialization()
    test_serialization_full_hierarchy()
    test_full_scenario()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
