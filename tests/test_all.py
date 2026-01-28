#!/usr/bin/env python3
"""
PBAI Thermal Manifold - Test Suite
Comprehensive tests for core components.
"""

import os
import sys
import json
import glob
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clean up any stale test files before imports
_test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_growth_dir = os.path.join(_test_dir, 'growth')
for f in glob.glob(os.path.join(_growth_dir, 'test_*.json')):
    try:
        os.remove(f)
    except:
        pass

from core import (
    # Constants
    MOVEMENT_CONSTANT, SIX, PHI, K, SELF_DIRECTIONS, ALL_DIRECTIONS, OPPOSITES,
    heat_required,
    
    # Compression
    compress, decompress, validate_position, get_axis_coordinates,
    positions_share_prefix, run_compression_tests,
    
    # Nodes
    Axis, Node, SelfNode, assert_self_valid,
    
    # Birth system
    reset_birth_for_testing,
    
    # Manifold
    Manifold, create_manifold,
    
    # Environment
    EnvironmentCore, create_environment_core, Perception, Action, ActionResult,
    MockDriver, NullPort,
)

from core.node_constants import (
    get_growth_path, K, PHI, SIX, MOVEMENT_CONSTANT,
    DIRECTIONS, DIRECTIONS_SELF, DIRECTIONS_UNIVERSAL,
    DIRECTIONS_RELATIVE, DIRECTIONS_ABSOLUTE,  # Aliases
    SELF_DIRECTIONS, OPPOSITES, heat_required
)


def create_test_manifold():
    """Create a fresh manifold for testing. Uses birth for full psychology."""
    reset_birth_for_testing()
    m = Manifold()
    m.birth()
    return m


def test_constants():
    """Test that constants are correct."""
    print("Testing constants...")
    
    assert MOVEMENT_CONSTANT == 12
    assert SIX == 6
    assert abs(PHI - 1.618033988749895) < 0.0001
    
    # K = 4/φ² = Σ(1/φⁿ) for n=1→6 ≈ 1.528
    assert abs(K - 1.527864) < 0.001, f"K should be ~1.528, got {K}"
    
    # THE FUNDAMENTAL IDENTITY: K × φ² = 4
    assert abs(K * PHI**2 - 4) < 1e-10, f"K × φ² must equal 4 (got {K * PHI**2})"
    
    # 12 = 6 × 2: 6 Self directions × 2 frames (Self + Universal)
    assert len(DIRECTIONS_SELF) == 6, "Should have 6 Self directions"
    assert len(DIRECTIONS_UNIVERSAL) == 6, "Should have 6 Universal directions"
    assert len(DIRECTIONS) == 12, "Should have 12 total directions"
    
    # Self directions for navigation
    assert 'up' in DIRECTIONS_SELF
    assert 'forward' in DIRECTIONS_SELF
    
    # Universal directions for locating righteous frames
    assert 'N' in DIRECTIONS_UNIVERSAL
    assert 'above' in DIRECTIONS_UNIVERSAL
    
    # Aliases work
    assert DIRECTIONS_RELATIVE == DIRECTIONS_SELF
    assert DIRECTIONS_ABSOLUTE == DIRECTIONS_UNIVERSAL
    
    # Legacy directions still work
    assert len(SELF_DIRECTIONS) == 5  # n, s, e, w, u (no d) - legacy
    assert 'd' not in SELF_DIRECTIONS
    
    # Opposites work for all 12 + legacy
    assert OPPOSITES['up'] == 'down'
    assert OPPOSITES['N'] == 'S'
    assert OPPOSITES['n'] == 's'  # Legacy
    
    # Heat required scales with phi
    assert heat_required(1) == K
    assert abs(heat_required(2) - (K * PHI)) < 0.0001
    
    print("  Constants: PASSED ✓")


def test_compression():
    """Test position compression/decompression."""
    print("Testing compression...")
    
    # Use built-in tests
    assert run_compression_tests()
    
    # Additional tests
    assert compress("nnnee") == "n(3)e(2)"
    assert decompress("n(3)e(2)") == "nnnee"
    
    # Roundtrip
    test_pos = "nnnnssseeeewwwwwuuudd"
    assert decompress(compress(test_pos)) == test_pos
    
    # Axis coordinates
    coords = get_axis_coordinates("nnwwu")
    assert coords['n'] == 2
    assert coords['w'] == 2
    assert coords['u'] == 1
    assert coords['s'] == 0
    
    print("  Compression: PASSED ✓")


def test_nodes():
    """Test node creation and operations."""
    print("Testing nodes...")
    
    # Basic node creation
    node = Node(concept="test", position="n")
    assert node.concept == "test"
    assert node.position == "n"
    assert node.heat == 1.0
    assert node.polarity == 1
    assert node.existence == "actual"
    
    # Self node
    self_node = SelfNode()
    assert self_node.concept == "self"
    assert self_node.position == ""
    assert self_node.heat == float('inf')
    assert self_node.polarity == 0
    assert self_node.righteousness == 0.0
    
    # Self invariants
    assert_self_valid(self_node)
    
    # Axis creation
    axis = Axis(target_id="test123", direction="n", polarity=1)
    assert axis.is_spatial
    assert not axis.is_semantic
    assert axis.capability == "righteous"
    
    # Make axis proper (ordered)
    axis.make_ordered()
    assert axis.capability == "ordered"
    assert axis.is_proper
    
    # Semantic axis
    sem_axis = Axis(target_id="test456", direction="color", polarity=1)
    assert sem_axis.is_semantic
    assert not sem_axis.is_spatial
    
    # Node add_axis
    node.add_axis("test_pred", "some_target_id")
    assert node.has_axis("test_pred")
    assert node.get_axis("test_pred").direction == "test_pred"
    
    # Heat operations
    initial_heat = node.heat
    node.add_heat(1.5)
    assert node.heat == initial_heat + 1.5
    
    print("  Nodes: PASSED ✓")


def test_manifold():
    """Test manifold creation and operations."""
    print("Testing manifold...")
    
    # Create with birth
    m = create_test_manifold()
    
    # Birth creates 8 nodes: Self + 5 spatial + 3 psychology
    assert m.self_node is not None
    assert m.born == True
    assert m.identity_node is not None
    assert m.ego_node is not None
    assert m.conscience_node is not None
    
    # Psychology nodes at correct positions (dual-space architecture)
    # All three are at cubic position "d" but have unique trig_position coordinates
    assert m.identity_node.position == "d"
    assert m.ego_node.position == "d"
    assert m.conscience_node.position == "d"
    
    # Each has unique trig_position (they're separated in trigonometric space)
    assert m.identity_node.trig_position != m.ego_node.trig_position
    assert m.ego_node.trig_position != m.conscience_node.trig_position
    assert m.identity_node.trig_position != m.conscience_node.trig_position
    
    # All trace to Self
    assert m.verify_all_trace_to_self()
    
    # Add a node
    node = Node(concept="test_node", position="nn")
    m.add_node(node)
    assert m.get_node_by_concept("test_node") is not None
    assert m.get_node_by_position("nn") is not None
    
    # Save and load (use temp file to avoid polluting growth/)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        test_path = f.name
    m.save_growth_map(test_path)
    
    m2 = Manifold()
    m2.load_growth_map(test_path)
    assert m2.born == True
    assert m2.identity_node is not None
    assert len(m2.nodes) == len(m.nodes)
    
    # Clean up
    os.remove(test_path)
    
    print("  Manifold: PASSED ✓")


def test_search():
    """Test manifold concept lookup operations."""
    print("Testing concept lookup...")
    
    m = create_test_manifold()
    
    # Create a concept node directly through manifold
    apple_node = Node(concept="apple", position="nn", heat=K)
    m.add_node(apple_node)
    
    # Find by concept
    found = m.get_node_by_concept("apple")
    assert found is not None
    assert found.concept == "apple"
    
    # Find by position
    found_pos = m.get_node_by_position("nn")
    assert found_pos is not None
    assert found_pos.concept == "apple"
    
    # Not found
    not_found = m.get_node_by_concept("banana")
    assert not_found is None
    
    # Update identity awareness
    m.update_identity("apple", heat_delta=0.1, known=True)
    assert m.identity_node.has_axis("apple")
    
    print("  Concept lookup: PASSED ✓")


def test_thermal_dynamics():
    """Test thermal dynamics and psychology."""
    print("Testing thermal dynamics...")
    
    m = create_test_manifold()
    
    # Initial psychology state
    initial_identity_heat = m.identity_node.heat
    initial_ego_heat = m.ego_node.heat
    
    # Psychology methods
    mood = m.get_mood()
    assert mood in ["learning", "curious", "confident", "balanced"]
    
    confidence = m.get_confidence()
    assert 0 <= confidence <= 1
    
    exploration = m.get_exploration_rate()
    assert 0 <= exploration <= 1
    
    curiosity = m.get_curiosity()
    assert curiosity >= 0
    
    # Create a test concept first (so update functions can work)
    test_node = Node(concept="test_concept", position="nn", heat=K)
    m.add_node(test_node)
    
    # Now update psychology (concept exists)
    m.update_identity("test_concept", heat_delta=0.5, known=False)
    # Identity should have gained heat from the search creating the concept
    
    m.update_ego("test_pattern", success=True, heat_delta=0.3)
    # Ego gains heat on success
    
    m.validate_conscience("test_belief", confirmed=True)
    # Conscience tracks validations
    
    # Entropy calculation
    entropy = m.calculate_entropy()
    assert entropy >= 0
    
    print("  Thermal dynamics: PASSED ✓")


def test_invariants():
    """Test system invariants."""
    print("Testing invariants...")
    
    m = create_test_manifold()
    
    # Self is always valid
    assert_self_valid(m.self_node)
    
    # All nodes trace to Self
    assert m.verify_all_trace_to_self()
    
    # Add more nodes and verify
    for i in range(5):
        node = Node(concept=f"concept_{i}", position=f"n{'n' * i}", heat=K)
        m.add_node(node)
    
    # Still valid
    assert m.verify_all_trace_to_self()
    assert_self_valid(m.self_node)
    
    # Self invariants
    assert m.self_node.position == ""
    assert m.self_node.heat == float('inf')
    assert m.self_node.righteousness == 0.0
    assert m.self_node.polarity == 0
    
    print("  Invariants: PASSED ✓")


def test_environment_core():
    """Test environment core functionality."""
    print("Testing environment core...")
    
    m = create_test_manifold()
    
    # Create with manifold connection
    env_core = create_environment_core(manifold=m, use_mock=True)
    assert env_core.manifold is m
    assert env_core.active_driver == "mock"
    
    # Perceive (should auto-integrate)
    initial_id_axes = len(m.identity_node.frame.axes)
    perception = env_core.perceive()
    assert len(perception.entities) > 0
    assert len(perception.locations) > 0
    
    # Identity should have learned from perception
    new_id_axes = len(m.identity_node.frame.axes)
    assert new_id_axes >= initial_id_axes
    
    # Act (should auto-update psychology)
    # Ensure Ego has enough heat for the action (perceptions drain heat)
    m.ego_node.add_heat(1.0)
    initial_ego_heat = m.ego_node.heat
    action = Action(action_type="observe", target="environment")
    result = env_core.act(action)
    assert result.success
    
    # Cleanup
    env_core.deactivate_driver()
    
    print("  Environment Core: PASSED ✓")


def test_inference():
    """Test Ego's inference engine."""
    print("Testing inference...")
    
    m = create_test_manifold()
    
    # Build a knowledge chain: self -> creator -> user -> name -> ian
    self_node = m.self_node
    
    # Create nodes
    user_node = Node(concept="user", position="n")
    m.add_node(user_node)
    self_node.add_axis("creator", user_node.id)
    
    ian_node = Node(concept="ian", position="nn")
    m.add_node(ian_node)
    user_node.add_axis("name", ian_node.id)
    
    # Test single hop
    result1 = m.infer("self", ["creator"])
    assert result1 == "user", f"Expected 'user', got '{result1}'"
    
    # Test two-hop inference
    result2 = m.infer("self", ["creator", "name"])
    assert result2 == "ian", f"Expected 'ian', got '{result2}'"
    
    # Test find_path
    path = m.find_path("self", "ian")
    assert path == ["creator", "name"], f"Expected ['creator', 'name'], got {path}"
    
    print("  Inference: PASSED ✓")


def run_all_tests():
    """Run all tests."""
    print()
    print("=" * 60)
    print("PBAI THERMAL MANIFOLD - TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_constants,
        test_compression,
        test_nodes,
        test_manifold,
        test_search,
        test_thermal_dynamics,
        test_invariants,
        test_environment_core,
        test_inference,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
