"""
Test that learning nodes persist correctly.

Verifies:
- SensorReport/MotorAction/ActionPlan (driver components)
- DecisionNode → decisions/
"""

import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Manifold
from core.driver_node import SensorReport, MotorAction, ActionPlan, MotorType, press, create_plan
from core.decision_node import DecisionNode, Choice
from core.nodes import reset_birth_for_testing


def test_driver_components():
    """Test SensorReport, MotorAction, ActionPlan serialize correctly"""
    print("\n=== Testing Driver Components ===")
    
    # Test SensorReport
    sensor = SensorReport.vision(
        description="tree 10 blocks north",
        objects=[{"type": "tree", "direction": "north"}],
        threats=[]
    )
    sensor_dict = sensor.to_dict()
    sensor_loaded = SensorReport.from_dict(sensor_dict)
    assert sensor_loaded.sensor_type == "vision"
    assert sensor_loaded.description == "tree 10 blocks north"
    print("  ✓ SensorReport serializes correctly")
    
    # Test MotorAction
    motor = press("w")
    motor_dict = motor.to_dict()
    motor_loaded = MotorAction.from_dict(motor_dict)
    assert motor_loaded.key == "w"
    assert motor_loaded.motor_type == MotorType.KEY_PRESS
    print("  ✓ MotorAction serializes correctly")
    
    # Test ActionPlan
    plan = create_plan(
        name="move_forward",
        goal="move character forward",
        steps=[press("w"), press("w")],
        requires=[],
        provides=["moved"]
    )
    plan_dict = plan.to_dict()
    plan_loaded = ActionPlan.from_dict(plan_dict)
    assert plan_loaded.name == "move_forward"
    assert len(plan_loaded.steps) == 2
    print("  ✓ ActionPlan serializes correctly")
    
    return True


def test_decision_node_persistence():
    """Test DecisionNode saves to decisions/"""
    print("\n=== Testing DecisionNode Persistence ===")
    
    reset_birth_for_testing()
    manifold = Manifold()
    manifold.birth()
    
    # Create decision node
    decision = DecisionNode(manifold)
    
    # Make a decision with context
    decision.begin_decision(
        state_key="test_state",
        options=["up", "down", "left", "right"],
        confidence=1.0,
        context={"near_cliff": True}
    )
    decision.commit_decision("up")
    decision.complete_decision("up_success", success=True, heat_delta=0.5)
    
    # Verify file exists
    decisions_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "decisions")
    filepath = os.path.join(decisions_path, "pbai_decisions.json")
    
    assert os.path.exists(filepath), f"Decisions file not created: {filepath}"
    
    # Verify content
    with open(filepath, "r") as f:
        data = json.load(f)
    
    assert "choices" in data, "No choices in decisions file"
    assert len(data["choices"]) > 0, "No choices recorded"
    
    # Verify context was saved
    last_choice = data["choices"][-1]
    assert "context" in last_choice, "Context not saved"
    assert last_choice["context"].get("near_cliff") == True, "Context value wrong"
    
    print(f"  ✓ DecisionNode persists to {filepath}")
    print(f"    - {len(data['choices'])} choice(s) recorded")
    print(f"    - Context preserved: {last_choice['context']}")
    
    return True


def main():
    """Run all persistence tests."""
    print("=" * 60)
    print("PBAI Learning Node Persistence Tests")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("DriverComponents", test_driver_components()))
    except Exception as e:
        print(f"  ✗ DriverComponents FAILED: {e}")
        results.append(("DriverComponents", False))
    
    try:
        results.append(("DecisionNode", test_decision_node_persistence()))
    except Exception as e:
        print(f"  ✗ DecisionNode FAILED: {e}")
        results.append(("DecisionNode", False))
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n{passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
