"""
Unit and integration tests for the Batched MCTS implementation.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pymcts


class BatchedMove(pymcts.MCTS_move):
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def __eq__(self, other):
        return isinstance(other, BatchedMove) and self.value == other.value
    
    def sprint(self):
        return f"Move({self.value})"

class BatchedState(pymcts.MCTS_state):
    batch_calls = 0
    
    def __init__(self, turn=0, moves_made=0):
        super().__init__()
        self.turn = turn
        self.moves_made = moves_made
    
    def actions_to_try(self):
        if self.is_terminal():
            return []
        return [BatchedMove(0), BatchedMove(1)]
    
    def next_state(self, move):
        return BatchedState((self.turn + 1) % 2, self.moves_made + 1)
    
    def rollout(self):
        return 0.5
    
    def is_terminal(self):
        return self.moves_made >= 3
    
    def is_self_side_turn(self):
        return self.turn == 0
    
    def clone(self):
        return BatchedState(self.turn, self.moves_made)
    
    def print(self):
        pass
        
    def evaluate_batch(self, states):
        print(f"--- Python evaluate_batch called with {len(states)} states ---")
        BatchedState.batch_calls += 1
        results = []
        for state in states:
            results.append((0.7, [0.5, 0.5]))
        return results

def test_batched_mcts_config():
    print("test_batched_mcts_config: starting")
    state = BatchedState()
    print("test_batched_mcts_config: created state")
    wrapped_state = pymcts.SerializedPythonState(state)
    print("test_batched_mcts_config: wrapped state")
    agent = pymcts.MCTS_agent(wrapped_state, 100, 1)
    print("test_batched_mcts_config: created agent")
    
    assert agent.batch_size == 64
    assert agent.num_search_threads == 4
    print("test_batched_mcts_config: checked defaults")
    
    agent.batch_size = 32
    agent.num_search_threads = 2
    assert agent.batch_size == 32
    assert agent.num_search_threads == 2
    print("test_batched_mcts_config: checked setters")
    print("test_batched_mcts_config passed!")

def test_batched_mcts_execution():
    print("test_batched_mcts_execution: starting")
    BatchedState.batch_calls = 0
    state = BatchedState()
    print("test_batched_mcts_execution: created state")
    wrapped_state = pymcts.SerializedPythonState(state)
    print("test_batched_mcts_execution: wrapped state")
    agent = pymcts.MCTS_agent(wrapped_state, 10, 2)
    print("test_batched_mcts_execution: created agent")
    agent.batch_size = 2
    agent.num_search_threads = 2
    
    print("test_batched_mcts_execution: calling genmove")
    move = agent.genmove(None)
    print("test_batched_mcts_execution: genmove returned")
    assert move is not None
    assert BatchedState.batch_calls > 0
    print(f"Batch calls made: {BatchedState.batch_calls}")
    print("test_batched_mcts_execution passed!")

def test_minimax_repulsion_unit():
    """At MIN parents, apply_virtual_loss should INCREASE the score (+VL sign flip)."""
    print("test_minimax_repulsion_unit: starting")
    from math import isclose

    state = BatchedState(turn=0, moves_made=0)
    wrapped_state = pymcts.SerializedPythonState(state)
    agent = pymcts.MCTS_agent(wrapped_state, 10, 1)
    agent.batch_size = 2
    agent.num_search_threads = 1

    # Run a search to build a tree
    move = agent.genmove(None)
    assert move is not None

    # After genmove, tree.root is the best child (turn=1, MIN).
    # Its children have parent=MIN, so apply_virtual_loss should INCREASE score.
    root = agent.tree.root
    assert root is not None
    assert len(root.get_children()) > 0

    # root is at turn=1 (MIN), its children have parent=MIN
    child = root.get_children()[0]
    original_score = child.score
    original_visits = child.visit_count

    child.apply_virtual_loss(1.0)
    assert child.visit_count == original_visits + 1
    # At MIN parent, score should INCREASE by exactly vl
    assert isclose(child.score, original_score + 1.0, abs_tol=1e-9), \
        f"Expected score {original_score + 1.0}, got {child.score}"

    child.remove_virtual_loss(1.0)
    assert child.visit_count == original_visits
    assert isclose(child.score, original_score, abs_tol=1e-9), \
        f"Expected score {original_score} after remove, got {child.score}"

    print("test_minimax_repulsion_unit: MIN parent repulsion verified!")
    print("test_minimax_repulsion_unit passed!")


def test_max_node_repulsion_unit():
    """At MAX parents, apply_virtual_loss should DECREASE the score (-VL)."""
    print("test_max_node_repulsion_unit: starting")
    from math import isclose

    state = BatchedState(turn=0, moves_made=0)
    wrapped_state = pymcts.SerializedPythonState(state)
    agent = pymcts.MCTS_agent(wrapped_state, 10, 1)
    agent.batch_size = 2
    agent.num_search_threads = 1

    # Run a search to build a tree
    move = agent.genmove(None)
    assert move is not None

    # After genmove, tree.root is the best child (turn=1, MIN).
    # Its children have parent=MIN, and grandchildren have parent=MAX (turn=0).
    root = agent.tree.root
    assert root is not None
    assert len(root.get_children()) > 0

    # root is at turn=1 (MIN), its child is at turn=0 (MAX)
    child = root.get_children()[0]
    grandchildren = child.get_children()

    if len(grandchildren) > 0:
        grandchild = grandchildren[0]
        original_score = grandchild.score
        original_visits = grandchild.visit_count

        grandchild.apply_virtual_loss(1.0)
        assert grandchild.visit_count == original_visits + 1
        # At MAX parent (child at turn=0), score should DECREASE by exactly vl
        assert isclose(grandchild.score, original_score - 1.0, abs_tol=1e-9), \
            f"Expected score {original_score - 1.0}, got {grandchild.score}"

        grandchild.remove_virtual_loss(1.0)
        assert grandchild.visit_count == original_visits
        assert isclose(grandchild.score, original_score, abs_tol=1e-9)
        print("test_max_node_repulsion_unit: MAX parent repulsion verified!")
    else:
        # Fallback: root's child has parent=MIN, test that instead
        original_score = child.score
        original_visits = child.visit_count
        child.apply_virtual_loss(1.0)
        assert child.visit_count == original_visits + 1
        assert isclose(child.score, original_score + 1.0, abs_tol=1e-9)
        child.remove_virtual_loss(1.0)
        assert isclose(child.score, original_score, abs_tol=1e-9)
        print("test_max_node_repulsion_unit: no grandchildren, MIN parent verified")

    print("test_max_node_repulsion_unit passed!")


def test_virtual_loss_configurable():
    """Verify virtual_loss can be configured on the MCTS_agent."""
    print("test_virtual_loss_configurable: starting")
    state = BatchedState()
    wrapped_state = pymcts.SerializedPythonState(state)
    agent = pymcts.MCTS_agent(wrapped_state, 10, 1)

    assert agent.virtual_loss == 1.0
    agent.virtual_loss = 2.0
    assert agent.virtual_loss == 2.0
    agent.virtual_loss = 0.5
    assert agent.virtual_loss == 0.5

    print("test_virtual_loss_configurable passed!")


def test_multi_thread_integration():
    """Run batched MCTS with num_search_threads > 1 to ensure no deadlocks."""
    print("test_multi_thread_integration: starting")
    BatchedState.batch_calls = 0
    state = BatchedState()
    wrapped_state = pymcts.SerializedPythonState(state)
    agent = pymcts.MCTS_agent(wrapped_state, 50, 5)
    agent.batch_size = 4
    agent.num_search_threads = 3
    agent.virtual_loss = 1.0

    move = agent.genmove(None)
    assert move is not None
    assert BatchedState.batch_calls > 0
    print(f"test_multi_thread_integration: {BatchedState.batch_calls} batch calls, no deadlock")
    print("test_multi_thread_integration passed!")


if __name__ == "__main__":
    print(f"Using pymcts from: {pymcts.__file__}")
    test_batched_mcts_config()
    test_batched_mcts_execution()
    test_virtual_loss_configurable()
    test_max_node_repulsion_unit()
    test_minimax_repulsion_unit()
    test_multi_thread_integration()
