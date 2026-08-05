"""
Unit and integration tests for the Batched MCTS implementation.
"""
import sys
import os
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

if __name__ == "__main__":
    print(f"Using pymcts from: {pymcts.__file__}")
    test_batched_mcts_config()
    test_batched_mcts_execution()
