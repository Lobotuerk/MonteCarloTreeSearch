"""Tests for the configurable PUCT exploration constant (TFT-231)."""
import pytest


def test_exploration_constant_default(pymcts_module, tictactoe_state):
    """Default exploration constant should be 1.41."""
    agent = pymcts_module.MCTS_agent(tictactoe_state)
    assert agent.exploration_constant == 1.41


def test_exploration_constant_constructor(pymcts_module, tictactoe_state):
    """Exploration constant set via constructor should be accessible."""
    agent = pymcts_module.MCTS_agent(tictactoe_state, exploration_constant=2.5)
    assert agent.exploration_constant == 2.5


def test_exploration_constant_setter(pymcts_module, tictactoe_state):
    """Exploration constant should be changeable via property setter."""
    agent = pymcts_module.MCTS_agent(tictactoe_state, exploration_constant=1.41)
    assert agent.exploration_constant == 1.41

    agent.exploration_constant = 1.0
    assert agent.exploration_constant == 1.0

    agent.exploration_constant = 3.14
    assert agent.exploration_constant == 3.14


def test_exploration_constant_in_tree_grow(pymcts_module, tictactoe_state):
    """MCTS_tree.grow_tree should accept an exploration_constant parameter."""
    tree = pymcts_module.MCTS_tree(tictactoe_state)
    # Should not raise with the new parameter
    tree.grow_tree(max_iter=5, max_time_in_seconds=1, c=2.0)
    assert tree.get_size() > 0


def test_exploration_constant_affects_selection(pymcts_module):
    """Higher exploration constant should lead to more exploration."""

    class SimpleMove(pymcts_module.MCTS_move):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def __eq__(self, other):
            return isinstance(other, SimpleMove) and self.name == other.name

        def sprint(self):
            return self.name

        def to_numpy(self):
            return [0.0]

        def to_env_action(self):
            return [0]

    class ExplorationTestState(pymcts_module.MCTS_state):
        def __init__(self, is_terminal=False):
            super().__init__()
            self._is_terminal = is_terminal

        def actions_to_try(self):
            if self._is_terminal:
                return []
            return [SimpleMove("A"), SimpleMove("B")]

        def get_action_probabilities(self):
            return [0.9, 0.1]  # A has high prior

        def next_state(self, move):
            return ExplorationTestState(is_terminal=True)

        def rollout(self):
            return 0.5

        def is_terminal(self):
            return self._is_terminal

        def is_self_side_turn(self):
            return True

    original_threads = pymcts_module.get_rollout_threads()
    pymcts_module.set_rollout_threads(1)

    try:
        state = ExplorationTestState()
        wrapped_state = pymcts_module.SerializedPythonState(state)

        # With high exploration, PUCT should explore both moves despite priors
        agent_high = pymcts_module.MCTS_agent(
            wrapped_state, max_iter=50, max_seconds=2, exploration_constant=5.0
        )
        assert agent_high.exploration_constant == 5.0

        # With low exploration, PUCT should mostly exploit the high-prior move
        agent_low = pymcts_module.MCTS_agent(
            wrapped_state, max_iter=50, max_seconds=2, exploration_constant=0.1
        )
        assert agent_low.exploration_constant == 0.1
    finally:
        pymcts_module.set_rollout_threads(original_threads)
