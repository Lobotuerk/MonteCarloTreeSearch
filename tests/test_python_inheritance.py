"""
Tests for Python inheritance from C++ base classes.
Tests the trampoline functionality that allows Python games to inherit from MCTS_move and MCTS_state.
NOTE: MCTS agent tests disabled due to C++ memory corruption issue.
"""
import pytest
import sys
import os

# Import demo games for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'demo'))

try:
    from connect_four_python import ConnectFourMove, ConnectFourState
    CONNECT_FOUR_AVAILABLE = True
except ImportError:
    CONNECT_FOUR_AVAILABLE = False


class TestPythonInheritance:
    """Test that Python classes can inherit from C++ base classes."""
    
    def test_simple_move_inheritance(self, pymcts_module, simple_python_move):
        """Test that Python moves inherit correctly from C++ base."""
        assert isinstance(simple_python_move, pymcts_module.MCTS_move)
        
        # Test required methods work
        assert simple_python_move.sprint() == "Move(0)"
        
        # Test equality
        other_move = simple_python_move.__class__(0)
        assert simple_python_move == other_move
        
        different_move = simple_python_move.__class__(1)
        assert simple_python_move != different_move
        
    def test_simple_state_inheritance(self, pymcts_module, simple_python_state):
        """Test that Python states inherit correctly from C++ base."""
        assert isinstance(simple_python_state, pymcts_module.MCTS_state)
        
        # Test all required methods exist and work
        assert hasattr(simple_python_state, 'actions_to_try')
        assert hasattr(simple_python_state, 'next_state')
        assert hasattr(simple_python_state, 'rollout')
        assert hasattr(simple_python_state, 'is_terminal')
        assert hasattr(simple_python_state, 'player1_turn')
        
        # Test methods return expected types
        assert isinstance(simple_python_state.is_terminal(), bool)
        assert isinstance(simple_python_state.player1_turn(), bool)
        assert isinstance(simple_python_state.rollout(), (int, float))
        
        moves = simple_python_state.actions_to_try()
        assert isinstance(moves, list)

    def test_state_transitions(self, simple_python_state):
        """Test that state transitions work correctly."""
        # Get initial moves
        moves = simple_python_state.actions_to_try()
        assert len(moves) > 0
        
        # Make a transition
        new_state = simple_python_state.next_state(moves[0])
        assert new_state is not None
        assert new_state != simple_python_state
        
        # Player should have switched
        if simple_python_state.player1_turn():
            assert not new_state.player1_turn()
        else:
            assert new_state.player1_turn()

    def test_rollout_values(self, simple_python_state):
        """Test that rollout returns valid values."""
        for _ in range(5):  # Test multiple times
            rollout_result = simple_python_state.rollout()
            assert isinstance(rollout_result, (int, float))
            assert 0.0 <= rollout_result <= 1.0

    def test_terminal_detection(self, simple_python_state):
        """Test terminal state detection works."""
        # Initial state should not be terminal
        assert not simple_python_state.is_terminal()
        
        # Advance the game to terminal state
        current_state = simple_python_state
        moves_made = 0
        
        while not current_state.is_terminal() and moves_made < 10:  # Safety limit
            moves = current_state.actions_to_try()
            if not moves:
                break
            current_state = current_state.next_state(moves[0])
            moves_made += 1
        
        # Should eventually reach terminal state
        assert current_state.is_terminal()
        
        # Terminal state should have no moves
        terminal_moves = current_state.actions_to_try()
        assert len(terminal_moves) == 0


@pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four demo not available")
class TestConnectFourInheritance:
    """Test Connect Four as a more complex example of Python inheritance."""
    
    def test_connect_four_move_inheritance(self, pymcts_module):
        """Test Connect Four move inheritance."""
        move = ConnectFourMove(3, 'X')
        assert isinstance(move, pymcts_module.MCTS_move)
        
        # Test string representation
        move_str = move.sprint()
        assert "3" in move_str
        assert "X" in move_str
        
    def test_connect_four_state_inheritance(self, pymcts_module):
        """Test Connect Four state inheritance."""
        state = ConnectFourState()
        assert isinstance(state, pymcts_module.MCTS_state)
        
        # Test initial state properties
        assert not state.is_terminal()
        assert state.player1_turn()
        
        # Should have moves available
        moves = state.actions_to_try()
        assert len(moves) == 7  # 7 columns in Connect Four

    def test_connect_four_basic_functionality(self):
        """Test Connect Four basic functionality without MCTS."""
        state = ConnectFourState()
        
        # Test move generation
        moves = state.actions_to_try()
        assert len(moves) == 7
        
        # Test state transition
        if moves:
            new_state = state.next_state(moves[0])
            assert new_state is not None
            assert not new_state.player1_turn()  # Should switch players
            
        # Test rollout
        rollout_result = state.rollout()
        assert 0.0 <= rollout_result <= 1.0


class TestInheritanceBasics:
    """Test basic inheritance functionality without MCTS agents."""
    
    def test_method_overrides_work(self, pymcts_module):
        """Test that all virtual methods are properly overridden."""
        
        class TestMove(pymcts_module.MCTS_move):
            def __init__(self):
                super().__init__()
                self.eq_called = False
                self.sprint_called = False
            
            def __eq__(self, other):
                self.eq_called = True
                return True
            
            def sprint(self):
                self.sprint_called = True
                return "test"
        
        class TestState(pymcts_module.MCTS_state):
            def __init__(self):
                super().__init__()
                self.method_calls = []
            
            def actions_to_try(self):
                self.method_calls.append('actions_to_try')
                return [TestMove()]
            
            def next_state(self, move):
                self.method_calls.append('next_state')
                return TestState()
            
            def rollout(self):
                self.method_calls.append('rollout')
                return 0.5
            
            def is_terminal(self):
                self.method_calls.append('is_terminal')
                return len(self.method_calls) > 5  # Terminal after a few calls
            
            def player1_turn(self):
                self.method_calls.append('player1_turn')
                return True
        
        # Test that methods work
        state = TestState()
        move = TestMove()
        
        # Test move methods
        assert move.sprint() == "test"
        assert move.sprint_called
        
        assert move == move
        assert move.eq_called
        
        # Test state methods
        assert not state.is_terminal()
        assert 'is_terminal' in state.method_calls
        
        assert state.player1_turn()
        assert 'player1_turn' in state.method_calls
        
        moves = state.actions_to_try()
        assert len(moves) > 0
        assert 'actions_to_try' in state.method_calls
        
        rollout_result = state.rollout()
        assert rollout_result == 0.5
        assert 'rollout' in state.method_calls


# Working MCTS agent tests - issue was incorrect constructor syntax

class TestPythonStateWithMCTS:
    """Test that Python states work with MCTS agent."""
    
    def test_python_state_with_mcts(self, pymcts_module, simple_python_state):
        """Test that Python states work with MCTS agent."""
        # This is the critical test - can MCTS use our Python implementation?
        agent = pymcts_module.MCTS_agent(simple_python_state, 10, 1)  # 10 iterations, 1 second max
        
        # Should be able to generate a move
        move = agent.genmove(None)
        assert move is not None
        assert hasattr(move, 'sprint')
        
        move_str = move.sprint()
        assert isinstance(move_str, str)
        assert len(move_str) > 0
        
        print(f"MCTS with Python state generated: {move_str}")


class TestConnectFourWithMCTS:
    """Test Connect Four with MCTS agent."""
    
    @pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four not available")
    def test_connect_four_with_mcts(self, pymcts_module):
        """Test Connect Four with MCTS agent."""
        state = ConnectFourState()
        
        # Use small parameters for quick test
        agent = pymcts_module.MCTS_agent(state, 20, 1)  # 20 iterations, 1 second max
        
        move = agent.genmove(None)
        assert move is not None
        
        # Should be a valid Connect Four move
        assert hasattr(move, 'column')
        assert 0 <= move.column <= 6
        assert hasattr(move, 'player')
        assert move.player in ['X', 'O']
        
        print(f"Connect Four MCTS move: {move.sprint()}")
        
    @pytest.mark.skipif(not CONNECT_FOUR_AVAILABLE, reason="Connect Four not available")
    def test_connect_four_game_progression(self, pymcts_module):
        """Test a short Connect Four game."""
        state = ConnectFourState()
        agent = pymcts_module.MCTS_agent(state, 10, 1)  # Quick test
        
        moves_played = 0
        while not agent.get_current_state().is_terminal() and moves_played < 6:
            # MCTS move
            move = agent.genmove(None)
            assert move is not None
            moves_played += 1
            
            current_state = agent.get_current_state()
            if current_state.is_terminal():
                break
                
            # Simulate opponent move (just take first available)
            opponent_moves = current_state.actions_to_try()
            if opponent_moves:
                agent.genmove(opponent_moves[0])
                moves_played += 1
        
        assert moves_played > 0
        print(f"Connect Four game played {moves_played} moves")


class TestTrampolineRobustness:
    """Test edge cases and robustness of the trampoline implementation."""
    
    def test_multiple_agents_same_state_type(self, pymcts_module, simple_python_state):
        """Test multiple agents using the same Python state type."""
        # Create multiple agents with the same state type
        agent1 = pymcts_module.MCTS_agent(simple_python_state, 5, 1)  # 5 iterations, 1 second max
        
        # Create another state for the second agent
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        # Create a fresh simple state using the fixture class type
        SimpleStateClass = type(simple_python_state)
        simple_python_state2 = SimpleStateClass()
        agent2 = pymcts_module.MCTS_agent(simple_python_state2, 5, 1)
        
        # Both should work
        move1 = agent1.genmove(None)
        move2 = agent2.genmove(None)
        
        assert move1 is not None
        assert move2 is not None
        
        print(f"Agent 1 move: {move1.sprint()}, Agent 2 move: {move2.sprint()}")
        
    def test_inheritance_method_overrides_with_mcts(self, pymcts_module):
        """Test that all virtual methods are properly overridden and work with MCTS."""
        
        class TestMove(pymcts_module.MCTS_move):
            def __init__(self):
                super().__init__()
                self.eq_called = False
                self.sprint_called = False
            
            def __eq__(self, other):
                self.eq_called = True
                return True
            
            def sprint(self):
                self.sprint_called = True
                return "test"
        
        class TestState(pymcts_module.MCTS_state):
            def __init__(self):
                super().__init__()
                self.method_calls = []
            
            def actions_to_try(self):
                self.method_calls.append('actions_to_try')
                return [TestMove()]
            
            def next_state(self, move):
                self.method_calls.append('next_state')
                return TestState()
            
            def rollout(self):
                self.method_calls.append('rollout')
                return 0.5
            
            def is_terminal(self):
                self.method_calls.append('is_terminal')
                return len(self.method_calls) > 10  # Terminal after several calls
            
            def player1_turn(self):
                self.method_calls.append('player1_turn')
                return True
        
        # Test that methods are called by MCTS
        state = TestState()
        agent = pymcts_module.MCTS_agent(state, 5, 1)  # 5 iterations, 1 second max
        move = agent.genmove(None)
        
        assert move is not None
        assert len(state.method_calls) > 0
        assert 'actions_to_try' in state.method_calls
        
        print(f"Methods called by MCTS: {state.method_calls}")