#!/usr/bin/env python3
"""
Example: Connect Four game implemented entirely in Python using PyMCTS

This demonstrates how to create new games without touching C++ code.
"""
import sys
sys.path.append('../build')
import pymcts
import random
from typing import List, Optional

class ConnectFourMove(pymcts.MCTS_move):
    """A move in Connect Four - just drop a piece in a column"""
    
    def __init__(self, column: int, player: str):
        super().__init__()
        self.column = column
        self.player = player
    
    def __eq__(self, other) -> bool:
        """Required: equality comparison for moves"""
        if isinstance(other, ConnectFourMove):
            return self.column == other.column and self.player == other.player
        return False
    
    def sprint(self) -> str:
        """Optional: string representation for debugging"""
        return f"Drop{self.player}@{self.column}"
    
    def __str__(self) -> str:
        return self.sprint()

class ConnectFourState(pymcts.MCTS_state):
    """Connect Four game state implemented in pure Python"""
    
    def __init__(self, rows: int = 6, cols: int = 7, board: Optional[List[List[str]]] = None):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.board = board if board else [[' ' for _ in range(cols)] for _ in range(rows)]
        self.current_player = 'X'  # X goes first
        self._terminal = None
        self._winner = None
    
    def actions_to_try(self) -> List[ConnectFourMove]:
        """Required: return list of valid moves"""
        if self.is_terminal():
            return []
        
        moves = []
        for col in range(self.cols):
            if self.board[0][col] == ' ':  # Column not full
                moves.append(ConnectFourMove(col, self.current_player))
        return moves
    
    def next_state(self, move: ConnectFourMove):
        """Required: return new state after applying move"""
        if not isinstance(move, ConnectFourMove):
            raise ValueError("Invalid move type")
        
        # Find the lowest empty row in the column
        row = -1
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][move.column] == ' ':
                row = r
                break
        
        if row == -1:
            raise ValueError(f"Column {move.column} is full")
        
        # Create new board with the move applied
        new_board = [row[:] for row in self.board]
        new_board[row][move.column] = move.player
        
        # Create new state
        new_state = ConnectFourState(self.rows, self.cols, new_board)
        new_state.current_player = 'O' if self.current_player == 'X' else 'X'
        return new_state
    
    def rollout(self) -> float:
        """Required: simulate random game and return win probability for player 1 (X)"""
        if self.is_terminal():
            winner = self.get_winner()
            if winner == 'X':
                return 1.0
            elif winner == 'O':
                return 0.0
            else:
                return 0.5  # Draw
        
        # Simulate random game
        current_state = ConnectFourState(self.rows, self.cols, 
                                       [row[:] for row in self.board])
        current_state.current_player = self.current_player
        
        while not current_state.is_terminal():
            moves = current_state.actions_to_try()
            if not moves:
                break
            
            # Random move
            move = random.choice(moves)
            current_state = current_state.next_state(move)
        
        winner = current_state.get_winner()
        if winner == 'X':
            return 1.0
        elif winner == 'O':
            return 0.0
        else:
            return 0.5  # Draw
    
    def is_terminal(self) -> bool:
        """Required: check if game is over"""
        if self._terminal is None:
            self._terminal = self._check_terminal()
        return self._terminal
    
    def player1_turn(self) -> bool:
        """Required: return True if it's player 1's (X) turn"""
        return self.current_player == 'X'
    
    def print(self) -> None:
        """Optional: print the game state"""
        print("  " + " ".join(str(i) for i in range(self.cols)))
        print("  " + "-" * (self.cols * 2 - 1))
        for row in self.board:
            print("| " + " ".join(row) + " |")
        print("  " + "-" * (self.cols * 2 - 1))
        print(f"Current player: {self.current_player}")
    
    def get_winner(self) -> Optional[str]:
        """Get the winner (X, O, or None for draw/ongoing)"""
        if self._winner is None:
            self._winner = self._check_winner()
        return self._winner
    
    def _check_terminal(self) -> bool:
        """Check if the game is over"""
        # Check for winner
        if self._check_winner() is not None:
            return True
        
        # Check if board is full
        for col in range(self.cols):
            if self.board[0][col] == ' ':
                return False
        return True  # Board is full
    
    def _check_winner(self) -> Optional[str]:
        """Check for a winner"""
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if (self.board[row][col] != ' ' and 
                    self.board[row][col] == self.board[row][col+1] == 
                    self.board[row][col+2] == self.board[row][col+3]):
                    return self.board[row][col]
        
        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if (self.board[row][col] != ' ' and 
                    self.board[row][col] == self.board[row+1][col] == 
                    self.board[row+2][col] == self.board[row+3][col]):
                    return self.board[row][col]
        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if (self.board[row][col] != ' ' and 
                    self.board[row][col] == self.board[row+1][col+1] == 
                    self.board[row+2][col+2] == self.board[row+3][col+3]):
                    return self.board[row][col]
        
        # Check diagonal (top-right to bottom-left)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if (self.board[row][col] != ' ' and 
                    self.board[row][col] == self.board[row+1][col-1] == 
                    self.board[row+2][col-2] == self.board[row+3][col-3]):
                    return self.board[row][col]
        
        return None

def demo_connect_four():
    """Demonstrate Connect Four implemented in pure Python"""
    print("=== Connect Four - Pure Python Implementation ===")
    
    # Create initial state
    state = ConnectFourState()
    print("Initial board:")
    state.print()
    
    # Create MCTS agent
    print("\nCreating MCTS agent...")
    agent = pymcts.MCTS_agent(state, max_iter=1000, max_seconds=3)
    
    # Configure parallel rollouts
    pymcts.set_rollout_threads(4)
    print(f"Using {pymcts.get_rollout_threads()} parallel rollout threads")
    
    # Play a few moves
    print("\n=== Game Play ===")
    move_count = 0
    
    while not agent.get_current_state().is_terminal() and move_count < 10:
        current_state = agent.get_current_state()
        
        print(f"\n--- Move {move_count + 1} ---")
        current_state.print()
        
        if current_state.player1_turn():
            # MCTS agent's turn (X)
            print("MCTS agent thinking...")
            move = agent.genmove(None)
            if move:
                print(f"MCTS chose: {move}")
            else:
                print("No moves available!")
                break
        else:
            # Random opponent (O)
            possible_moves = current_state.actions_to_try()
            if possible_moves:
                opponent_move = random.choice(possible_moves)
                print(f"Random opponent chose: {opponent_move}")
                agent.genmove(opponent_move)
            else:
                print("No moves available!")
                break
        
        move_count += 1
    
    # Final state
    final_state = agent.get_current_state()
    print("\n=== Final State ===")
    final_state.print()
    
    if final_state.is_terminal():
        winner = final_state.get_winner()
        if winner:
            print(f"\n🎉 Winner: {winner}!")
        else:
            print("\n🤝 It's a draw!")
    
    print(f"\nGame completed in {move_count} moves")
    return True

def test_connect_four_basics():
    """Test basic Connect Four functionality"""
    print("\n=== Testing Connect Four Basics ===")
    
    # Test move creation
    move = ConnectFourMove(3, 'X')
    print(f"Created move: {move}")
    
    # Test state creation
    state = ConnectFourState()
    print(f"Board size: {state.rows}x{state.cols}")
    print(f"Is terminal: {state.is_terminal()}")
    print(f"Player 1 turn: {state.player1_turn()}")
    
    # Test actions
    actions = state.actions_to_try()
    print(f"Available actions: {len(actions)}")
    print(f"First few actions: {[str(a) for a in actions[:3]]}")
    
    # Test making a move
    if actions:
        new_state = state.next_state(actions[0])
        print(f"After move, player 1 turn: {new_state.player1_turn()}")
    
    print("✅ Basic tests passed!")

if __name__ == "__main__":
    # Run tests
    test_connect_four_basics()
    
    # Run demo
    try:
        demo_connect_four()
        print("\n🎉 Connect Four demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()