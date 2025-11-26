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
    
    def is_self_side_turn(self) -> bool:
        """Check if it's the self side's turn"""
        return self.current_player == 'X'
    
    def _check_terminal(self) -> bool:
        """Check if the game is over (win or draw)"""
        # Check for winner
        if self._get_winner() is not None:
            return True
        
        # Check for draw (board full)
        for col in range(self.cols):
            if self.board[0][col] == ' ':
                return False  # Still has empty space
        return True  # Board is full
    
    def _get_winner(self) -> Optional[str]:
        """Check for a winner and return 'X', 'O', or None"""
        if self._winner is not None:
            return self._winner
        
        # Check all possible winning conditions
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] != ' ':
                    player = self.board[row][col]
                    # Check horizontal
                    if col + 3 < self.cols:
                        if all(self.board[row][col + i] == player for i in range(4)):
                            self._winner = player
                            return player
                    
                    # Check vertical
                    if row + 3 < self.rows:
                        if all(self.board[row + i][col] == player for i in range(4)):
                            self._winner = player
                            return player
                    
                    # Check diagonal (down-right)
                    if row + 3 < self.rows and col + 3 < self.cols:
                        if all(self.board[row + i][col + i] == player for i in range(4)):
                            self._winner = player
                            return player
                    
                    # Check diagonal (down-left)
                    if row + 3 < self.rows and col - 3 >= 0:
                        if all(self.board[row + i][col - i] == player for i in range(4)):
                            self._winner = player
                            return player
        
        return None
    
    def get_winner(self) -> Optional[str]:
        """Public method to get the winner"""
        return self._get_winner()
    
    def print(self) -> None:
        """Print the current board state"""
        print("\n  " + " ".join(str(i) for i in range(self.cols)))
        for row in self.board:
            print("| " + " ".join(row) + " |")
        print("-" * (2 * self.cols + 3))
        print(f"Current player: {self.current_player}")


def demo_connect_four():
    """Demonstrate Connect Four implemented in pure Python"""
    print("=== Connect Four - Pure Python Implementation ===")
    
    # Create initial state
    state = ConnectFourState()
    print("Initial board:")
    state.print()
    
    print("\nStarting Connect Four game with MCTS vs Random...")
    moves_played = 0
    max_moves = 20
    
    while not state.is_terminal() and moves_played < max_moves:
        print(f"\n--- Move {moves_played + 1} ---")
        state.print()
        
        current_player = state.current_player
        print(f"Current player: {current_player}")
        
        if current_player == 'X':
            # MCTS agent's turn
            print("MCTS agent thinking...")
            agent = pymcts.MCTS_agent(pymcts.SerializedPythonState(state), max_iter=500, max_seconds=5)
            
            move = agent.genmove()
            
            if move:
                # Convert the MCTS_move back to a ConnectFourMove
                move_str = str(move)  # Should be "DropX@column"
                print(f"MCTS chose: {move_str}")
                
                # Parse the move string to recreate the ConnectFourMove
                # Format is "DropPLAYER@COLUMN"
                if move_str.startswith('Drop') and '@' in move_str:
                    parts = move_str.split('@')
                    if len(parts) == 2:
                        player_part = parts[0][4:]  # Remove "Drop" prefix
                        column = int(parts[1])
                        python_move = ConnectFourMove(column, player_part)
                        state = state.next_state(python_move)
                    else:
                        print("Error: Could not parse move string")
                        break
                else:
                    print("Error: Unexpected move format")
                    break
            else:
                print("No move returned from MCTS")
                break
        else:
            # Random opponent's turn (O)
            moves = state.actions_to_try()
            if moves:
                random_move = random.choice(moves)
                print(f"Random opponent chose: {random_move}")
                state = state.next_state(random_move)
            else:
                print("No moves available!")
                break
        
        moves_played += 1
    
    print("\n=== Final State ===")
    state.print()
    
    if state.is_terminal():
        winner = state.get_winner()
        if winner:
            print(f"\n🎉 Winner: {winner}!")
        else:
            print("\n🤝 It's a draw!")
    else:
        print(f"\n⏰ Demo stopped after {max_moves} moves")
    
    print("✅ Connect Four demo completed!")
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
    print(f"Self side turn: {state.is_self_side_turn()}")
    
    # Test actions
    actions = state.actions_to_try()
    print(f"Available actions: {len(actions)}")
    print(f"First few actions: {[str(a) for a in actions[:3]]}")
    
    # Test making a move
    if actions:
        new_state = state.next_state(actions[0])
        print(f"After move, self side turn: {new_state.is_self_side_turn()}")
    
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