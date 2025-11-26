#!/usr/bin/env python3
"""
Demonstration of heuristic rollout enhancement for MCTS

This demo shows how to implement heuristic-guided rollouts in Python games
to improve MCTS performance compared to pure random rollouts.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('build')
import pymcts
import random
import time

class EnhancedTicTacToeMove(pymcts.MCTS_move):
    """Enhanced TicTacToe move with heuristic evaluation"""
    def __init__(self, x: int, y: int, player: str):
        super().__init__()
        self.x = x
        self.y = y
        self.player = player
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, EnhancedTicTacToeMove):
            return False
        return self.x == other.x and self.y == other.y and self.player == other.player
    
    def sprint(self) -> str:
        return f"({self.x},{self.y},{self.player})"

class EnhancedTicTacToeState(pymcts.MCTS_state):
    """Enhanced TicTacToe state with heuristic rollout capabilities"""
    
    def __init__(self, board=None, current_player='X'):
        super().__init__()
        self.board = board if board is not None else [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = current_player
        self.winner = self._calculate_winner()
        
        # Heuristic configuration
        self.use_heuristic_rollouts = True
        self.heuristic_strength = 0.8  # Probability of using heuristic vs random
    
    def actions_to_try(self):
        """Get all possible moves"""
        if self.is_terminal():
            return []
        
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append(EnhancedTicTacToeMove(i, j, self.current_player))
        return moves
    
    def next_state(self, move):
        """Create new state after applying move"""
        new_board = [row[:] for row in self.board]  # Deep copy
        new_board[move.x][move.y] = move.player
        next_player = 'O' if self.current_player == 'X' else 'X'
        return EnhancedTicTacToeState(new_board, next_player)
    
    def rollout(self) -> float:
        """Standard random rollout"""
        return self._simulate_game(use_heuristics=False)
    
    def heuristic_rollout(self) -> float:
        """Heuristic-guided rollout"""
        return self._simulate_game(use_heuristics=True)
    
    def _simulate_game(self, use_heuristics=False) -> float:
        """Simulate game to completion with optional heuristics"""
        if self.is_terminal():
            if self.winner == 'X':
                return 1.0
            elif self.winner == 'O':
                return 0.0
            else:
                return 0.5  # Draw
        
        # Create copy for simulation
        state = EnhancedTicTacToeState([row[:] for row in self.board], self.current_player)
        
        while not state.is_terminal():
            moves = state.actions_to_try()
            if not moves:
                break
            
            if use_heuristics and random.random() < self.heuristic_strength:
                # Use heuristic move selection
                move = self._select_heuristic_move(state, moves)
            else:
                # Random move selection
                move = random.choice(moves)
            
            state = state.next_state(move)
        
        # Return result from Player X perspective
        if state.winner == 'X':
            return 1.0
        elif state.winner == 'O':
            return 0.0
        else:
            return 0.5
    
    def _select_heuristic_move(self, state, moves):
        """Select move using heuristics"""
        # Priority 1: Win if possible
        for move in moves:
            next_state = state.next_state(move)
            if next_state.winner == state.current_player:
                return move
        
        # Priority 2: Block opponent's win
        opponent = 'O' if state.current_player == 'X' else 'X'
        for move in moves:
            # Check if opponent would win with this move
            opponent_move = EnhancedTicTacToeMove(move.x, move.y, opponent)
            test_state = EnhancedTicTacToeState([row[:] for row in state.board], opponent)
            next_state = test_state.next_state(opponent_move)
            if next_state.winner == opponent:
                return move
        
        # Priority 3: Take center
        for move in moves:
            if move.x == 1 and move.y == 1:
                return move
        
        # Priority 4: Take corners
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for move in moves:
            if (move.x, move.y) in corners:
                return move
        
        # Priority 5: Random choice
        return random.choice(moves)
    
    def evaluate_move(self, move) -> float:
        """Evaluate the quality of a move (0.0 to 1.0)"""
        next_state = self.next_state(move)
        
        # Winning move
        if next_state.winner == self.current_player:
            return 1.0
        
        # Blocking move (check if opponent would win)
        opponent = 'O' if self.current_player == 'X' else 'X'
        opponent_move = EnhancedTicTacToeMove(move.x, move.y, opponent)
        test_state = EnhancedTicTacToeState([row[:] for row in self.board], opponent)
        opponent_next = test_state.next_state(opponent_move)
        if opponent_next.winner == opponent:
            return 0.8
        
        # Positional values
        if move.x == 1 and move.y == 1:  # Center
            return 0.6
        elif (move.x, move.y) in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # Corners
            return 0.4
        else:  # Edges
            return 0.2
    
    def evaluate_position(self) -> float:
        """Evaluate current position (0.0 to 1.0)"""
        if self.is_terminal():
            if self.winner == 'X':
                return 1.0
            elif self.winner == 'O':
                return 0.0
            else:
                return 0.5
        
        # Simple heuristic: count potential winning lines
        x_score = self._count_potential_wins('X')
        o_score = self._count_potential_wins('O')
        total = x_score + o_score
        
        if total == 0:
            return 0.5
        return x_score / total
    
    def _count_potential_wins(self, player) -> int:
        """Count lines where player can still win"""
        lines = [
            # Rows
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            # Columns
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            # Diagonals
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]
        
        count = 0
        opponent = 'O' if player == 'X' else 'X'
        
        for line in lines:
            has_opponent = any(self.board[x][y] == opponent for x, y in line)
            if not has_opponent:  # Line is still winnable
                count += 1
        
        return count
    
    def is_terminal(self) -> bool:
        """Check if game is over"""
        return self.winner is not None
    
    def is_self_side_turn(self) -> bool:
        """Check if it's the self side's turn"""
        return self.current_player == 'X'
    
    

def compare_rollout_strategies():
    """Compare random vs heuristic rollouts"""
    print("🎯 Heuristic Rollout Enhancement Demo")
    print("=" * 50)
    
    iterations = 1000
    
    # Test 1: Random rollouts
    print("\n📊 Test 1: Pure Random Rollouts")
    state = EnhancedTicTacToeState()
    state.use_heuristic_rollouts = False
    
    agent_random = pymcts.MCTS_agent(state, max_iter=iterations, max_seconds=5)
    
    start_time = time.time()
    move_random = agent_random.genmove(None)
    time_random = time.time() - start_time
    
    print(f"⏱️  Time taken: {time_random:.3f}s")
    print(f"🎯 Best move: {move_random}")
    agent_random.feedback()
    
    # Test 2: Heuristic rollouts
    print("\n📊 Test 2: Heuristic-Enhanced Rollouts")
    state2 = EnhancedTicTacToeState()
    state2.use_heuristic_rollouts = True
    
    agent_heuristic = pymcts.MCTS_agent(state2, max_iter=iterations, max_seconds=5)
    
    start_time = time.time()
    move_heuristic = agent_heuristic.genmove(None)
    time_heuristic = time.time() - start_time
    
    print(f"⏱️  Time taken: {time_heuristic:.3f}s")
    print(f"🎯 Best move: {move_heuristic}")
    agent_heuristic.feedback()
    
    # Compare performance
    print("\n📈 Performance Comparison")
    print("-" * 30)
    speedup = time_random / time_heuristic if time_heuristic > 0 else 1.0
    print(f"Random rollouts:    {time_random:.3f}s")
    print(f"Heuristic rollouts: {time_heuristic:.3f}s")
    print(f"Speedup factor:     {speedup:.2f}x")

def play_interactive_game():
    """Play an interactive game against MCTS with heuristic rollouts"""
    print("\n🎮 Interactive Game: Human vs Enhanced MCTS")
    print("=" * 50)
    
    state = EnhancedTicTacToeState()
    agent = pymcts.MCTS_agent(state, max_iter=2000, max_seconds=3)
    
    print("You are 'O', MCTS is 'X'")
    print("Enter moves as 'row,col' (0-2)")
    
    while not state.is_terminal():
        state.print()
        print()
        
        if state.current_player == 'X':
            # MCTS move
            print("🤖 MCTS is thinking...")
            start_time = time.time()
            move = agent.genmove(None)
            think_time = time.time() - start_time
            
            if move:
                print(f"🎯 MCTS chooses: ({move.x},{move.y}) in {think_time:.2f}s")
                state = state.next_state(move)
            else:
                break
        else:
            # Human move
            try:
                user_input = input("Your move (row,col): ").strip()
                if user_input.lower() == 'quit':
                    break
                
                x, y = map(int, user_input.split(','))
                if 0 <= x <= 2 and 0 <= y <= 2 and state.board[x][y] == ' ':
                    move = EnhancedTicTacToeMove(x, y, 'O')
                    state = state.next_state(move)
                    # Update agent with human move
                    agent.genmove(move)
                else:
                    print("❌ Invalid move! Try again.")
                    continue
            except (ValueError, IndexError):
                print("❌ Invalid format! Use 'row,col' (e.g., '1,1')")
                continue
    
    # Game over
    print("\n🏁 Game Over!")
    state.print()
    
    if state.winner == 'X':
        print("🤖 MCTS wins!")
    elif state.winner == 'O':
        print("🎉 You win!")
    else:
        print("🤝 It's a draw!")

def demonstrate_heuristic_methods():
    """Demonstrate the heuristic evaluation methods"""
    print("\n🧠 Heuristic Methods Demonstration")
    print("=" * 50)
    
    # Create a test position
    state = EnhancedTicTacToeState()
    state.board = [
        ['X', ' ', ' '],
        [' ', 'X', ' '],
        [' ', ' ', ' ']
    ]
    state.current_player = 'X'
    
    print("Test position:")
    state.print()
    
    print(f"\n📊 Position evaluation: {state.evaluate_position():.3f}")
    
    print("\n🎯 Move evaluations:")
    moves = state.actions_to_try()
    for move in moves:
        score = state.evaluate_move(move)
        print(f"  Move {move.sprint()}: {score:.3f}")
    
    print("\n🎲 Rollout comparison:")
    random_results = [state.rollout() for _ in range(100)]
    heuristic_results = [state.heuristic_rollout() for _ in range(100)]
    
    avg_random = sum(random_results) / len(random_results)
    avg_heuristic = sum(heuristic_results) / len(heuristic_results)
    
    print(f"  Random rollouts avg:    {avg_random:.3f}")
    print(f"  Heuristic rollouts avg: {avg_heuristic:.3f}")
    print(f"  Improvement:            {((avg_heuristic - avg_random) / avg_random * 100):.1f}%")

def main():
    """Main demo function"""
    print("🚀 MCTS Heuristic Rollout Enhancement Demo")
    print("=" * 60)
    
    try:
        # Test basic functionality
        demonstrate_heuristic_methods()
        
        # Compare strategies
        compare_rollout_strategies()
        
        # Ask if user wants to play
        print("\n" + "=" * 60)
        play_game = input("Would you like to play against enhanced MCTS? (y/n): ").lower().strip()
        if play_game == 'y':
            play_interactive_game()
        
        print("\n✅ Demo completed successfully!")
        print("\n💡 Key takeaways:")
        print("   • Heuristic rollouts can improve MCTS move quality")
        print("   • Smart move evaluation guides simulation toward better outcomes")
        print("   • Position evaluation helps assess game states")
        print("   • Hybrid approaches (random + heuristic) offer flexibility")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()