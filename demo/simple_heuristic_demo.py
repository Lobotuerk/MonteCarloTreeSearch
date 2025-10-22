#!/usr/bin/env python3
"""
Simple demonstration of heuristic rollout enhancement for MCTS
"""
import sys
import os
# Add the build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build'))
import pymcts
import random
import time

class SmartTicTacToeMove(pymcts.MCTS_move):
    def __init__(self, x: int, y: int, player: str):
        super().__init__()
        self.x = x
        self.y = y
        self.player = player
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, SmartTicTacToeMove):
            return False
        return self.x == other.x and self.y == other.y and self.player == other.player
    
    def sprint(self) -> str:
        return f"({self.x},{self.y})"

class SmartTicTacToeState(pymcts.MCTS_state):
    def __init__(self, board=None, current_player='X'):
        super().__init__()
        self.board = board if board is not None else [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = current_player
        self.winner = self._calculate_winner()
    
    def actions_to_try(self):
        if self.is_terminal():
            return []
        
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append(SmartTicTacToeMove(i, j, self.current_player))
        return moves
    
    def next_state(self, move):
        new_board = [row[:] for row in self.board]
        new_board[move.x][move.y] = move.player
        next_player = 'O' if self.current_player == 'X' else 'X'
        return SmartTicTacToeState(new_board, next_player)
    
    def rollout(self) -> float:
        """Enhanced rollout that uses heuristics 70% of the time"""
        if self.is_terminal():
            if self.winner == 'X':
                return 1.0
            elif self.winner == 'O':
                return 0.0
            else:
                return 0.5
        
        state = SmartTicTacToeState([row[:] for row in self.board], self.current_player)
        
        while not state.is_terminal():
            moves = state.actions_to_try()
            if not moves:
                break
            
            # Use heuristics 70% of the time for smarter play
            if random.random() < 0.7:
                move = self._smart_move_selection(state, moves)
            else:
                move = random.choice(moves)
            
            state = state.next_state(move)
        
        if state.winner == 'X':
            return 1.0
        elif state.winner == 'O':
            return 0.0
        else:
            return 0.5
    
    def _smart_move_selection(self, state, moves):
        """Smart move selection using game heuristics"""
        # 1. Win if possible
        for move in moves:
            next_state = state.next_state(move)
            if next_state.winner == state.current_player:
                return move
        
        # 2. Block opponent's win
        opponent = 'O' if state.current_player == 'X' else 'X'
        for move in moves:
            test_move = SmartTicTacToeMove(move.x, move.y, opponent)
            test_state = SmartTicTacToeState([row[:] for row in state.board], opponent)
            next_state = test_state.next_state(test_move)
            if next_state.winner == opponent:
                return move
        
        # 3. Take center
        for move in moves:
            if move.x == 1 and move.y == 1:
                return move
        
        # 4. Take corners
        for move in moves:
            if (move.x, move.y) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                return move
        
        # 5. Random choice
        return random.choice(moves)
    
    def is_terminal(self) -> bool:
        return self.winner is not None
    
    def player1_turn(self) -> bool:
        return self.current_player == 'X'
    
    def _calculate_winner(self):
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        # Check for draw
        if all(self.board[i][j] != ' ' for i in range(3) for j in range(3)):
            return 'D'
        
        return None
    
    def print(self):
        print("   0   1   2")
        for i in range(3):
            print(f"{i}  {self.board[i][0]} | {self.board[i][1]} | {self.board[i][2]} ")
            if i < 2:
                print("  -----------")

class RandomTicTacToeState(SmartTicTacToeState):
    """Pure random rollout version for comparison"""
    def rollout(self) -> float:
        if self.is_terminal():
            if self.winner == 'X':
                return 1.0
            elif self.winner == 'O':
                return 0.0
            else:
                return 0.5
        
        state = RandomTicTacToeState([row[:] for row in self.board], self.current_player)
        
        while not state.is_terminal():
            moves = state.actions_to_try()
            if not moves:
                break
            move = random.choice(moves)  # Pure random
            state = state.next_state(move)
        
        if state.winner == 'X':
            return 1.0
        elif state.winner == 'O':
            return 0.0
        else:
            return 0.5

def test_heuristic_enhancement():
    """Quick test comparing random vs heuristic rollouts"""
    print("🎯 MCTS Heuristic Rollout Enhancement")
    print("=" * 45)
    
    iterations = 500
    time_limit = 2
    
    print("\n🎲 Test 1: Random Rollouts")
    random_state = RandomTicTacToeState()
    random_agent = pymcts.MCTS_agent(random_state, max_iter=iterations, max_seconds=time_limit)
    
    start_time = time.time()
    random_move = random_agent.genmove(None)
    random_time = time.time() - start_time
    
    print(f"Time: {random_time:.2f}s")
    print(f"Move: {random_move.sprint()}")
    
    print("\n🧠 Test 2: Heuristic-Enhanced Rollouts")
    smart_state = SmartTicTacToeState()
    smart_agent = pymcts.MCTS_agent(smart_state, max_iter=iterations, max_seconds=time_limit)
    
    start_time = time.time()
    smart_move = smart_agent.genmove(None)
    smart_time = time.time() - start_time
    
    print(f"Time: {smart_time:.2f}s")
    print(f"Move: {smart_move.sprint()}")
    
    print("\n📈 Results:")
    print(f"Random rollouts:    {random_time:.2f}s -> {random_move.sprint()}")
    print(f"Heuristic rollouts: {smart_time:.2f}s -> {smart_move.sprint()}")
    
    # The heuristic version should often choose center (1,1) as it's strategically better
    if smart_move.x == 1 and smart_move.y == 1:
        print("✅ Heuristic MCTS chose the center - excellent strategy!")
    elif (smart_move.x, smart_move.y) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        print("✅ Heuristic MCTS chose a corner - good strategy!")
    else:
        print("ℹ️  Heuristic MCTS chose an edge - acceptable strategy")
    
    print("\n💡 Key Benefits:")
    print("   • Heuristic rollouts guide simulations toward better outcomes")
    print("   • Smart move prioritization (win > block > center > corner)")
    print("   • Better strategic understanding in game tree exploration")

if __name__ == "__main__":
    test_heuristic_enhancement()