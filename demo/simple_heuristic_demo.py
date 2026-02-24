#!/usr/bin/env python3
"""
Simple demonstration of different MCTS configurations
"""
import sys
import os
# Add the project directory to path to find pymcts module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pymcts
import time

def test_heuristic_enhancement():
    """Quick test comparing C++ TicTacToe with different MCTS parameters"""
    print("🎯 MCTS TicTacToe Performance Demo")
    print("=" * 45)
    
    iterations_low = 100
    iterations_high = 500
    time_limit = 2
    
    print("\n🎲 Test 1: Low Iterations")
    state1 = pymcts.TicTacToe_state()
    agent1 = pymcts.MCTS_agent(state1, iterations_low, time_limit)
    
    start_time = time.time()
    move1 = agent1.genmove(None)
    time1 = time.time() - start_time
    
    print(f"Time: {time1:.2f}s")
    print(f"Move: {move1.sprint()}")
    
    print("\n🧠 Test 2: High Iterations")
    state2 = pymcts.TicTacToe_state()
    agent2 = pymcts.MCTS_agent(state2, iterations_high, time_limit)
    
    start_time = time.time()
    move2 = agent2.genmove(None)
    time2 = time.time() - start_time
    
    print(f"Time: {time2:.2f}s")
    print(f"Move: {move2.sprint()}")
    
    print("\n📈 Results:")
    print(f"Low iterations ({iterations_low}):  {time1:.2f}s -> {move1.sprint()}")
    print(f"High iterations ({iterations_high}): {time2:.2f}s -> {move2.sprint()}")
    
    print("\n📊 Game State Analysis:")
    print(f"Initial state - Terminal: {state1.is_terminal()}")
    print(f"Self side turn: {state1.is_self_side_turn()}")
    
    moves = state1.actions_to_try()
    print(f"Available moves: {len(moves)}")
    
    # Show the board state
    print("\n🎮 Initial Board:")
    state1.print()
    
    print("\n💡 Key Observations:")
    print("   • Higher iterations generally lead to better move selection")
    print("   • MCTS explores the game tree more thoroughly with more time")
    print("   • Built-in C++ TicTacToe provides fast, reliable game mechanics")

if __name__ == "__main__":
    test_heuristic_enhancement()