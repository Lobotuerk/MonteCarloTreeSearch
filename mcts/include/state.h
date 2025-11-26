#ifndef MCTS_STATE_H
#define MCTS_STATE_H

#include <stdexcept>
#include <queue>
#include <iostream>
#include <string>


using namespace std;


struct MCTS_move {
    virtual ~MCTS_move() = default;
    virtual bool operator==(const MCTS_move& other) const = 0;             // implement this!
    virtual string sprint() const { return "Not implemented"; }   // and optionally this
};


/** Implement all pure virtual methods. Notes:
 * - rollout() must return something in [0, 1] for UCT to work as intended and specifically
 * the winning chance of the self side (the side making decisions).
 * - self side is determined by is_self_side_turn()
 * - supports 1 vs N player scenarios where self_side competes against other_side(s)
 * - minimal interface: only requires is_self_side_turn() for turn determination
 */
class MCTS_state {
public:
    // Implement these:
    virtual ~MCTS_state() = default;
    virtual queue<MCTS_move *> *actions_to_try() const = 0;
    virtual MCTS_state *next_state(const MCTS_move *move) const = 0;
    virtual double rollout() const = 0;
    virtual bool is_terminal() const = 0;
    virtual void print() const {
        cout << "Printing not implemented" << endl;
    }
    virtual bool is_self_side_turn() const = 0;     // true if it's the self side's turn
    
    // Deep copy method for C++ ownership transfer
    virtual MCTS_state* clone() const = 0;
    
    // Heuristic rollout support (optional override)
    virtual double heuristic_rollout() const {
        return rollout();  // Default to random rollout
    }
    
    // Move evaluation heuristic (optional override)
    virtual double evaluate_move(const MCTS_move* move) const {
        return 0.0;  // Default: no preference
    }
    
    // Position evaluation heuristic (optional override)
    virtual double evaluate_position() const {
        return 0.5;  // Default: neutral position
    }
};


#endif
