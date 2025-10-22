#ifndef PY_WRAPPERS_H
#define PY_WRAPPERS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../mcts/include/state.h"
#include "mcts_python.h"  // Use Python-specific header that conditionally includes JobScheduler

#include <vector>
#include <memory>
#include <queue>

// Forward declaration of RolloutStrategy
enum class RolloutStrategy {
    RANDOM,           // Pure random rollouts (default)
    HEURISTIC,        // Use heuristic_rollout() method
    MIXED,            // Mix of random and heuristic (configurable ratio)
    HEAVY             // Deeper heuristic evaluation
};

namespace py = pybind11;

/**
 * Trampoline class for MCTS_move to enable Python inheritance
 */
class PyMCTS_move : public MCTS_move {
public:
    using MCTS_move::MCTS_move;

    bool operator==(const MCTS_move& other) const override {
        try {
            py::function override_func = py::get_override(this, "__eq__");
            if (override_func) {
                return override_func(&other).cast<bool>();
            }
        } catch (const std::exception& e) {
            // Fallback
        }
        return false;  // Default fallback
    }

    std::string sprint() const override {
        try {
            py::function override_func = py::get_override(this, "sprint");
            if (override_func) {
                return override_func().cast<std::string>();
            }
        } catch (const std::exception& e) {
            // Fallback
        }
        return "Move";  // Default fallback
    }
};

/**
 * Trampoline class for MCTS_state to enable Python inheritance
 */
class PyMCTS_state : public MCTS_state {
public:
    using MCTS_state::MCTS_state;

    ~PyMCTS_state() override = default;

    std::queue<MCTS_move*>* actions_to_try() const override {
        try {
            // Call Python method that returns a list of moves
            py::function override_func = py::get_override(this, "actions_to_try");
            if (override_func) {
                py::list py_moves = override_func();
                
                // Convert Python list to C++ queue
                std::queue<MCTS_move*>* queue = new std::queue<MCTS_move*>();
                for (auto item : py_moves) {
                    MCTS_move* move = item.cast<MCTS_move*>();
                    queue->push(move);
                }
                return queue;
            }
        } catch (const std::exception& e) {
            // Fallback - return empty queue
            return new std::queue<MCTS_move*>();
        }
        return new std::queue<MCTS_move*>();
    }

    MCTS_state* next_state(const MCTS_move* move) const override {
        try {
            // Call Python method that returns a new state
            py::function override_func = py::get_override(this, "next_state");
            if (override_func) {
                py::object py_state = override_func(move);
                return py_state.cast<MCTS_state*>();
            }
        } catch (const std::exception& e) {
            // Fallback - return null
            return nullptr;
        }
        return nullptr;
    }

    double rollout() const override {
        try {
            py::function override_func = py::get_override(this, "rollout");
            if (override_func) {
                return override_func().cast<double>();
            }
        } catch (const std::exception& e) {
            // Fallback
        }
        return 0.5;  // Default fallback
    }

    bool is_terminal() const override {
        try {
            py::function override_func = py::get_override(this, "is_terminal");
            if (override_func) {
                return override_func().cast<bool>();
            }
        } catch (const std::exception& e) {
            // Fallback
        }
        return true;  // Default fallback
    }

    void print() const override {
        try {
            py::function override_func = py::get_override(this, "print");
            if (override_func) {
                override_func();
                return;
            }
        } catch (const std::exception& e) {
            // Fallback
        }
        // Default fallback - do nothing
    }

    bool player1_turn() const override {
        try {
            py::function override_func = py::get_override(this, "player1_turn");
            if (override_func) {
                return override_func().cast<bool>();
            }
        } catch (const std::exception& e) {
            // Fallback
        }
        return true;  // Default fallback
    }
};

/**
 * Helper function to convert queue<MCTS_move*>* to vector for Python
 * This takes ownership of the queue and all moves in it
 */
std::vector<MCTS_move*> queue_to_vector(std::queue<MCTS_move*>* q);

/**
 * Helper function to convert vector to queue<MCTS_move*>*
 * This creates new queue and transfers ownership of moves
 */
std::queue<MCTS_move*>* vector_to_queue(const std::vector<MCTS_move*>& vec);

/**
 * Safe wrapper for MCTS_agent that handles move ownership
 */
class SafeMCTS_agent {
private:
    MCTS_agent* agent;
    
public:
    SafeMCTS_agent(MCTS_state* starting_state, int max_iter = 100000, int max_seconds = 30);
    ~SafeMCTS_agent();
    
    // Returns nullptr if no move available (game ended)
    const MCTS_move* genmove(const MCTS_move* enemy_move = nullptr);
    const MCTS_state* get_current_state() const;
    void feedback() const;
};

#endif // PY_WRAPPERS_H