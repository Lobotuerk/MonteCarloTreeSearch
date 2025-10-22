#include "py_wrappers.h"
#include <iostream>

std::vector<MCTS_move*> queue_to_vector(std::queue<MCTS_move*>* q) {
    std::vector<MCTS_move*> result;
    if (q == nullptr) {
        return result;
    }
    
    while (!q->empty()) {
        result.push_back(q->front());
        q->pop();
    }
    delete q;  // Clean up the queue
    return result;
}

std::queue<MCTS_move*>* vector_to_queue(const std::vector<MCTS_move*>& vec) {
    auto* q = new std::queue<MCTS_move*>();
    for (MCTS_move* move : vec) {
        q->push(move);
    }
    return q;
}

SafeMCTS_agent::SafeMCTS_agent(MCTS_state* starting_state, int max_iter, int max_seconds) {
    agent = new MCTS_agent(starting_state, max_iter, max_seconds);
}

SafeMCTS_agent::~SafeMCTS_agent() {
    delete agent;
}

const MCTS_move* SafeMCTS_agent::genmove(const MCTS_move* enemy_move) {
    return agent->genmove(enemy_move);
}

const MCTS_state* SafeMCTS_agent::get_current_state() const {
    return agent->get_current_state();
}

void SafeMCTS_agent::feedback() const {
    agent->feedback();
}