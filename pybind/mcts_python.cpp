#include <iostream>
#include <cassert>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <thread>
#include <future>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "py_wrappers.h"
#include "mcts_python.h"

// #define DEBUG

using namespace std;

// Static member initialization
unsigned int MCTS_node::num_rollout_threads = DEFAULT_NUMBER_OF_THREADS;

/*** MCTS NODE ***/
MCTS_node::MCTS_node(MCTS_node *parent, MCTS_state *state, const MCTS_move *move, bool owns_state, double prior_probability)
        : parent(parent), state(state->clone()), move(move), score(0.0), number_of_simulations(0), size(0),
          owns_state(true), prior_probability(prior_probability), is_evaluated(false) {
    terminal = this->state->is_terminal();
    children.reserve(STARTING_NUMBER_OF_CHILDREN);
    auto* tmp = state->actions_to_try();
    untried_actions.swap(*tmp);
    delete tmp;
    
    bool has_batch_support = false;
    SerializedPythonState* sps_check = dynamic_cast<SerializedPythonState*>(this->state);
    if (sps_check != nullptr) {
        has_batch_support = py::hasattr(sps_check->get_python_state(), "evaluate_batch");
    }
    
    if (!untried_actions.empty()) {
        if (has_batch_support) {
            action_probabilities.assign(untried_actions.size(), 1.0);
        } else {
            vector<double> probs = this->state->get_action_probabilities();
            if (!probs.empty()) {
                vector<pair<double, MCTS_move*>> paired;
                size_t i = 0;
                while (!untried_actions.empty()) {
                    double p = (i < probs.size()) ? probs[i] : 1.0;
                    paired.push_back({p, untried_actions.front()});
                    untried_actions.pop();
                    i++;
                }
                sort(paired.begin(), paired.end(), [](const pair<double, MCTS_move*>& a, const pair<double, MCTS_move*>& b) {
                    return a.first > b.first;
                });
                for (auto& item : paired) {
                    untried_actions.push(item.second);
                    action_probabilities.push_back(item.first);
                }
            } else {
                action_probabilities.assign(untried_actions.size(), 1.0);
            }
        }
    }
}

MCTS_node::~MCTS_node() {
    if (owns_state) {
        delete state;
    }
    delete move;
    for (auto *child : children) {
        delete child;
    }
    while (!untried_actions.empty()) {
        delete untried_actions.front();
        untried_actions.pop();
    }
}

void MCTS_node::expand() {
    if (is_terminal()) {              // can legitimately happen in end-game situations
        rollout();                    // keep rolling out, eventually causing UCT to pick another node to expand due to exploration
        return;
    } else if (is_fully_expanded()) {
        cerr << "Warning: Cannot expanded this node any more!" << endl;
        return;
    }
    // get next untried action
    MCTS_move *next_move = untried_actions.front();
    untried_actions.pop();
    
    // get corresponding probability
    double prob = 1.0;
    if (!action_probabilities.empty()) {
        prob = action_probabilities[0];
        action_probabilities.erase(action_probabilities.begin());
    }
    
    MCTS_state *next_state = state->next_state(next_move);
    // build a new MCTS node from it
    MCTS_node *new_node = new MCTS_node(this, next_state, next_move, true, prob);  // Try to own, constructor will decide
    delete next_state; // Prevent memory leak since MCTS_node constructor clones it
    // rollout, updating its stats
    new_node->rollout();
    // add new node to tree
    children.push_back(new_node);
}

void MCTS_node::rollout() {
#ifdef PARALLEL_ROLLOUTS
    // Use std::thread for parallel rollouts
    if (num_rollout_threads <= 1) {
        // Single-threaded fallback
        double w = state->rollout();
        backpropagate(w, 1);
        return;
    }
    
    // Launch parallel rollout tasks
    std::vector<std::future<double>> futures;
    futures.reserve(num_rollout_threads);
    
    for (unsigned int i = 0; i < num_rollout_threads; i++) {
        futures.push_back(std::async(std::launch::async, [this]() {
            return ParallelRollouts::perform_rollout(this->state);
        }));
    }
    
    // Collect results
    double score_sum = 0.0;
    unsigned int successful_rollouts = 0;
    
    for (auto& future : futures) {
        try {
            double result = future.get();
            if (result >= 0.0 && result <= 1.0) {
                score_sum += result;
                successful_rollouts++;
            } else {
                cerr << "Warning: Invalid rollout result: " << result << endl;
            }
        } catch (const std::exception& e) {
            cerr << "Warning: Rollout threw exception: " << e.what() << endl;
        }
    }
    
    // Ensure we have at least one successful rollout
    if (successful_rollouts == 0) {
        cerr << "Warning: All parallel rollouts failed, falling back to single rollout" << endl;
        double w = state->rollout();
        backpropagate(w, 1);
    } else {
        backpropagate(score_sum, successful_rollouts);
    }
    
#else
    // Single-threaded rollouts
    double w = state->rollout();
    backpropagate(w, 1);
#endif
}

void MCTS_node::backpropagate(double w, int n) {
    score += w;
    number_of_simulations += n;
    if (parent != NULL) {
        parent->size++;
        parent->backpropagate(w, n);
    }
}

bool MCTS_node::is_fully_expanded() const {
    return is_terminal() || untried_actions.empty();
}

bool MCTS_node::is_terminal() const {
    return terminal;
}

unsigned int MCTS_node::get_size() const {
    return size;
}

MCTS_node *MCTS_node::select_best_child(double c) const {
    /** selects best child based on the winrate of whose turn it is to play */
    if (children.empty()) return NULL;
    else if (children.size() == 1) return children[0];
    else {
        double score, max = -1e20;
        MCTS_node *argmax = NULL;
        for (auto *child : children) {
            double winrate = 0.5;
            if (child->number_of_simulations > 0) {
                winrate = child->score / ((double) child->number_of_simulations);
            }
            // If it's not the self side's turn, apply UCT based on opponent winrate (our loss rate)
            if (!state->is_self_side_turn()){
                winrate = 1.0 - winrate;
            }
            if (c > 0) {
                // PUCT formula: Q + C * P * sqrt(ParentN) / (1 + ChildN)
                double exploration = c * child->prior_probability * sqrt((double)this->number_of_simulations) / (1.0 + (double)child->number_of_simulations);
                score = winrate + exploration;
            } else {
                score = winrate;
            }
            if (score > max) {
                max = score;
                argmax = child;
            }
        }
        return argmax;
    }
}

MCTS_node *MCTS_node::advance_tree(const MCTS_move *m) {
    // Find child with this m and delete all others
    MCTS_node *next = NULL;
    for (auto *child: children) {
        if (child->move->get_id() == m->get_id() && *(child->move) == *(m)) {
            next = child;
        } else {
            delete child;
        }
    }
    // remove children from queue so that they won't be re-deleted by the destructor when this node dies (!)
    children.clear();
    // if not found then we have to create a new node
    if (next == NULL) {
        // Note: UCT may lead to not fully explored tree even for short-term children due to terminal nodes being chosen
        cout << "INFO: Didn't find child node. Had to start over." << endl;
        MCTS_state *next_state = state->next_state(m);
        next = new MCTS_node(NULL, next_state, NULL, true, 1.0);  // Try to own, constructor will decide
    } else {
        next->parent = NULL;     // make parent NULL
        // IMPORTANT: m and next->move can be the same here if we pass the move from select_best_child()
        // (which is what we will typically be doing). If not then it's the caller's responsibility to delete m (!)
    }
    // return the next root
    return next;
}

/*** MCTS TREE ***/
MCTS_node *MCTS_tree::select(double c) {
    MCTS_node *node = root;
    while (!node->is_terminal()) {
        if (!node->is_fully_expanded()) {
            return node;
        } else {
            node = node->select_best_child(c);
            if (node == NULL) {
                break;
            }
        }
    }
    return node;
}

MCTS_tree::MCTS_tree(MCTS_state *starting_state)
    : batch_size(64), num_search_threads(4) {
    assert(starting_state != NULL);
    root = new MCTS_node(NULL, starting_state, NULL, false);
}

MCTS_tree::~MCTS_tree() {
    delete root;
}

void MCTS_tree::grow_tree(int max_iter, double max_time_in_seconds, double exploration_constant) {
    // Check if the root state supports batch evaluation
    bool has_batch_support = false;
    SerializedPythonState* root_sps_check = dynamic_cast<SerializedPythonState*>(root->get_state());
    if (root_sps_check != nullptr) {
        has_batch_support = py::hasattr(root_sps_check->get_python_state(), "evaluate_batch");
    }
    
    if (!has_batch_support) {
        // Graceful fallback: use legacy sequential loop
        MCTS_node *node;
        time_t start_t, now_t;
        time(&start_t);
        for (int i = 0; i < max_iter; i++) {
            node = select(exploration_constant);
            node->expand();
            time(&now_t);
            if (difftime(now_t, start_t) > max_time_in_seconds) break;
        }
        return;
    }
    
    // Batched state machine loop
    // max_iter = total number of leaf evaluations
    int total_evaluated = 0;
    time_t start_t, now_t;
    time(&start_t);
    
    // Spawn search threads with configured count
    SearchThreadPool pool(this, exploration_constant, num_search_threads);
    pool.start();
    
    while (total_evaluated < max_iter) {
        time(&now_t);
        if (difftime(now_t, start_t) > max_time_in_seconds) break;
        
        // Wait for the queue to fill up or all threads to be parked
        pool.wait_all_parked(5000);
        
        // Pause all threads to freeze the queue during evaluation
        pool.pause();
        
        // Take the queue
        auto nodes = pool.take_queue();
        
        if (nodes.empty()) {
            // Nothing to evaluate, resume and try again
            pool.resume();
            continue;
        }
        
        // Evaluate, Backpropagate & Expand phase: hold the GIL
        std::vector<std::pair<double, std::vector<double>>> results;
        {
            py::gil_scoped_acquire gil;
            // Build SerializedPythonStates for the batch
            std::vector<MCTS_state*> batch_states;
            for (size_t ni = 0; ni < nodes.size(); ni++) {
                MCTS_node* node = nodes[ni];
                SerializedPythonState* sps = dynamic_cast<SerializedPythonState*>(node->get_state());
                if (sps != nullptr) {
                    batch_states.push_back(sps);
                } else {
                    // Fallback: can't batch this state
                    batch_states.clear();
                    break;
                }
            }
            
            if (!batch_states.empty()) {
                // Use the root state's SerializedPythonState to call evaluate_batch
                SerializedPythonState* root_sps_for_eval = dynamic_cast<SerializedPythonState*>(root->get_state());
                if (root_sps_for_eval != nullptr) {
                    results = root_sps_for_eval->evaluate_batch(batch_states);
                }
            }
            
            // Backpropagate & expand phase
            size_t min_size = std::min(nodes.size(), results.size());
            for (size_t i = 0; i < min_size; i++) {
                MCTS_node* node = nodes[i];
                if (i < results.size()) {
                    double value = results[i].first;
                    const auto& priors = results[i].second;
                    
                    // Undo virtual loss first
                    node->remove_virtual_loss();
                    
                    // Expand with priors (AlphaZero-style, calls state->next_state which requires GIL)
                    node->expand_with_priors(priors);
                    
                    // Backpropagate the true neural value
                    node->backpropagate(value, 1);
                }
            }
            total_evaluated += min_size;
        }
        
        // Clear queue and resume threads
        pool.resume();
    }
    
  // Stop the pool
    pool.stop_threads();
}

unsigned int MCTS_tree::get_size() const {
    return root->get_size();
}

const MCTS_move *MCTS_node::get_move() const {
    return move;
}

const MCTS_state *MCTS_node::get_current_state() const { return state; }

void MCTS_node::set_rollout_threads(unsigned int num_threads) {
    if (num_threads == 0) {
        num_rollout_threads = 1;  // Minimum 1 thread
    } else {
        num_rollout_threads = num_threads;
    }
}

unsigned int MCTS_node::get_rollout_threads() {
    return num_rollout_threads;
}

void MCTS_node::apply_virtual_loss() {
    // Apply virtual loss along the entire root->leaf path
    MCTS_node* node = this;
    while (node != NULL) {
        node->number_of_simulations += 1;
        node->score -= 1.0;
        node = node->parent;
    }
}

void MCTS_node::remove_virtual_loss() {
    // Remove virtual loss along the entire root->leaf path
    MCTS_node* node = this;
    while (node != NULL) {
        node->number_of_simulations -= 1;
        node->score += 1.0;
        node = node->parent;
    }
}

void MCTS_node::expand_with_priors(const vector<double>& priors) {
    is_evaluated = true;
    
    if (!untried_actions.empty()) {
        size_t i = 0;
        while (!untried_actions.empty()) {
            MCTS_move* next_move = untried_actions.front();
            untried_actions.pop();
            
            double prob = 1.0;
            if (i < priors.size()) {
                prob = priors[i];
            }
            
            MCTS_state* next_state = state->next_state(next_move);
            MCTS_node* child = new MCTS_node(this, next_state, next_move, true, prob);
            delete next_state;
            
            children.push_back(child);
            i++;
        }
    }
}

void MCTS_node::print_stats() const {
    #define TOPK 10
    if (number_of_simulations == 0) {
        cout << "Tree not expanded yet" << endl;
        return;
    }
    cout << "___ INFO _______________________" << endl
         << "Tree size: " << size << endl
         << "Number of simulations: " << number_of_simulations << endl
         << "Branching factor at root: " << children.size() << endl
         << "Chances of self side winning: " << setprecision(4) << 100.0 * (score / number_of_simulations) << "%" << endl;
    // sort children based on winrate of current player's turn for this node
    if (state->is_self_side_turn()) {
        std::sort(children.begin(), children.end(), [](const MCTS_node *n1, const MCTS_node *n2){
            return n1->calculate_winrate(true) > n2->calculate_winrate(true);
        });
    } else {
        std::sort(children.begin(), children.end(), [](const MCTS_node *n1, const MCTS_node *n2){
            return n1->calculate_winrate(false) > n2->calculate_winrate(false);
        });
    }
    // print TOPK of them along with their winrates
    cout << "Best moves:" << endl;
    for (int i = 0 ; i < children.size() && i < TOPK ; i++) {
        cout << "  " << i + 1 << ". " << children[i]->move->sprint() << "  -->  "
             << setprecision(4) << 100.0 * children[i]->calculate_winrate(state->is_self_side_turn()) << "%" << endl;
    }
    cout << "________________________________" << endl;
}

double MCTS_node::calculate_winrate(bool player1turn) const {
    if (player1turn) {
        return score / number_of_simulations;
    } else {
        return 1.0 - score / number_of_simulations;
    }
}

void MCTS_tree::advance_tree(const MCTS_move *move) {
    MCTS_node *old_root = root;
    root = root->advance_tree(move);
    delete old_root;       // this won't delete the new root since we have emptied old_root's children
}

const MCTS_state *MCTS_tree::get_current_state() const { return root->get_current_state(); }

MCTS_node *MCTS_tree::select_best_child() {
    return root->select_best_child(0.0);
}

void MCTS_tree::print_stats() const { root->print_stats(); }

/*** SearchThreadPool implementation ***/

SearchThreadPool::SearchThreadPool(MCTS_tree* tree, double exploration_constant, int num_threads)
    : tree(tree), exploration_constant(exploration_constant), num_threads(num_threads),
      paused(false), parked_count(0), stop_flag(false) {
}

SearchThreadPool::~SearchThreadPool() {
    stop_threads();
}

void SearchThreadPool::search_thread_func() {
    while (!stop_flag.load()) {
        // Wait if the pool is paused
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv.wait(lock, [this] { return !paused.load() || stop_flag.load(); });
        }
        
        if (stop_flag.load()) break;
        
        // Search phase: walk the tree, find a leaf to evaluate
        MCTS_node* node = tree->select(exploration_constant);
        
        if (node == NULL) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // If terminal, backpropagate immediately and continue
        if (node->is_terminal()) {
            // Terminal node: compute true value and backpropagate
            double true_value = node->state->rollout();
            node->backpropagate(true_value, 1);
            continue;
        }
        
        // Unexplored leaf: apply virtual loss and enqueue
        node->apply_virtual_loss();
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            evaluation_queue.push_back(node);
            if (evaluation_queue.size() >= (size_t)tree->get_batch_size()) {
                paused.store(true);
            }
        }
        
        // Signal that a thread is parked and wait for resume/evaluation
        parked_count.fetch_add(1, std::memory_order_relaxed);
        cv.notify_all();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv.wait(lock, [this] { return !paused.load() || stop_flag.load(); });
        }
        
        parked_count.fetch_sub(1, std::memory_order_relaxed);
    }
}

void SearchThreadPool::start() {
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(&SearchThreadPool::search_thread_func, this);
    }
}

void SearchThreadPool::pause() {
    paused.store(true);
    cv.notify_all();
}

bool SearchThreadPool::wait_all_parked(int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (evaluation_queue.size() >= (size_t)tree->get_batch_size()) {
                return true;
            }
        }
        if (parked_count.load() >= num_threads) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return (parked_count.load() >= num_threads || evaluation_queue.size() > 0);
}

void SearchThreadPool::resume() {
    paused.store(false);
    cv.notify_all();
}

void SearchThreadPool::stop_threads() {
    stop_flag.store(true);
    paused.store(false);
    cv.notify_all();
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    threads.clear();
}

std::vector<MCTS_node*> SearchThreadPool::take_queue() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    std::vector<MCTS_node*> result = evaluation_queue;
    evaluation_queue.clear();
    return result;
}

void SearchThreadPool::add_to_queue(MCTS_node* node) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    evaluation_queue.push_back(node);
}

bool SearchThreadPool::is_stopped() {
    return stop_flag.load();
}

int SearchThreadPool::get_parked_count() {
    return parked_count.load();
}

int SearchThreadPool::get_num_threads() {
    return num_threads;
}

/*** MCTS agent ***/
MCTS_agent::MCTS_agent(MCTS_state *starting_state, int max_iter, int max_seconds, double exploration_constant)
: max_iter(max_iter), max_seconds(max_seconds), exploration_constant(exploration_constant),
  batch_size(64), num_search_threads(4) {
    tree = new MCTS_tree(starting_state);
}

const MCTS_move *MCTS_agent::genmove(const MCTS_move *enemy_move) {
    if (enemy_move != NULL) {
        tree->advance_tree(enemy_move);
    }
    // If game ended from opponent move, we can't do anything
    if (tree->get_current_state()->is_terminal()) {
        return NULL;
    }
    // Pass batch config to tree
    tree->set_batch_size(batch_size);
    tree->set_num_search_threads(num_search_threads);
    #ifdef DEBUG
    cout << "___ DEBUG ______________________" << endl
         << "Growing tree..." << endl;
    #endif
    tree->grow_tree(max_iter, max_seconds, exploration_constant);
    #ifdef DEBUG
    cout << "Tree size: " << tree->get_size() << endl
         << "________________________________" << endl;
    #endif
    MCTS_node *best_child = tree->select_best_child();
    if (best_child == NULL) {
        cerr << "Warning: Tree root has no children! Possibly terminal node!" << endl;
        return NULL;
    }
    const MCTS_move *best_move = best_child->get_move();
    tree->advance_tree(best_move);
    return best_move;
}

MCTS_agent::~MCTS_agent() {
    delete tree;
}

const MCTS_state *MCTS_agent::get_current_state() const { return tree->get_current_state(); }

void MCTS_agent::set_rollout_threads(unsigned int num_threads) {
    MCTS_node::set_rollout_threads(num_threads);
}

unsigned int MCTS_agent::get_rollout_threads() const {
    return MCTS_node::get_rollout_threads();
}

void MCTS_agent::set_exploration_constant(double c) {
    exploration_constant = c;
}

double MCTS_agent::get_exploration_constant() const {
    return exploration_constant;
}

// Parallel rollout utility functions
namespace ParallelRollouts {
    double perform_rollout(const MCTS_state* state) {
        // Thread-safe rollout execution
        return state->rollout();
    }
    
    unsigned int get_optimal_thread_count() {
        unsigned int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) {
            return DEFAULT_NUMBER_OF_THREADS;  // Fallback if detection fails
        }
        // Use hardware threads but cap at reasonable limit
        return std::min(hw_threads, 8u);
    }
}