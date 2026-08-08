#ifndef MCTS_PYTHON_H
#define MCTS_PYTHON_H

#include "state.h"
#include <vector>
#include <queue>
#include <iomanip>
#include <thread>
#include <future>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

#define STARTING_NUMBER_OF_CHILDREN 32   // expected number so that we can preallocate this many pointers
// #define PARALLEL_ROLLOUTS                // Enable parallel rollouts with std::thread (DISABLED due to destructor issues)
#define DEFAULT_NUMBER_OF_THREADS 1      // Default number of parallel rollout threads (disabled)

using namespace std;

/** Ideas for improvements:
 * - state should probably be const like move is (currently problematic because of Quoridor's example)
 * - Instead of a FIFO Queue use a Priority Queue with priority on most probable (better) actions to be explored first
  or maybe this should just be an iterable and we let the implementation decide but these have no superclasses in C++ it seems
 * - vectors, queues and these structures allocate data on the heap anyway so there is little point in using the heap for them
 * so use stack instead?
 */

class SearchThreadPool;

class MCTS_node {
    bool terminal;
    unsigned int size;
    unsigned int number_of_simulations;
    double score;                       // e.g. number of wins (could be int but double is more general if we use evaluation functions)
    double prior_probability;           // prior probability for PUCT
    MCTS_state *state;                  // current state
    const MCTS_move *move;              // move to get here from parent node's state
    mutable vector<MCTS_node *> children;
    MCTS_node *parent;
    queue<MCTS_move *> untried_actions;
    vector<double> action_probabilities; // stored probabilities for untried actions
    bool owns_state;                    // true if this node should delete the state in destructor
    bool is_evaluated;                  // true if this node has received its neural network evaluation (AlphaZero-style)
    void backpropagate(double w, int n);
    
    // Configuration for parallel rollouts
    static unsigned int num_rollout_threads;
    
    friend class SearchThreadPool;
    friend class MCTS_tree;
    
public:
    MCTS_node(MCTS_node *parent, MCTS_state *state, const MCTS_move *move, bool owns_state = true, double prior_probability = 1.0);
    ~MCTS_node();
    bool is_fully_expanded() const;
    bool is_terminal() const;
    const MCTS_move *get_move() const;
    unsigned int get_size() const;
    double get_prior_probability() const { return prior_probability; }
    unsigned int get_number_of_simulations() const { return number_of_simulations; }
    double get_score() const { return score; }
    MCTS_node *get_parent() const { return parent; }
    const vector<MCTS_node *> &get_children() const { return children; }
    void expand();
    void rollout();
    MCTS_node *select_best_child(double c) const;
    MCTS_node *advance_tree(const MCTS_move *m);
    const MCTS_state *get_current_state() const;
    MCTS_state *get_state() { return state; }
    void print_stats() const;
    double calculate_winrate(bool player1turn) const;
    
    // Static method to configure parallel rollouts
    static void set_rollout_threads(unsigned int num_threads);
    static unsigned int get_rollout_threads();
    
    // Virtual loss: apply/reverse virtual loss along the parent chain to prevent thread collisions
    void apply_virtual_loss(double v);
    void remove_virtual_loss(double v);
    
    // AlphaZero-style expansion: instantiate children from returned priors
    void expand_with_priors(const std::vector<double>& priors);
};

class MCTS_tree;

class SearchThreadPool;

/**
 * SearchThreadPool: manages T search threads for batched MCTS.
 * Threads walk the tree, apply virtual losses, and push leaves to evaluation_queue.
 * They block on a condvar barrier until the main thread finishes evaluation.
 */
class SearchThreadPool {
    std::vector<std::thread> threads;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::condition_variable all_parked_cv;
    std::vector<MCTS_node*> evaluation_queue;
    
    // State
    MCTS_tree* tree;
    double exploration_constant;
    int num_threads;
    std::atomic<bool> paused{false};
    std::atomic<int> parked_count{0};
    std::atomic<bool> stop_flag{false};
    
    // Callback that each thread runs
    std::function<void(MCTS_node*)> search_fn;
    
    void search_thread_func();
    
public:
    SearchThreadPool(MCTS_tree* tree, double exploration_constant, int num_threads);
    ~SearchThreadPool();
    
    // Signal threads to start searching
    void start();
    // Signal threads to pause and wait
    void pause();
    // Wait until all threads are parked (waiting on the condvar)
    bool wait_all_parked(int timeout_ms);
    // Wake all threads
    void resume();
    // Stop all threads
    void stop_threads();
    // Get the evaluation queue (thread-safe copy)
    std::vector<MCTS_node*> take_queue();
    // Add a node to the queue (thread-safe)
    void add_to_queue(MCTS_node* node);
    // Check if all threads are stopped
    bool is_stopped();
    int get_parked_count();
    int get_num_threads();
};

class MCTS_tree {
    MCTS_node *root;
    int batch_size;
    int num_search_threads;
    double virtual_loss;
    friend class SearchThreadPool;
public:
    MCTS_tree(MCTS_state *starting_state);
    ~MCTS_tree();
    MCTS_node *select(double c=1.41);        // select child node to expand according to tree policy (UCT)
    MCTS_node *select_best_child();          // select the most promising child of the root node
    void grow_tree(int max_iter, double max_time_in_seconds, double exploration_constant = 1.41);
    void advance_tree(const MCTS_move *move);      // if the move is applicable advance the tree, else start over
    unsigned int get_size() const;
    const MCTS_state *get_current_state() const;
    void print_stats() const;
    MCTS_node *get_root() const { return root; }
    
    // Batched search configuration
    void set_batch_size(int size) { batch_size = size; }
    int get_batch_size() const { return batch_size; }
    void set_num_search_threads(int n) { num_search_threads = n; }
    int get_num_search_threads() const { return num_search_threads; }

    // Virtual loss configuration
    void set_virtual_loss(double v) { this->virtual_loss = v; }
    double get_virtual_loss() const { return this->virtual_loss; }
};

class MCTS_agent {                           // example of an agent based on the MCTS_tree. One can also use the tree directly.
    MCTS_tree *tree;
    int max_iter, max_seconds;
    double exploration_constant;
    
    // Batched search configuration
    int batch_size;
    int num_search_threads;
    
public:
    MCTS_agent(MCTS_state *starting_state, int max_iter = 100000, int max_seconds = 30, double exploration_constant = 1.41);
    ~MCTS_agent();
    const MCTS_move *genmove(const MCTS_move *enemy_move);
    const MCTS_state *get_current_state() const;
    MCTS_tree *get_tree() const { return tree; }
    void feedback() const { tree->print_stats(); }
    void set_exploration_constant(double c);
    double get_exploration_constant() const;
    
    // Configure parallel rollouts for this agent's tree
    void set_rollout_threads(unsigned int num_threads);
    unsigned int get_rollout_threads() const;
    
    // Batched search configuration
    void set_batch_size(int size) { batch_size = size; }
    int get_batch_size() const { return batch_size; }
   void set_num_search_threads(int n) { num_search_threads = n; }
    int get_num_search_threads() const { return num_search_threads; }

    // Virtual loss configuration (delegated to tree)
    void set_virtual_loss(double v) { tree->set_virtual_loss(v); }
    double get_virtual_loss() const { return tree->get_virtual_loss(); }
};

// Utility functions for parallel rollouts
namespace ParallelRollouts {
    // Perform a single rollout simulation (thread-safe)
    double perform_rollout(const MCTS_state* state);
    
    // Get optimal number of threads based on hardware
    unsigned int get_optimal_thread_count();
}

#endif