### 📋 Technical Specification

## 1. Overview
Introduce AlphaZero-style Batched MCTS to the `pybind` Python boundary. This replaces single-threaded, node-by-node rollouts with a batched queue to heavily amortize GPU/Python boundary costs. 

## 2. Scope & Target
- **Target Files**: `pybind/mcts_python.h`, `pybind/mcts_python.cpp`, `pybind/py_wrappers.h`, `pybind/py_wrappers.cpp`, `pybind/pymcts.cpp`
- **Exclusion**: Pure C++ core (`mcts/` directory) remains unchanged.

## 3. Architecture & Data Structures

### 3.1 The Python Boundary (`pybind/py_wrappers.h` & `py_wrappers.cpp`)
Add a new method to `SerializedPythonState` to cross the Python boundary once for a batch of states:
```cpp
std::vector<std::pair<double, std::vector<double>>> evaluate_batch(const std::vector<MCTS_state*>& states) const;
```
- **Implementation logic**: 
  - Check if `py::hasattr(python_state, "evaluate_batch")`.
  - If true, call it passing a `py::list` of the underlying Python state objects. Parse the returned list of tuples `(value, priors)` into the C++ `std::vector<std::pair<double, std::vector<double>>>`.
  - If false, throw an exception or return an empty vector to signal a fallback.

### 3.2 Tree & Nodes (`pybind/mcts_python.h` & `mcts_python.cpp`)
- **`MCTS_node` Updates**:
  - Add `void apply_virtual_loss()`: Adds `1` to `number_of_simulations` and subtracts `1.0` from `score` recursively up the `parent` chain.
  - Add `void remove_virtual_loss()`: Subtracts `1` from `number_of_simulations` and adds `1.0` to `score` recursively up the `parent` chain.
  - Add a flag `bool is_evaluated` (default `false`) to indicate if the node has received its neural network evaluation.
  - Update `is_fully_expanded()` to incorporate AlphaZero expansion logic (true if `is_evaluated` is true and `untried_actions` is empty).
  - Modify constructor: Do not call `get_action_probabilities()` upon instantiation if batching is configured.
  - Add `void expand_with_priors(const std::vector<double>& priors)`: Populates the priors and instantiates children directly from `untried_actions` if adopting full AlphaZero-style expansion, or prepares `untried_actions` for progressive expansion.

- **`MCTS_tree` & `MCTS_agent` Updates**:
  - Expose `int batch_size` and `int num_search_threads` as configurable parameters in `MCTS_agent` (defaults: `batch_size=64`, `num_search_threads=4`) and `pymcts.cpp`.
  - Introduce `SearchThreadPool`: A nested manager containing `std::vector<std::thread>`, `std::mutex`, `std::condition_variable`, and a `std::vector<MCTS_node*> evaluation_queue`.

### 3.3 The Batched State Machine (`MCTS_tree::grow_tree`)
Rewrite `grow_tree` to use a threaded, batched state machine:
1. **Graceful Fallback**: Check if the root state supports `evaluate_batch`. If not, fall back to the legacy sequential `for (int i=0; i<max_iter; i++) { ... }` loop.
2. **Batched State Machine Loop**:
   - `max_iter` now represents the **total number of leaf evaluations**.
   - Spawn `T` pure C++ threads (`num_search_threads`).
   - **Search Phase (Threads)**:
     - Each thread runs `select()` to find a leaf node.
     - If the leaf is terminal, backpropagate its true score immediately and continue to the next `select()`.
     - If it is an unexplored leaf:
       - Acquire lock, push to `evaluation_queue`.
       - Call `leaf->apply_virtual_loss()`.
       - Wait on a `condition_variable` (Pause Phase).
   - **Evaluate Phase (Main Thread)**:
     - Wait until `evaluation_queue.size() == batch_size` OR all `T` threads are waiting.
     - Extract `MCTS_state*` from each enqueued node.
     - Acquire the GIL (`py::gil_scoped_acquire`), cross the Python boundary via `evaluate_batch(states)`.
     - Release the GIL.
   - **Backpropagate & Resume Phase (Main Thread)**:
     - For each node and its corresponding `(value, priors)`:
       - `node->remove_virtual_loss()`
       - `node->expand_with_priors(priors)`
       - `node->backpropagate(value, 1)`
     - Clear the queue.
     - Wake up all waiting search threads via `condition_variable.notify_all()`.
   - Repeat until `total_leaves_evaluated >= max_iter`.

## 4. Nuances & Guarantees
- **Virtual Loss Isolation**: Virtual loss prevents duplicate walks down the exact same path in parallel. By removing it exactly before adding the true Neural Value, the node's statistics remain mathematically pure.
- **GIL & Concurrency**: Search threads run purely in C++ without the GIL. Only the main thread acquires the GIL during `evaluate_batch()`, guaranteeing perfect isolation.
- **Backward Compatibility**: Simple games without `evaluate_batch` perfectly route to the synchronous per-node loop.
- **Terminal Nodes**: Handled completely inline by the search threads to avoid wasting GPU FLOPs on known game-theoretic outcomes.