# Technical Specification for TFT-231: PUCT exploration

## File Structure Changes
- `mcts/include/mcts.h`: Modify `MCTS_tree` and `MCTS_agent` declarations.
- `mcts/src/mcts.cpp`: Update `grow_tree` definition and `MCTS_agent` methods.
- `pybind/mcts_python.h`: Modify python bindings for `MCTS_tree` and `MCTS_agent`.
- `pybind/mcts_python.cpp`: Update python bindings for `grow_tree` and `MCTS_agent`.
- `pybind/py_wrappers.h`: Update `SafeMCTS_agent` definition.
- `pybind/py_wrappers.cpp`: Update `SafeMCTS_agent` implementation.
- `pybind/pymcts.cpp`: Bind the exploration constant parameter to python's `SafeMCTS_agent` and `MCTS_tree`.

## Interfaces & Signatures
**mcts/include/mcts.h** & **pybind/mcts_python.h**
- Update `MCTS_tree::grow_tree`:
  ```cpp
  void grow_tree(int max_iter, double max_time_in_seconds, double exploration_constant = 1.41);
  ```
- Update `MCTS_agent` class:
  ```cpp
  double exploration_constant;
  MCTS_agent(MCTS_state *starting_state, int max_iter = 100000, int max_seconds = 30, double exploration_constant = 1.41);
  void set_exploration_constant(double c);
  double get_exploration_constant() const;
  ```

**mcts/src/mcts.cpp** & **pybind/mcts_python.cpp**
- Implement `grow_tree` to take `exploration_constant` and pass it to `select(c)`.
- Update `MCTS_agent::genmove` to pass `exploration_constant` to `tree->grow_tree(..., exploration_constant)`.
- Implement `set_exploration_constant` and `get_exploration_constant`.

**pybind/py_wrappers.h** & **pybind/py_wrappers.cpp**
- Update `SafeMCTS_agent` class:
  ```cpp
  SafeMCTS_agent(MCTS_state* starting_state, int max_iter = 100000, int max_seconds = 30, double exploration_constant = 1.41);
  void set_exploration_constant(double c) { agent->set_exploration_constant(c); }
  double get_exploration_constant() const { return agent->get_exploration_constant(); }
  ```

**pybind/pymcts.cpp**
- Update `SafeMCTS_agent` binding in python:
  ```cpp
  .def(py::init<MCTS_state*, int, int, double>(),
       "Create an MCTS agent with the given starting state and parameters",
       py::arg("starting_state"), py::arg("max_iter") = 100000, 
       py::arg("max_seconds") = 30, py::arg("exploration_constant") = 1.41)
  .def_property("exploration_constant", 
                &SafeMCTS_agent::get_exploration_constant,
                &SafeMCTS_agent::set_exploration_constant,
                "PUCT exploration constant (c)")
  ```
- Update `MCTS_tree::grow_tree` binding in python:
  ```cpp
  .def("grow_tree", &MCTS_tree::grow_tree,
       "Grow the tree for the specified iterations or time",
       py::arg("max_iter"), py::arg("max_time_in_seconds"), py::arg("c") = 1.41)
  ```

## Edge Cases
- **Backward Compatibility:** All added constructor parameters and function arguments default to `1.41`, matching the current hardcoded behavior. Existing instantiations won't break.
- **Constant Propagation:** `genmove` must explicitly pass the agent's `exploration_constant` to `tree->grow_tree`, which forwards it to `select(c)`.
- **Duplicate Code synchronization:** Both the pure C++ `mcts` code and the `pybind` python-specific C++ code must be updated identically since the repository duplicates the tree implementation. `SafeMCTS_agent` acts as a facade delegating `get/set_exploration_constant` directly to the underlying `MCTS_agent`.

## Testing Strategy
- Create a test python script (e.g., `tests/test_exploration.py`) that initializes `MCTS_agent(state, exploration_constant=2.5)`.
- Assert that `agent.exploration_constant == 2.5`.
- Change the exploration constant via the property `agent.exploration_constant = 1.0` and assert the new value is updated.
