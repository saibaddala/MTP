/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PYTREE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PYTREE_H_

// See https://jax.readthedocs.io/en/latest/pytrees.html for the documentation
// about pytree.

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "flat_hash_map.h"
#include "inlined_vector.h"
#include "hash.h"
#include "pybind11.h"  // from @pybind11
#include "pytypes.h"  // from @pybind11
#include "stl.h"  // from @pybind11  // IWYU pragma: keep
#include "pytree.pb.h"

namespace xla {

enum class PyTreeKind {
  kLeaf,        // An opaque leaf node
  kNone,        // None.
  kTuple,       // A tuple
  kNamedTuple,  // A collections.namedtuple
  kList,        // A list
  kDict,        // A dict
  kCustom,      // A custom type.
};

// Registry of custom node types.
class PyTreeRegistry : public std::enable_shared_from_this<PyTreeRegistry> {
 public:
  PyTreeRegistry(bool enable_none, bool enable_tuple, bool enable_namedtuple,
                 bool enable_list, bool enable_dict);
  struct Registration {
    PyTreeKind kind;

    // The following values are populated for custom types.
    // The Python type object, used to identify the type.
    pybind11::object type;
    // A function with signature: object -> (iterable, aux_data)
    pybind11::function to_iterable;
    // A function with signature: (aux_data, iterable) -> object
    pybind11::function from_iterable;
  };

  // Registers a new custom type. Objects of `type` will be treated as container
  // node types in PyTrees.
  void Register(pybind11::object type, pybind11::function to_iterable,
                pybind11::function from_iterable);

  // Finds the custom type registration for `type`. Returns nullptr if none
  // exists.
  const Registration* Lookup(pybind11::handle type) const;

  PyTreeKind KindOfObject(pybind11::handle obj,
                          PyTreeRegistry::Registration const** custom) const;

 private:
  struct TypeHash {
    using is_transparent = void;
    size_t operator()(const pybind11::object& t) const {
      return absl::HashOf(t.ptr());
    }
    size_t operator()(const pybind11::handle& t) const {
      return absl::HashOf(t.ptr());
    }
  };
  struct TypeEq {
    using is_transparent = void;
    bool operator()(const pybind11::object& a,
                    const pybind11::object& b) const {
      return a.ptr() == b.ptr();
    }
    bool operator()(const pybind11::object& a,
                    const pybind11::handle& b) const {
      return a.ptr() == b.ptr();
    }
  };
  absl::flat_hash_map<pybind11::object, std::unique_ptr<Registration>, TypeHash,
                      TypeEq>
      registrations_;
  bool enable_namedtuple_;
};

// Returns the default pytree registry.
std::shared_ptr<PyTreeRegistry> DefaultPyTreeRegistry();

// A PyTreeDef describes the tree structure of a PyTree. A PyTree is a tree of
// Python values, where the interior nodes are tuples, lists, dictionaries, or
// user-defined containers, and the leaves are other objects.
class PyTreeDef {
 public:
  // Unowned registry: the registry must remain live at least as long as the
  // PyTreeDef. It is the caller's responsibility to enforce this.
  explicit PyTreeDef(PyTreeRegistry* registry) : registry_(registry) {}

  explicit PyTreeDef(std::shared_ptr<PyTreeRegistry> registry)
      : registry_(registry.get()), registry_ref_(std::move(registry)) {}

  // Flattens a Pytree into a list of leaves and a PyTreeDef.
  // Returns references to the flattened objects, which might be temporary
  // objects in the case of custom pytype handlers.
  static std::pair<std::vector<pybind11::object>, std::unique_ptr<PyTreeDef>>
  Flatten(pybind11::handle x,
          std::optional<pybind11::function> leaf_predicate = std::nullopt,
          std::shared_ptr<PyTreeRegistry> registry = nullptr);

  // Flattens a Pytree into a list of `leaves` and a PyTreeDef (this).
  // `leaves` owns references to the flattened objects, which might be
  // temporary objects in the case of custom pytype handlers.
  void Flatten(pybind11::handle handle, std::vector<pybind11::object>& leaves,
               std::optional<pybind11::function> leaf_predicate = std::nullopt);
  void Flatten(pybind11::handle handle,
               absl::InlinedVector<pybind11::object, 2>& leaves,
               std::optional<pybind11::function> leaf_predicate = std::nullopt);

  // Tests whether the given list is a flat list of leaves.
  static bool AllLeaves(PyTreeRegistry* registry, const pybind11::iterable& x);

  // Flattens a Pytree up to this PyTreeDef. 'this' must be a tree prefix of
  // the tree-structure of 'x'. For example, if we flatten a value
  // [(1, (2, 3)), {"foo": 4}] with a treedef [(*, *), *], the result is the
  // list of leaves [1, (2, 3), {"foo": 4}].
  pybind11::list FlattenUpTo(pybind11::handle x) const;

  // Returns an unflattened PyTree given an iterable of leaves and a PyTreeDef.
  pybind11::object Unflatten(pybind11::iterable leaves) const;
  pybind11::object Unflatten(absl::Span<const pybind11::object> leaves) const;

  // Composes two PyTreeDefs, replacing the leaves of this tree with copies of
  // `inner`. The returned PyTreeDef holds a reference to its registry.
  std::unique_ptr<PyTreeDef> Compose(const PyTreeDef& inner) const;

  // Makes a Tuple PyTreeDef out of a vector of PyTreeDefs.
  static std::unique_ptr<PyTreeDef> Tuple(
      std::shared_ptr<PyTreeRegistry> registry,
      absl::Span<PyTreeDef* const> defs);

  // The returned PyTreeDefs hold a reference to the registry.
  std::vector<std::unique_ptr<PyTreeDef>> Children() const;

  // Maps a function over a PyTree structure, applying f_leaf to each leaf, and
  // f_node(node, node_data) to each container node.
  pybind11::object Walk(const pybind11::function& f_node,
                        pybind11::handle f_leaf,
                        pybind11::iterable leaves) const;

  // Given a tree of iterables with the same node/leaf structure as this PyTree,
  // build the corresponding PyTree.
  // TODO(phawkins): use flattening everywhere instead and delete this method.
  pybind11::object FromIterableTree(pybind11::handle xs) const;

  int num_leaves() const {
    if (traversal_.empty()) {
      return 0;
    }
    return traversal_.back().num_leaves;
  }

  int num_nodes() const { return traversal_.size(); }

  PyTreeRegistry* registry() const { return registry_; }

  size_t Hash() const;

  bool operator==(const PyTreeDef& other) const;
  bool operator!=(const PyTreeDef& other) const { return !(*this == other); }

  std::string ToString() const;

  // Transforms the PyTreeDef into a pickleable object. Used to implement
  // `PyTreeDef.__getstate__`.
  pybind11::object ToPickle() const;

  // Transforms the object returned by `ToPickleable()` back to PyTreeDef. Used
  // to implement `PyTreeDef.__setstate__`.
  static PyTreeDef FromPickle(pybind11::object pickleable);

  void SerializeTo(jax::PyTreeDefProto& result) const;

  static PyTreeDef DeserializeFrom(std::shared_ptr<PyTreeRegistry> registry,
                                   const jax::PyTreeDefProto& input);

  std::optional<std::pair<pybind11::type, pybind11::object>> GetNodeData()
      const;

  static PyTreeDef MakeFromNodeDataAndChildren(
      std::shared_ptr<PyTreeRegistry> registry,
      std::optional<std::pair<pybind11::type, pybind11::object>> node_data,
      pybind11::iterable children);

 private:
  void SetNumLeavesAndNumNodes();

  struct Node {
    PyTreeKind kind = PyTreeKind::kLeaf;

    // Arity for non-kLeaf types.
    int arity = 0;

    // Kind-specific auxiliary data. For a kNamedTuple, contains the tuple type
    // object. For a kDict, use `sorted_dict_keys` field below. For a kCustom
    // type, contains the auxiliary data returned by the `to_iterable` function.
    pybind11::object node_data;

    // Kind-specific auxiliary data specialized for kDict. Use a c++ vector
    // to hold the sorted dict keys instead of a py::list to avoid creating
    // a new python list object when flattening kDict. For deeply nested dict,
    // using c++ vector instead of py::list avoids creating too many python
    // objects that make python gc sweep slow.
    std::vector<pybind11::object> sorted_dict_keys;

    // Custom type registration. Must be null for non-custom types.
    const PyTreeRegistry::Registration* custom = nullptr;

    // Number of leaf nodes in the subtree rooted at this node.
    int num_leaves = 0;

    // Number of leaf and interior nodes in the subtree rooted at this node.
    int num_nodes = 0;
  };
  template <typename H>
  friend H AbslHashValue(H h, const Node& n);

  template <typename H>
  friend H AbslHashValue(H h, const PyTreeDef& t);

  // Helper that manufactures an instance of a node given its children.
  static pybind11::object MakeNode(const Node& node,
                                   absl::Span<pybind11::object> children);

  // Recursive helper used to implement FromIterableTree()
  pybind11::object FromIterableTreeHelper(
      pybind11::handle xs,
      absl::InlinedVector<PyTreeDef::Node, 1>::const_reverse_iterator* it)
      const;

  template <typename T>
  void FlattenImpl(pybind11::handle handle, T& leaves,
                   const std::optional<pybind11::function>& leaf_predicate);

  template <typename T>
  pybind11::object UnflattenImpl(T leaves) const;

  // Pytree registry. Not owned.
  PyTreeRegistry* registry_;
  // If this class holds a reference to `registry`, it is held by
  // `registry_ref_`.
  std::shared_ptr<PyTreeRegistry> registry_ref_;

  // Nodes, in a post-order traversal. We use an ordered traversal to minimize
  // allocations, and post-order corresponds to the order we need to rebuild the
  // tree structure.
  absl::InlinedVector<Node, 1> traversal_;
};

template <typename H>
H AbslHashValue(H h, const PyTreeDef::Node& n) {
  h = H::combine(std::move(h), n.kind, n.arity, n.custom);
  return h;
}

template <typename H>
H AbslHashValue(H h, const PyTreeDef& t) {
  h = H::combine(std::move(h), t.traversal_);
  return h;
}

// pybind11-index-annotation BEGIN
// refs {
//   module_path: "tensorflow/compiler/xla/python/xla.cc"
//   module_arg {}
// }
// pybind11-index-annotation END
void BuildPytreeSubmodule(pybind11::module& m);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PYTREE_H_
