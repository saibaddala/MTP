// Copyright 2018 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// File: flat_hash_set.h
// -----------------------------------------------------------------------------
//
// An `absl::flat_hash_set<T>` is an unordered associative container designed to
// be a more efficient replacement for `std::unordered_set`. Like
// `unordered_set`, search, insertion, and deletion of set elements can be done
// as an `O(1)` operation. However, `flat_hash_set` (and other unordered
// associative containers known as the collection of Abseil "Swiss tables")
// contain other optimizations that result in both memory and computation
// advantages.
//
// In most cases, your default choice for a hash set should be a set of type
// `flat_hash_set`.
#ifndef ABSL_CONTAINER_FLAT_HASH_SET_H_
#define ABSL_CONTAINER_FLAT_HASH_SET_H_

#include <type_traits>
#include <utility>

#include "container.h"
#include "macros.h"
#include "container_memory.h"
#include "hash_function_defaults.h"  // IWYU pragma: export
#include "raw_hash_set.h"  // IWYU pragma: export
#include "memory.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace container_internal {
template <typename T>
struct FlatHashSetPolicy;
}  // namespace container_internal

// -----------------------------------------------------------------------------
// absl::flat_hash_set
// -----------------------------------------------------------------------------
//
// An `absl::flat_hash_set<T>` is an unordered associative container which has
// been optimized for both speed and memory footprint in most common use cases.
// Its interface is similar to that of `std::unordered_set<T>` with the
// following notable differences:
//
// * Requires keys that are CopyConstructible
// * Supports heterogeneous lookup, through `find()` and `insert()`, provided
//   that the set is provided a compatible heterogeneous hashing function and
//   equality operator.
// * Invalidates any references and pointers to elements within the table after
//   `rehash()`.
// * Contains a `capacity()` member function indicating the number of element
//   slots (open, deleted, and empty) within the hash set.
// * Returns `void` from the `erase(iterator)` overload.
//
// By default, `flat_hash_set` uses the `absl::Hash` hashing framework. All
// fundamental and Abseil types that support the `absl::Hash` framework have a
// compatible equality operator for comparing insertions into `flat_hash_set`.
// If your type is not yet supported by the `absl::Hash` framework, see
// absl/hash/hash.h for information on extending Abseil hashing to user-defined
// types.
//
// Using `absl::flat_hash_set` at interface boundaries in dynamically loaded
// libraries (e.g. .dll, .so) is unsupported due to way `absl::Hash` values may
// be randomized across dynamically loaded libraries.
//
// NOTE: A `flat_hash_set` stores its keys directly inside its implementation
// array to avoid memory indirection. Because a `flat_hash_set` is designed to
// move data when rehashed, set keys will not retain pointer stability. If you
// require pointer stability, consider using
// `absl::flat_hash_set<std::unique_ptr<T>>`. If your type is not moveable and
// you require pointer stability, consider `absl::node_hash_set` instead.
//
// Example:
//
//   // Create a flat hash set of three strings
//   absl::flat_hash_set<std::string> ducks =
//     {"huey", "dewey", "louie"};
//
//  // Insert a new element into the flat hash set
//  ducks.insert("donald");
//
//  // Force a rehash of the flat hash set
//  ducks.rehash(0);
//
//  // See if "dewey" is present
//  if (ducks.contains("dewey")) {
//    std::cout << "We found dewey!" << std::endl;
//  }
template <class T, class Hash = absl::container_internal::hash_default_hash<T>,
          class Eq = absl::container_internal::hash_default_eq<T>,
          class Allocator = std::allocator<T>>
class flat_hash_set
    : public absl::container_internal::raw_hash_set<
          absl::container_internal::FlatHashSetPolicy<T>, Hash, Eq, Allocator> {
  using Base = typename flat_hash_set::raw_hash_set;

 public:
  // Constructors and Assignment Operators
  //
  // A flat_hash_set supports the same overload set as `std::unordered_set`
  // for construction and assignment:
  //
  // *  Default constructor
  //
  //    // No allocation for the table's elements is made.
  //    absl::flat_hash_set<std::string> set1;
  //
  // * Initializer List constructor
  //
  //   absl::flat_hash_set<std::string> set2 =
  //       {{"huey"}, {"dewey"}, {"louie"},};
  //
  // * Copy constructor
  //
  //   absl::flat_hash_set<std::string> set3(set2);
  //
  // * Copy assignment operator
  //
  //  // Hash functor and Comparator are copied as well
  //  absl::flat_hash_set<std::string> set4;
  //  set4 = set3;
  //
  // * Move constructor
  //
  //   // Move is guaranteed efficient
  //   absl::flat_hash_set<std::string> set5(std::move(set4));
  //
  // * Move assignment operator
  //
  //   // May be efficient if allocators are compatible
  //   absl::flat_hash_set<std::string> set6;
  //   set6 = std::move(set5);
  //
  // * Range constructor
  //
  //   std::vector<std::string> v = {"a", "b"};
  //   absl::flat_hash_set<std::string> set7(v.begin(), v.end());
  flat_hash_set() {}
  using Base::Base;

  // flat_hash_set::begin()
  //
  // Returns an iterator to the beginning of the `flat_hash_set`.
  using Base::begin;

  // flat_hash_set::cbegin()
  //
  // Returns a const iterator to the beginning of the `flat_hash_set`.
  using Base::cbegin;

  // flat_hash_set::cend()
  //
  // Returns a const iterator to the end of the `flat_hash_set`.
  using Base::cend;

  // flat_hash_set::end()
  //
  // Returns an iterator to the end of the `flat_hash_set`.
  using Base::end;

  // flat_hash_set::capacity()
  //
  // Returns the number of element slots (assigned, deleted, and empty)
  // available within the `flat_hash_set`.
  //
  // NOTE: this member function is particular to `absl::flat_hash_set` and is
  // not provided in the `std::unordered_set` API.
  using Base::capacity;

  // flat_hash_set::empty()
  //
  // Returns whether or not the `flat_hash_set` is empty.
  using Base::empty;

  // flat_hash_set::max_size()
  //
  // Returns the largest theoretical possible number of elements within a
  // `flat_hash_set` under current memory constraints. This value can be thought
  // of the largest value of `std::distance(begin(), end())` for a
  // `flat_hash_set<T>`.
  using Base::max_size;

  // flat_hash_set::size()
  //
  // Returns the number of elements currently within the `flat_hash_set`.
  using Base::size;

  // flat_hash_set::clear()
  //
  // Removes all elements from the `flat_hash_set`. Invalidates any references,
  // pointers, or iterators referring to contained elements.
  //
  // NOTE: this operation may shrink the underlying buffer. To avoid shrinking
  // the underlying buffer call `erase(begin(), end())`.
  using Base::clear;

  // flat_hash_set::erase()
  //
  // Erases elements within the `flat_hash_set`. Erasing does not trigger a
  // rehash. Overloads are listed below.
  //
  // void erase(const_iterator pos):
  //
  //   Erases the element at `position` of the `flat_hash_set`, returning
  //   `void`.
  //
  //   NOTE: returning `void` in this case is different than that of STL
  //   containers in general and `std::unordered_set` in particular (which
  //   return an iterator to the element following the erased element). If that
  //   iterator is needed, simply post increment the iterator:
  //
  //     set.erase(it++);
  //
  // iterator erase(const_iterator first, const_iterator last):
  //
  //   Erases the elements in the open interval [`first`, `last`), returning an
  //   iterator pointing to `last`.
  //
  // size_type erase(const key_type& key):
  //
  //   Erases the element with the matching key, if it exists, returning the
  //   number of elements erased (0 or 1).
  using Base::erase;

  // flat_hash_set::insert()
  //
  // Inserts an element of the specified value into the `flat_hash_set`,
  // returning an iterator pointing to the newly inserted element, provided that
  // an element with the given key does not already exist. If rehashing occurs
  // due to the insertion, all iterators are invalidated. Overloads are listed
  // below.
  //
  // std::pair<iterator,bool> insert(const T& value):
  //
  //   Inserts a value into the `flat_hash_set`. Returns a pair consisting of an
  //   iterator to the inserted element (or to the element that prevented the
  //   insertion) and a bool denoting whether the insertion took place.
  //
  // std::pair<iterator,bool> insert(T&& value):
  //
  //   Inserts a moveable value into the `flat_hash_set`. Returns a pair
  //   consisting of an iterator to the inserted element (or to the element that
  //   prevented the insertion) and a bool denoting whether the insertion took
  //   place.
  //
  // iterator insert(const_iterator hint, const T& value):
  // iterator insert(const_iterator hint, T&& value):
  //
  //   Inserts a value, using the position of `hint` as a non-binding suggestion
  //   for where to begin the insertion search. Returns an iterator to the
  //   inserted element, or to the existing element that prevented the
  //   insertion.
  //
  // void insert(InputIterator first, InputIterator last):
  //
  //   Inserts a range of values [`first`, `last`).
  //
  //   NOTE: Although the STL does not specify which element may be inserted if
  //   multiple keys compare equivalently, for `flat_hash_set` we guarantee the
  //   first match is inserted.
  //
  // void insert(std::initializer_list<T> ilist):
  //
  //   Inserts the elements within the initializer list `ilist`.
  //
  //   NOTE: Although the STL does not specify which element may be inserted if
  //   multiple keys compare equivalently within the initializer list, for
  //   `flat_hash_set` we guarantee the first match is inserted.
  using Base::insert;

  // flat_hash_set::emplace()
  //
  // Inserts an element of the specified value by constructing it in-place
  // within the `flat_hash_set`, provided that no element with the given key
  // already exists.
  //
  // The element may be constructed even if there already is an element with the
  // key in the container, in which case the newly constructed element will be
  // destroyed immediately.
  //
  // If rehashing occurs due to the insertion, all iterators are invalidated.
  using Base::emplace;

  // flat_hash_set::emplace_hint()
  //
  // Inserts an element of the specified value by constructing it in-place
  // within the `flat_hash_set`, using the position of `hint` as a non-binding
  // suggestion for where to begin the insertion search, and only inserts
  // provided that no element with the given key already exists.
  //
  // The element may be constructed even if there already is an element with the
  // key in the container, in which case the newly constructed element will be
  // destroyed immediately.
  //
  // If rehashing occurs due to the insertion, all iterators are invalidated.
  using Base::emplace_hint;

  // flat_hash_set::extract()
  //
  // Extracts the indicated element, erasing it in the process, and returns it
  // as a C++17-compatible node handle. Overloads are listed below.
  //
  // node_type extract(const_iterator position):
  //
  //   Extracts the element at the indicated position and returns a node handle
  //   owning that extracted data.
  //
  // node_type extract(const key_type& x):
  //
  //   Extracts the element with the key matching the passed key value and
  //   returns a node handle owning that extracted data. If the `flat_hash_set`
  //   does not contain an element with a matching key, this function returns an
  //   empty node handle.
  using Base::extract;

  // flat_hash_set::merge()
  //
  // Extracts elements from a given `source` flat hash set into this
  // `flat_hash_set`. If the destination `flat_hash_set` already contains an
  // element with an equivalent key, that element is not extracted.
  using Base::merge;

  // flat_hash_set::swap(flat_hash_set& other)
  //
  // Exchanges the contents of this `flat_hash_set` with those of the `other`
  // flat hash set, avoiding invocation of any move, copy, or swap operations on
  // individual elements.
  //
  // All iterators and references on the `flat_hash_set` remain valid, excepting
  // for the past-the-end iterator, which is invalidated.
  //
  // `swap()` requires that the flat hash set's hashing and key equivalence
  // functions be Swappable, and are exchaged using unqualified calls to
  // non-member `swap()`. If the set's allocator has
  // `std::allocator_traits<allocator_type>::propagate_on_container_swap::value`
  // set to `true`, the allocators are also exchanged using an unqualified call
  // to non-member `swap()`; otherwise, the allocators are not swapped.
  using Base::swap;

  // flat_hash_set::rehash(count)
  //
  // Rehashes the `flat_hash_set`, setting the number of slots to be at least
  // the passed value. If the new number of slots increases the load factor more
  // than the current maximum load factor
  // (`count` < `size()` / `max_load_factor()`), then the new number of slots
  // will be at least `size()` / `max_load_factor()`.
  //
  // To force a rehash, pass rehash(0).
  //
  // NOTE: unlike behavior in `std::unordered_set`, references are also
  // invalidated upon a `rehash()`.
  using Base::rehash;

  // flat_hash_set::reserve(count)
  //
  // Sets the number of slots in the `flat_hash_set` to the number needed to
  // accommodate at least `count` total elements without exceeding the current
  // maximum load factor, and may rehash the container if needed.
  using Base::reserve;

  // flat_hash_set::contains()
  //
  // Determines whether an element comparing equal to the given `key` exists
  // within the `flat_hash_set`, returning `true` if so or `false` otherwise.
  using Base::contains;

  // flat_hash_set::count(const Key& key) const
  //
  // Returns the number of elements comparing equal to the given `key` within
  // the `flat_hash_set`. note that this function will return either `1` or `0`
  // since duplicate elements are not allowed within a `flat_hash_set`.
  using Base::count;

  // flat_hash_set::equal_range()
  //
  // Returns a closed range [first, last], defined by a `std::pair` of two
  // iterators, containing all elements with the passed key in the
  // `flat_hash_set`.
  using Base::equal_range;

  // flat_hash_set::find()
  //
  // Finds an element with the passed `key` within the `flat_hash_set`.
  using Base::find;

  // flat_hash_set::bucket_count()
  //
  // Returns the number of "buckets" within the `flat_hash_set`. Note that
  // because a flat hash set contains all elements within its internal storage,
  // this value simply equals the current capacity of the `flat_hash_set`.
  using Base::bucket_count;

  // flat_hash_set::load_factor()
  //
  // Returns the current load factor of the `flat_hash_set` (the average number
  // of slots occupied with a value within the hash set).
  using Base::load_factor;

  // flat_hash_set::max_load_factor()
  //
  // Manages the maximum load factor of the `flat_hash_set`. Overloads are
  // listed below.
  //
  // float flat_hash_set::max_load_factor()
  //
  //   Returns the current maximum load factor of the `flat_hash_set`.
  //
  // void flat_hash_set::max_load_factor(float ml)
  //
  //   Sets the maximum load factor of the `flat_hash_set` to the passed value.
  //
  //   NOTE: This overload is provided only for API compatibility with the STL;
  //   `flat_hash_set` will ignore any set load factor and manage its rehashing
  //   internally as an implementation detail.
  using Base::max_load_factor;

  // flat_hash_set::get_allocator()
  //
  // Returns the allocator function associated with this `flat_hash_set`.
  using Base::get_allocator;

  // flat_hash_set::hash_function()
  //
  // Returns the hashing function used to hash the keys within this
  // `flat_hash_set`.
  using Base::hash_function;

  // flat_hash_set::key_eq()
  //
  // Returns the function used for comparing keys equality.
  using Base::key_eq;
};

// erase_if(flat_hash_set<>, Pred)
//
// Erases all elements that satisfy the predicate `pred` from the container `c`.
// Returns the number of erased elements.
template <typename T, typename H, typename E, typename A, typename Predicate>
typename flat_hash_set<T, H, E, A>::size_type erase_if(
    flat_hash_set<T, H, E, A>& c, Predicate pred) {
  return container_internal::EraseIf(pred, &c);
}

namespace container_internal {

template <class T>
struct FlatHashSetPolicy {
  using slot_type = T;
  using key_type = T;
  using init_type = T;
  using constant_iterators = std::true_type;

  template <class Allocator, class... Args>
  static void construct(Allocator* alloc, slot_type* slot, Args&&... args) {
    absl::allocator_traits<Allocator>::construct(*alloc, slot,
                                                 std::forward<Args>(args)...);
  }

  template <class Allocator>
  static void destroy(Allocator* alloc, slot_type* slot) {
    absl::allocator_traits<Allocator>::destroy(*alloc, slot);
  }

  static T& element(slot_type* slot) { return *slot; }

  template <class F, class... Args>
  static decltype(absl::container_internal::DecomposeValue(
      std::declval<F>(), std::declval<Args>()...))
  apply(F&& f, Args&&... args) {
    return absl::container_internal::DecomposeValue(
        std::forward<F>(f), std::forward<Args>(args)...);
  }

  static size_t space_used(const T*) { return 0; }
};
}  // namespace container_internal

namespace container_algorithm_internal {

// Specialization of trait in absl/algorithm/container.h
template <class Key, class Hash, class KeyEqual, class Allocator>
struct IsUnorderedContainer<absl::flat_hash_set<Key, Hash, KeyEqual, Allocator>>
    : std::true_type {};

}  // namespace container_algorithm_internal

ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_CONTAINER_FLAT_HASH_SET_H_
