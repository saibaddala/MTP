/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Utilities for dealing with Literal protobufs.

#ifndef TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_

#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "function_ref.h"
#include "string_view.h"
#include "span.h"
#include "array2d.h"
#include "array3d.h"
#include "array4d.h"
#include "index_util.h"
#include "layout_util.h"
#include "literal.h"
#include "primitive_util.h"
#include "shape_util.h"
#include "status_macros.h"
#include "types.h"
#include "util.h"
#include "xla_data.pb.h"
#include "bitmap.h"
#include "logging.h"
#include "protobuf.h"
#include "status.h"

namespace xla {

class LiteralUtil {
 public:
  LiteralUtil() = delete;

  // Returns a literal scalar representing the first element.
  static Literal GetFirstScalarLiteral(const LiteralSlice& literal);
  // Returns a literal scalar representing the element at `multi_index`.
  static Literal GetScalarLiteral(const LiteralBase& literal,
                                  absl::Span<const int64_t> multi_index);
  // Sets the value of the element at `multi_index` with a scalar literal.
  static void SetScalarLiteral(MutableLiteralBase& literal,
                               absl::Span<const int64_t> multi_index,
                               const LiteralBase& scalar);

  // Creates a new literal of a given rank. To minimize ambiguity (for users
  // and the compiler) these CreateR[0-2] methods should explicitly specify the
  // native type. For example:
  //
  //  CreateR1<float>({1.0, 42.0});
  //  CreateR2<uint32_t>({{1, 2}, {3, 4}});
  //
  // The variants not ending with WithLayout use the default XLA layout for the
  // literal's linear representation in memory.
  template <typename NativeT>
  static Literal CreateR0(NativeT value);
  template <typename NativeT>
  static Literal CreateR1(absl::Span<const NativeT> values);
  static Literal CreateR1(const tsl::core::Bitmap& values);
  template <typename NativeT>
  static Literal CreateR2(
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  static Literal CreateR2WithLayout(
      std::initializer_list<std::initializer_list<NativeT>> values,
      const Layout& layout);
  template <typename NativeT>
  static Literal CreateR3(std::initializer_list<
                          std::initializer_list<std::initializer_list<NativeT>>>
                              values);
  template <typename NativeT>
  static Literal CreateR3WithLayout(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values,
      const Layout& layout);
  template <typename NativeT>
  static Literal CreateR4(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values);
  template <typename NativeT>
  static Literal CreateR4WithLayout(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values,
      const Layout& layout);

  // Creates a scalar literal value zero of the given primitive type.
  static Literal Zero(PrimitiveType primitive_type);
  // Creates a scalar literal value one of the given primitive type.
  static Literal One(PrimitiveType primitive_type);
  // Creates a scalar literal value containing the minimum value of the given
  // primitive type. For floating-point types supporting inf, returns -inf.
  static Literal MinValue(PrimitiveType primitive_type);
  // Creates a scalar literal value containing the maximum value of the given
  // primitive type. For floating-point types supporting inf, returns inf.
  static Literal MaxValue(PrimitiveType primitive_type);
  // Creates a scalar literal value containing the NaN value of the given
  // primitive type. Fail for non-inexact types. For complex types, returns a
  // nan + nan * j value.
  static StatusOr<Literal> NanValue(PrimitiveType primitive_type);
  // Creates a literal of the given shape where each element is `value`.
  template <typename NativeT>
  static Literal CreateFullWithDescendingLayout(
      absl::Span<const int64_t> dimensions, NativeT value);

  // Creates a new literal from an Array type. The variants not ending with
  // WithLayout use the default XLA layout for the literal's linear
  // representation in memory.
  template <typename NativeT>
  static Literal CreateFromArray(const Array<NativeT>& values);
  template <typename NativeT>
  static Literal CreateFromArrayWithLayout(const Array<NativeT>& values,
                                           const Layout& layout);
  template <typename NativeT>
  static Literal CreateR2FromArray2D(const Array2D<NativeT>& values);
  template <typename NativeT>
  static Literal CreateR2FromArray2DWithLayout(const Array2D<NativeT>& values,
                                               const Layout& layout);
  template <typename NativeT>
  static Literal CreateR3FromArray3D(const Array3D<NativeT>& values);
  template <typename NativeT>
  static Literal CreateR3FromArray3DWithLayout(const Array3D<NativeT>& values,
                                               const Layout& layout);
  template <typename NativeT>
  static Literal CreateR4FromArray4D(const Array4D<NativeT>& values);
  template <typename NativeT>
  static Literal CreateR4FromArray4DWithLayout(const Array4D<NativeT>& values,
                                               const Layout& layout);

  // Creates a new vector of U8s literal value from a string.
  static Literal CreateR1U8(absl::string_view value);

  // Creates a linspace-populated literal with the given number of rows and
  // columns.
  static Literal CreateR2F32Linspace(float from, float to, int64_t rows,
                                     int64_t cols);

  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z dimension given by "projection".
  template <typename NativeT>
  static Literal CreateR3Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64_t projection);

  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z and p dimensions given.
  template <typename NativeT>
  static Literal CreateR4Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64_t projection_p, int64_t projection_z);

  // Returns an identity matrix (rank 2) with the given row and column count.
  template <typename NativeT>
  static Literal MakeIdentityR2(int64_t size);

  // Returns a tuple literal composed of given literals. Data is copied from the
  // given elements into the returned literal.
  static Literal MakeTuple(absl::Span<const Literal* const> elements);

  static Literal MakeTupleFromSlices(absl::Span<const LiteralSlice> elements);

  // As above, but intended to be invoked with move semantics; i.e.
  //
  //  std::vector<Literal> elements = ...;
  //  auto result = LiteralUtil::MakeTupleOwned(std::move(elements));
  //
  // This would have been declared as an overload, but there is ambiguity
  // in invocation between the above signature and this one.
  static Literal MakeTupleOwned(std::vector<Literal> elements);

  // This overload lets you pass a list of Literals to MakeTupleOwned:
  //
  //   LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR1(...), ...).
  //
  // Simply relying on the MakeTupleOwned(std::vector<Literal>)
  // overload doesn't work because std::initializer_list's elements are always
  // const.
  //
  // The arguments to this function must all be Literal.
  template <typename... Ts>
  static Literal MakeTupleOwned(Ts... elements) {
    std::array<Literal, sizeof...(Ts)> arr{std::move(elements)...};
    std::vector<Literal> v;
    v.insert(v.begin(), std::make_move_iterator(arr.begin()),
             std::make_move_iterator(arr.end()));
    return MakeTupleOwned(std::move(v));
  }

  // Create a constant token literal. Token types have no value.
  static Literal CreateToken();

  // Creates a new Literal object with its values havings the primitive_type
  // type, and with dimensions defined by the dimensions parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static Literal CreateFromDimensions(PrimitiveType primitive_type,
                                      absl::Span<const int64_t> dimensions);

  // Convert<SrcType>To<DstType> family of functions:
  // If the given literal's data type is <SrcType>, converts it to a <DstType>
  // literal; otherwise, returns a copy of it. If the literal is a tuple,
  // recursively converts its elements.
  static Literal ConvertBF16ToF32(const LiteralSlice& bf16_literal);
  static Literal ConvertBF16ToF64(const LiteralSlice& bf16_literal);
  static Literal ConvertF32ToF8E4M3FNUZ(const LiteralSlice& f32_literal);
  static Literal ConvertF32ToF8E5M2FNUZ(const LiteralSlice& f32_literal);
  static Literal ConvertF32ToBF16(const LiteralSlice& f32_literal);
  static Literal ConvertF32ToS8(const LiteralSlice& f32_literal);
  static Literal ConvertF32ToF64(const LiteralSlice& f32_literal);
  static Literal ConvertF64ToBF16(const LiteralSlice& f64_literal);
  static Literal ConvertF64ToF32(const LiteralSlice& f64_literal);
  static Literal ConvertS32ToF32(const LiteralSlice& s32_literal);

  // Creates a scalar literal whose value is the maximum value of a given
  // literal slice.
  static Literal MaxElement(const LiteralSlice& literal);

  // Creates a literal with a new shape with the given new dimensions using the
  // data in the given input literal. For reshaping purposes the (flat) data
  // buffer of the input literal is assumed to have the given minor_to_major
  // layout order.
  static Literal ReshapeSlice(absl::Span<const int64_t> new_dimensions,
                              absl::Span<const int64_t> minor_to_major,
                              const LiteralSlice& literal);

  // Creates a literal with the supplied shape, and uses the provided value
  // generator to populate the literal's values.
  // Returns the new literal object, or an error Status if failed.
  template <PrimitiveType type, typename T = primitive_util::NativeTypeOf<type>>
  static StatusOr<Literal> CreateLiteralWithGenerator(
      const Shape& shape,
      absl::FunctionRef<T(absl::Span<const int64_t>)> generator);

  // Creates a literal with the supplied shape, and initializes the literal
  // values using a normal distribution with given mean and stddev standard
  // deviation, and using the engine as entropy generator.
  // Returns the new literal object, or an error Status if failed.
  template <PrimitiveType type, typename E,
            typename T = primitive_util::NativeTypeOf<type>>
  static StatusOr<Literal> CreateRandomLiteral(const Shape& shape, E* engine,
                                               T mean, T stddev);

  // Creates a literal with the supplied shape, and initializes the literal
  // values using a normal distribution with given mean and stddev standard
  // deviation.
  // Returns the new literal object, or an error Status if failed.
  template <PrimitiveType type, typename T = primitive_util::NativeTypeOf<type>>
  static StatusOr<Literal> CreateRandomLiteral(const Shape& shape, T mean,
                                               T stddev);

  //
  // End of factory methods.

  // Returns a multi-dimensional index as a string. For example: '{7, 8}' will
  // be returned for a 2-dimensional index with dimension 0 index equal to 7,
  // dimension 1 equal to 8.
  static std::string MultiIndexAsString(absl::Span<const int64_t> multi_index);
};

std::ostream& operator<<(std::ostream& out, const Literal& literal);

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR0(NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {}));
  literal.Set({}, value);
  return literal;
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR1(absl::Span<const NativeT> values) {
  Literal literal(
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<NativeT>(),
                           {static_cast<int64_t>(values.size())}));
  literal.PopulateR1(values);
  return literal;
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR2WithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    const Layout& layout) {
  Literal literal(ShapeUtil::MakeShapeWithDenseLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {static_cast<int64_t>(values.size()),
       static_cast<int64_t>(values.begin()->size())},
      layout.minor_to_major()));
  literal.PopulateR2(values);
  return literal;
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  return CreateR2WithLayout(values, LayoutUtil::GetDefaultLayoutForR2());
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR3WithLayout(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values,
    const Layout& layout) {
  const int64_t d0 = values.size();
  const int64_t d1 = values.begin()->size();
  const int64_t d2 = values.begin()->begin()->size();
  Array3D<NativeT> tmp(d0, d1, d2);
  int64_t i0 = 0;
  for (auto d1_values : values) {
    int64_t i1 = 0;
    for (auto d2_values : d1_values) {
      int64_t i2 = 0;
      for (auto value : d2_values) {
        tmp(i0, i1, i2) = value;
        ++i2;
      }
      ++i1;
    }
    ++i0;
  }
  return CreateR3FromArray3DWithLayout(tmp, layout);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR3(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values) {
  return CreateR3WithLayout(values, LayoutUtil::GetDefaultLayoutForR3());
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR4WithLayout(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values,
    const Layout& layout) {
  const int64_t d0 = values.size();
  const int64_t d1 = values.begin()->size();
  const int64_t d2 = values.begin()->begin()->size();
  const int64_t d3 = values.begin()->begin()->begin()->size();
  Array4D<NativeT> tmp(d0, d1, d2, d3);
  int64_t i0 = 0;
  for (auto d1_values : values) {
    int64_t i1 = 0;
    for (auto d2_values : d1_values) {
      int64_t i2 = 0;
      for (auto d3_values : d2_values) {
        int64_t i3 = 0;
        for (auto value : d3_values) {
          tmp(i0, i1, i2, i3) = value;
          ++i3;
        }
        ++i2;
      }
      ++i1;
    }
    ++i0;
  }
  return CreateR4FromArray4DWithLayout(tmp, layout);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR4(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values) {
  return CreateR4WithLayout(values, LayoutUtil::GetDefaultLayoutForR4());
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateFromArrayWithLayout(
    const Array<NativeT>& values, const Layout& layout) {
  Literal literal(ShapeUtil::MakeShapeWithDenseLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), values.dimensions(),
      layout.minor_to_major()));
  literal.PopulateFromArray(values);
  return literal;
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateFromArray(
    const Array<NativeT>& values) {
  return CreateFromArrayWithLayout(
      values, LayoutUtil::GetDefaultLayoutForRank(values.num_dimensions()));
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR2FromArray2DWithLayout(
    const Array2D<NativeT>& values, const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR2FromArray2D(
    const Array2D<NativeT>& values) {
  return CreateFromArray(values);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR3FromArray3DWithLayout(
    const Array3D<NativeT>& values, const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR3FromArray3D(
    const Array3D<NativeT>& values) {
  return CreateFromArray(values);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR3Projected(
    std::initializer_list<std::initializer_list<NativeT>> values,
    int64_t projection) {
  int64_t dim0_size = projection;
  int64_t dim1_size = values.size();
  int64_t dim2_size = values.begin()->size();

  Array3D<NativeT> array(dim0_size, dim1_size, dim2_size);
  for (int64_t dim0 = 0; dim0 < dim0_size; ++dim0) {
    int64_t dim1 = 0;
    for (auto inner_list : values) {
      int64_t dim2 = 0;
      for (auto value : inner_list) {
        array(dim0, dim1, dim2) = value;
        ++dim2;
      }
      CHECK_EQ(dim2_size, dim2);
      ++dim1;
    }
    CHECK_EQ(dim1_size, dim1);
  }
  return CreateR3FromArray3D(array);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR4Projected(
    std::initializer_list<std::initializer_list<NativeT>> values,
    int64_t projection_p, int64_t projection_z) {
  int64_t dim0_size = projection_p;
  int64_t dim1_size = projection_z;
  int64_t dim2_size = values.size();
  int64_t dim3_size = values.begin()->size();

  Array4D<NativeT> array(dim0_size, dim1_size, dim2_size, dim3_size);
  for (int64_t dim0 = 0; dim0 < dim0_size; ++dim0) {
    for (int64_t dim1 = 0; dim1 < dim1_size; ++dim1) {
      int64_t dim2 = 0;
      for (auto inner_list : values) {
        int64_t dim3 = 0;
        for (auto value : inner_list) {
          array(dim0, dim1, dim2, dim3) = value;
          ++dim3;
        }
        CHECK_EQ(dim3_size, dim3);
        ++dim2;
      }
      CHECK_EQ(dim2_size, dim2);
    }
  }
  return CreateR4FromArray4D(array);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR4FromArray4D(
    const Array4D<NativeT>& values) {
  return CreateFromArray(values);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateR4FromArray4DWithLayout(
    const Array4D<NativeT>& values, const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

// Returns an identity matrix (rank 2) with the given row and column count.
template <typename NativeT>
/* static */ Literal LiteralUtil::MakeIdentityR2(int64_t size) {
  Array2D<NativeT> array(size, size, 0);
  for (int64_t i = 0; i < size; ++i) {
    array(i, i) = 1;
  }
  return CreateR2FromArray2D(array);
}

template <typename NativeT>
/* static */ Literal LiteralUtil::CreateFullWithDescendingLayout(
    absl::Span<const int64_t> dimensions, NativeT value) {
  Literal literal(ShapeUtil::MakeShapeWithDescendingLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions));
  literal.PopulateWithValue(value);
  return literal;
}

template <PrimitiveType type, typename T>
/* static */ StatusOr<Literal> LiteralUtil::CreateLiteralWithGenerator(
    const Shape& shape,
    absl::FunctionRef<T(absl::Span<const int64_t>)> generator) {
  using NativeT = primitive_util::NativeTypeOf<type>;
  TF_RET_CHECK(shape.element_type() == type);
  Literal literal(shape);
  TF_RETURN_IF_ERROR(literal.Populate<NativeT>(
      [=](absl::Span<const int64_t> indexes) { return generator(indexes); }));
  return std::move(literal);
}

template <PrimitiveType type, typename E, typename T>
/* static */ StatusOr<Literal> LiteralUtil::CreateRandomLiteral(
    const Shape& shape, E* engine, T mean, T stddev) {
  using NativeT = primitive_util::NativeTypeOf<type>;
  std::normal_distribution<NativeT> generator(mean, stddev);
  return CreateLiteralWithGenerator<type, NativeT>(
      shape, [&](absl::Span<const int64_t> /*indexes*/) {
        return generator(*engine);
      });
}

template <PrimitiveType type, typename T>
/* static */ StatusOr<Literal> LiteralUtil::CreateRandomLiteral(
    const Shape& shape, T mean, T stddev) {
  std::minstd_rand0 engine;
  return CreateRandomLiteral<type>(shape, &engine, mean, stddev);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_
