// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_EVOLUTIONV1_EVOLUTION_V1_H_
#define FLATBUFFERS_GENERATED_EVOLUTIONV1_EVOLUTION_V1_H_

#include "flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 23 &&
              FLATBUFFERS_VERSION_MINOR == 5 &&
              FLATBUFFERS_VERSION_REVISION == 26,
             "Non-compatible flatbuffers version included");

namespace Evolution {
namespace V1 {

struct TableA;
struct TableABuilder;

struct TableB;
struct TableBBuilder;

struct Struct;

struct Root;
struct RootBuilder;

enum class Enum : int8_t {
  King = 0,
  Queen = 1,
  MIN = King,
  MAX = Queen
};

inline const Enum (&EnumValuesEnum())[2] {
  static const Enum values[] = {
    Enum::King,
    Enum::Queen
  };
  return values;
}

inline const char * const *EnumNamesEnum() {
  static const char * const names[3] = {
    "King",
    "Queen",
    nullptr
  };
  return names;
}

inline const char *EnumNameEnum(Enum e) {
  if (::flatbuffers::IsOutRange(e, Enum::King, Enum::Queen)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesEnum()[index];
}

enum class Union : uint8_t {
  NONE = 0,
  TableA = 1,
  TableB = 2,
  MIN = NONE,
  MAX = TableB
};

inline const Union (&EnumValuesUnion())[3] {
  static const Union values[] = {
    Union::NONE,
    Union::TableA,
    Union::TableB
  };
  return values;
}

inline const char * const *EnumNamesUnion() {
  static const char * const names[4] = {
    "NONE",
    "TableA",
    "TableB",
    nullptr
  };
  return names;
}

inline const char *EnumNameUnion(Union e) {
  if (::flatbuffers::IsOutRange(e, Union::NONE, Union::TableB)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesUnion()[index];
}

template<typename T> struct UnionTraits {
  static const Union enum_value = Union::NONE;
};

template<> struct UnionTraits<Evolution::V1::TableA> {
  static const Union enum_value = Union::TableA;
};

template<> struct UnionTraits<Evolution::V1::TableB> {
  static const Union enum_value = Union::TableB;
};

bool VerifyUnion(::flatbuffers::Verifier &verifier, const void *obj, Union type);
bool VerifyUnionVector(::flatbuffers::Verifier &verifier, const ::flatbuffers::Vector<::flatbuffers::Offset<void>> *values, const ::flatbuffers::Vector<Union> *types);

FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(8) Struct FLATBUFFERS_FINAL_CLASS {
 private:
  int32_t a_;
  int32_t padding0__;
  double b_;

 public:
  Struct()
      : a_(0),
        padding0__(0),
        b_(0) {
    (void)padding0__;
  }
  Struct(int32_t _a, double _b)
      : a_(::flatbuffers::EndianScalar(_a)),
        padding0__(0),
        b_(::flatbuffers::EndianScalar(_b)) {
    (void)padding0__;
  }
  int32_t a() const {
    return ::flatbuffers::EndianScalar(a_);
  }
  double b() const {
    return ::flatbuffers::EndianScalar(b_);
  }
};
FLATBUFFERS_STRUCT_END(Struct, 16);

inline bool operator==(const Struct &lhs, const Struct &rhs) {
  return
      (lhs.a() == rhs.a()) &&
      (lhs.b() == rhs.b());
}

inline bool operator!=(const Struct &lhs, const Struct &rhs) {
    return !(lhs == rhs);
}


struct TableA FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef TableABuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_A = 4,
    VT_B = 6
  };
  float a() const {
    return GetField<float>(VT_A, 0.0f);
  }
  int32_t b() const {
    return GetField<int32_t>(VT_B, 0);
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<float>(verifier, VT_A, 4) &&
           VerifyField<int32_t>(verifier, VT_B, 4) &&
           verifier.EndTable();
  }
};

struct TableABuilder {
  typedef TableA Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_a(float a) {
    fbb_.AddElement<float>(TableA::VT_A, a, 0.0f);
  }
  void add_b(int32_t b) {
    fbb_.AddElement<int32_t>(TableA::VT_B, b, 0);
  }
  explicit TableABuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<TableA> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<TableA>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<TableA> CreateTableA(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    float a = 0.0f,
    int32_t b = 0) {
  TableABuilder builder_(_fbb);
  builder_.add_b(b);
  builder_.add_a(a);
  return builder_.Finish();
}

struct TableB FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef TableBBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_A = 4
  };
  int32_t a() const {
    return GetField<int32_t>(VT_A, 0);
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_A, 4) &&
           verifier.EndTable();
  }
};

struct TableBBuilder {
  typedef TableB Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_a(int32_t a) {
    fbb_.AddElement<int32_t>(TableB::VT_A, a, 0);
  }
  explicit TableBBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<TableB> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<TableB>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<TableB> CreateTableB(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    int32_t a = 0) {
  TableBBuilder builder_(_fbb);
  builder_.add_a(a);
  return builder_.Finish();
}

struct Root FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef RootBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_A = 4,
    VT_B = 6,
    VT_C_TYPE = 8,
    VT_C = 10,
    VT_D = 12,
    VT_E = 14,
    VT_F = 16,
    VT_G = 18,
    VT_H = 20,
    VT_I = 22,
    VT_J_TYPE = 24,
    VT_J = 26
  };
  int32_t a() const {
    return GetField<int32_t>(VT_A, 0);
  }
  bool b() const {
    return GetField<uint8_t>(VT_B, 0) != 0;
  }
  Evolution::V1::Union c_type() const {
    return static_cast<Evolution::V1::Union>(GetField<uint8_t>(VT_C_TYPE, 0));
  }
  const void *c() const {
    return GetPointer<const void *>(VT_C);
  }
  template<typename T> const T *c_as() const;
  const Evolution::V1::TableA *c_as_TableA() const {
    return c_type() == Evolution::V1::Union::TableA ? static_cast<const Evolution::V1::TableA *>(c()) : nullptr;
  }
  const Evolution::V1::TableB *c_as_TableB() const {
    return c_type() == Evolution::V1::Union::TableB ? static_cast<const Evolution::V1::TableB *>(c()) : nullptr;
  }
  Evolution::V1::Enum d() const {
    return static_cast<Evolution::V1::Enum>(GetField<int8_t>(VT_D, 0));
  }
  const Evolution::V1::TableA *e() const {
    return GetPointer<const Evolution::V1::TableA *>(VT_E);
  }
  const Evolution::V1::Struct *f() const {
    return GetStruct<const Evolution::V1::Struct *>(VT_F);
  }
  const ::flatbuffers::Vector<int32_t> *g() const {
    return GetPointer<const ::flatbuffers::Vector<int32_t> *>(VT_G);
  }
  const ::flatbuffers::Vector<::flatbuffers::Offset<Evolution::V1::TableB>> *h() const {
    return GetPointer<const ::flatbuffers::Vector<::flatbuffers::Offset<Evolution::V1::TableB>> *>(VT_H);
  }
  int32_t i() const {
    return GetField<int32_t>(VT_I, 1234);
  }
  Evolution::V1::Union j_type() const {
    return static_cast<Evolution::V1::Union>(GetField<uint8_t>(VT_J_TYPE, 0));
  }
  const void *j() const {
    return GetPointer<const void *>(VT_J);
  }
  template<typename T> const T *j_as() const;
  const Evolution::V1::TableA *j_as_TableA() const {
    return j_type() == Evolution::V1::Union::TableA ? static_cast<const Evolution::V1::TableA *>(j()) : nullptr;
  }
  const Evolution::V1::TableB *j_as_TableB() const {
    return j_type() == Evolution::V1::Union::TableB ? static_cast<const Evolution::V1::TableB *>(j()) : nullptr;
  }
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_A, 4) &&
           VerifyField<uint8_t>(verifier, VT_B, 1) &&
           VerifyField<uint8_t>(verifier, VT_C_TYPE, 1) &&
           VerifyOffset(verifier, VT_C) &&
           VerifyUnion(verifier, c(), c_type()) &&
           VerifyField<int8_t>(verifier, VT_D, 1) &&
           VerifyOffset(verifier, VT_E) &&
           verifier.VerifyTable(e()) &&
           VerifyField<Evolution::V1::Struct>(verifier, VT_F, 8) &&
           VerifyOffset(verifier, VT_G) &&
           verifier.VerifyVector(g()) &&
           VerifyOffset(verifier, VT_H) &&
           verifier.VerifyVector(h()) &&
           verifier.VerifyVectorOfTables(h()) &&
           VerifyField<int32_t>(verifier, VT_I, 4) &&
           VerifyField<uint8_t>(verifier, VT_J_TYPE, 1) &&
           VerifyOffset(verifier, VT_J) &&
           VerifyUnion(verifier, j(), j_type()) &&
           verifier.EndTable();
  }
};

template<> inline const Evolution::V1::TableA *Root::c_as<Evolution::V1::TableA>() const {
  return c_as_TableA();
}

template<> inline const Evolution::V1::TableB *Root::c_as<Evolution::V1::TableB>() const {
  return c_as_TableB();
}

template<> inline const Evolution::V1::TableA *Root::j_as<Evolution::V1::TableA>() const {
  return j_as_TableA();
}

template<> inline const Evolution::V1::TableB *Root::j_as<Evolution::V1::TableB>() const {
  return j_as_TableB();
}

struct RootBuilder {
  typedef Root Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  void add_a(int32_t a) {
    fbb_.AddElement<int32_t>(Root::VT_A, a, 0);
  }
  void add_b(bool b) {
    fbb_.AddElement<uint8_t>(Root::VT_B, static_cast<uint8_t>(b), 0);
  }
  void add_c_type(Evolution::V1::Union c_type) {
    fbb_.AddElement<uint8_t>(Root::VT_C_TYPE, static_cast<uint8_t>(c_type), 0);
  }
  void add_c(::flatbuffers::Offset<void> c) {
    fbb_.AddOffset(Root::VT_C, c);
  }
  void add_d(Evolution::V1::Enum d) {
    fbb_.AddElement<int8_t>(Root::VT_D, static_cast<int8_t>(d), 0);
  }
  void add_e(::flatbuffers::Offset<Evolution::V1::TableA> e) {
    fbb_.AddOffset(Root::VT_E, e);
  }
  void add_f(const Evolution::V1::Struct *f) {
    fbb_.AddStruct(Root::VT_F, f);
  }
  void add_g(::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> g) {
    fbb_.AddOffset(Root::VT_G, g);
  }
  void add_h(::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<Evolution::V1::TableB>>> h) {
    fbb_.AddOffset(Root::VT_H, h);
  }
  void add_i(int32_t i) {
    fbb_.AddElement<int32_t>(Root::VT_I, i, 1234);
  }
  void add_j_type(Evolution::V1::Union j_type) {
    fbb_.AddElement<uint8_t>(Root::VT_J_TYPE, static_cast<uint8_t>(j_type), 0);
  }
  void add_j(::flatbuffers::Offset<void> j) {
    fbb_.AddOffset(Root::VT_J, j);
  }
  explicit RootBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<Root> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Root>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<Root> CreateRoot(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    int32_t a = 0,
    bool b = false,
    Evolution::V1::Union c_type = Evolution::V1::Union::NONE,
    ::flatbuffers::Offset<void> c = 0,
    Evolution::V1::Enum d = Evolution::V1::Enum::King,
    ::flatbuffers::Offset<Evolution::V1::TableA> e = 0,
    const Evolution::V1::Struct *f = nullptr,
    ::flatbuffers::Offset<::flatbuffers::Vector<int32_t>> g = 0,
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<Evolution::V1::TableB>>> h = 0,
    int32_t i = 1234,
    Evolution::V1::Union j_type = Evolution::V1::Union::NONE,
    ::flatbuffers::Offset<void> j = 0) {
  RootBuilder builder_(_fbb);
  builder_.add_j(j);
  builder_.add_i(i);
  builder_.add_h(h);
  builder_.add_g(g);
  builder_.add_f(f);
  builder_.add_e(e);
  builder_.add_c(c);
  builder_.add_a(a);
  builder_.add_j_type(j_type);
  builder_.add_d(d);
  builder_.add_c_type(c_type);
  builder_.add_b(b);
  return builder_.Finish();
}

inline ::flatbuffers::Offset<Root> CreateRootDirect(
    ::flatbuffers::FlatBufferBuilder &_fbb,
    int32_t a = 0,
    bool b = false,
    Evolution::V1::Union c_type = Evolution::V1::Union::NONE,
    ::flatbuffers::Offset<void> c = 0,
    Evolution::V1::Enum d = Evolution::V1::Enum::King,
    ::flatbuffers::Offset<Evolution::V1::TableA> e = 0,
    const Evolution::V1::Struct *f = nullptr,
    const std::vector<int32_t> *g = nullptr,
    const std::vector<::flatbuffers::Offset<Evolution::V1::TableB>> *h = nullptr,
    int32_t i = 1234,
    Evolution::V1::Union j_type = Evolution::V1::Union::NONE,
    ::flatbuffers::Offset<void> j = 0) {
  auto g__ = g ? _fbb.CreateVector<int32_t>(*g) : 0;
  auto h__ = h ? _fbb.CreateVector<::flatbuffers::Offset<Evolution::V1::TableB>>(*h) : 0;
  return Evolution::V1::CreateRoot(
      _fbb,
      a,
      b,
      c_type,
      c,
      d,
      e,
      f,
      g__,
      h__,
      i,
      j_type,
      j);
}

inline bool VerifyUnion(::flatbuffers::Verifier &verifier, const void *obj, Union type) {
  switch (type) {
    case Union::NONE: {
      return true;
    }
    case Union::TableA: {
      auto ptr = reinterpret_cast<const Evolution::V1::TableA *>(obj);
      return verifier.VerifyTable(ptr);
    }
    case Union::TableB: {
      auto ptr = reinterpret_cast<const Evolution::V1::TableB *>(obj);
      return verifier.VerifyTable(ptr);
    }
    default: return true;
  }
}

inline bool VerifyUnionVector(::flatbuffers::Verifier &verifier, const ::flatbuffers::Vector<::flatbuffers::Offset<void>> *values, const ::flatbuffers::Vector<Union> *types) {
  if (!values || !types) return !values && !types;
  if (values->size() != types->size()) return false;
  for (::flatbuffers::uoffset_t i = 0; i < values->size(); ++i) {
    if (!VerifyUnion(
        verifier,  values->Get(i), types->GetEnum<Union>(i))) {
      return false;
    }
  }
  return true;
}

inline const Evolution::V1::Root *GetRoot(const void *buf) {
  return ::flatbuffers::GetRoot<Evolution::V1::Root>(buf);
}

inline const Evolution::V1::Root *GetSizePrefixedRoot(const void *buf) {
  return ::flatbuffers::GetSizePrefixedRoot<Evolution::V1::Root>(buf);
}

inline bool VerifyRootBuffer(
    ::flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<Evolution::V1::Root>(nullptr);
}

inline bool VerifySizePrefixedRootBuffer(
    ::flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<Evolution::V1::Root>(nullptr);
}

inline void FinishRootBuffer(
    ::flatbuffers::FlatBufferBuilder &fbb,
    ::flatbuffers::Offset<Evolution::V1::Root> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedRootBuffer(
    ::flatbuffers::FlatBufferBuilder &fbb,
    ::flatbuffers::Offset<Evolution::V1::Root> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace V1
}  // namespace Evolution

#endif  // FLATBUFFERS_GENERATED_EVOLUTIONV1_EVOLUTION_V1_H_
