/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_LAYOUT_H_
#define XLA_PYTHON_IFRT_LAYOUT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.pb.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

class Client;
class Device;
class Sharding;
class Layout;
class ConcreteLayout;

// General layout reference that can be either a default or custom layout.
using LayoutRef = absl_nonnull std::shared_ptr<Layout>;

// Concrete layout reference.
using ConcreteLayoutRef = absl_nonnull std::shared_ptr<ConcreteLayout>;

// Optional custom layout reference. `nullptr` indicates a default layout, and
// non-`nullptr` indicates a custom layout.
using CustomLayoutRef = absl_nullable std::shared_ptr<ConcreteLayout>;

// Abstract layout type.
//
// `Layout` describes how the elements of a single shard in an array are
// arranged in memory. `Layout` may contain transpose, padding, packing,
// sparsity, indirection, and so forth. All shards of a single array use the
// same layout.
//
// Note that within-element layouts such as big/little endian are not expressed
// through `Layout`. They may be expressed through `DType`.
class Layout : public llvm::RTTIExtends<Layout, Serializable> {
 public:
  Layout(const Layout& other) = delete;
  Layout& operator=(const Layout& other) = delete;
  Layout(Layout&& other) = delete;
  Layout& operator=(Layout&& other) = delete;

  // Returns true if this layout represents a default layout. Note that a
  // concrete (resolved) layout is not considered a default layout.
  bool is_default() const;

  // Returns itself if it is already a concrete layout. Otherwise, resolves to a
  // concrete layout and returns it. Both default and custom layouts will return
  // a concrete layout.
  virtual absl::StatusOr<ConcreteLayoutRef> GetConcrete() const = 0;

  // Returns itself if it is a custom layout. Otherwise, returns a `nullptr`.
  // Used for APIs that takes `CustomLayoutRef`.
  virtual CustomLayoutRef GetIfCustom() const = 0;

  // Returns if this layout is equal to `other`.
  //
  // Unless compared layouts are `ConcreteLayout`s, comparisons are done without
  // comparing concrete layout information.
  virtual bool operator==(const Layout& other) const = 0;
  bool operator!=(const Layout& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const Layout& layout) {
    return H::combine(std::move(h), layout.hash());
  }

  // Constructs `Layout` from `LayoutProto`.
  static absl::StatusOr<absl_nonnull std::unique_ptr<Layout>> FromProto(
      Client* client, const LayoutProto& proto);

  // Returns a `LayoutProto` representation.
  absl::StatusOr<LayoutProto> ToProto() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Layout& layout) {
    sink.Append(layout.ToString());
  }

  static char ID;  // NOLINT

 protected:
  Layout() = default;

 private:
  // Returns the hash of the layout. This hash is stable only within the
  // process.
  virtual uint64_t hash() const = 0;

  // Returns a string representation of the layout.
  virtual std::string ToString() const = 0;
};

// Abstract layout type that contains the concrete (resolved) layout
// information. A concrete layout contains the complete information to describe
// how the elements of a shard are arranged in memory.
//
// The layout may represent either default layout or custom layout.
class ConcreteLayout : public llvm::RTTIExtends<ConcreteLayout, Layout>,
                       public std::enable_shared_from_this<ConcreteLayout> {
 public:
  ~ConcreteLayout() override = default;

  // Returns a tuple of (byte offset of the first element, byte offset right
  // after the last element), relative to the element at [0, 0, ...].
  virtual absl::StatusOr<std::pair<int64_t, int64_t>> ByteBound(
      DType dtype, const Shape& shard_shape) const = 0;

  // Computes the byte size of a shard shape using the layout. It is computed as
  // (upper bound - lower bound).
  absl::StatusOr<int64_t> ByteSize(DType dtype, const Shape& shard_shape) const;

  // Layout implementation.

  absl::StatusOr<ConcreteLayoutRef> GetConcrete() const override;

  CustomLayoutRef GetIfCustom() const override;

  static char ID;  // NOLINT
};

// Default layout that is not yet resolved to a concrete layout, but can
// be resolved when requested.
//
// Typically, `DefaultLayout` is created by IFRT implementations to return a
// `LayoutRef` value.
//
// In contrast, the user would use a `nullptr` value for a "custom_layout" that
// uses `CustomLayoutRef` to indicate a request to let the runtime pick the
// default layout or specify a custom layout to use. In other words, the user
// typically does not need to construct `DefaultLayout` unless they use
// `LayoutRef` in their own code.
class DefaultLayout final : public llvm::RTTIExtends<DefaultLayout, Layout> {
 public:
  // Creates a default layout.
  //
  // TODO(hyeontaek): Consider replacing `device` and `memory_kind` with
  // `client` and a new ID that identifies a homogeneous domain of default
  // layout calculation rules (defined by IFRT implementations), and also change
  // `Client::GetDefaultLayout()` to use the new ID. This will improve any cache
  // hit and reduce false positives in Layout equality checks.
  static absl_nonnull std::unique_ptr<DefaultLayout> Create(
      DType dtype, Shape shard_shape, Device* device, MemoryKind memory_kind);

  // Alternative `Create()` that computes the shard shape from `shape` and
  // `sharding`.
  static absl::StatusOr<absl_nonnull std::unique_ptr<DefaultLayout>> Create(
      DType dtype, const Shape& shape, const Sharding& sharding);

  ~DefaultLayout() override = default;

  DType dtype() const { return dtype_; }
  const Shape& shard_shape() const { return shard_shape_; }
  Device* device() const { return device_; }
  MemoryKind memory_kind() const { return memory_kind_; }

  // Layout implementation.

  absl::StatusOr<ConcreteLayoutRef> GetConcrete() const override;

  CustomLayoutRef GetIfCustom() const override;

  bool operator==(const Layout& other) const override;

  static char ID;  // NOLINT

 private:
  DefaultLayout(DType dtype, Shape shard_shape, Device* device,
                MemoryKind memory_kind)
      : dtype_(dtype),
        shard_shape_(std::move(shard_shape)),
        device_(device),
        memory_kind_(memory_kind) {}

  uint64_t hash() const override;
  std::string ToString() const override;

  DType dtype_;
  Shape shard_shape_;
  Device* device_;
  MemoryKind memory_kind_;

  // Tentative implementation; see `GetConcrete()` for details.
  // TODO(hyeontaek): Implement.
  // absl::Mutex mutex_;
  // std::shared_ptr<ConcreteLayout> cached_concrete_layout_
  //     ABSL_GUARDED_BY(mutex_);
};

// Concrete layout that uses byte strides. Conceptually, a byte stride indices
// how many bytes to skip forward or backward to find an element when
// incrementing a dimension in the shape.
//
// The layout may represent either default layout or custom layout.
//
// Semantics:
//
// The byte offset of an element at an index from the element at [0, 0, ...] is
// calculated as
//
//   sum_i { element_index[i] * byte_strides[i] }
//
// The resulting byte offset may be negative if some byte_strides[i] is
// negative.
//
// Caution:
//
// * By being a concrete layout, a `ByteStridesLayout` is typically meaningful
// only for a specific `DType` and `Shape`. At minimum, it can only be used with
// a `Shape` whose dimension count matches.
//
// * Size-less dtypes such as `DType::kToken use 0 bytes as element size.
// For the time being, such a dtype is expected to have an empty shape
// dimension.
//
// * Sub-byte `DType`s such as `DType::kS4` uses 1 byte as element size.
//
// * Byte strides are used as-is. There is no check on whether all element
// access is aligned to a element size boundary or each element has a unique
// byte offset.
class ByteStridesLayout final
    : public llvm::RTTIExtends<ByteStridesLayout, ConcreteLayout> {
 public:
  using ByteStrides = absl::InlinedVector<int64_t, 6>;

  // Creates a byte strides layout.
  static absl_nonnull std::unique_ptr<ByteStridesLayout> Create(
      ByteStrides byte_strides);

  // Create a byte strides layout that represents a compact major-to-minor
  // layout (C layout) for `dtype` and `shard_shape`. `dtype.byte_size()` must
  // be defined.
  //
  // The byte strides is determined from
  //
  //   byte_strides[i] = dtype.byte_size() * prod_{i < j < N} { shard_shape[j] }
  //
  static absl::StatusOr<absl_nonnull std::unique_ptr<ByteStridesLayout>>
  CreateCompactMajorToMinor(DType dtype, const Shape& shard_shape);

  ~ByteStridesLayout() override = default;

  absl::Span<const int64_t> byte_strides() const { return byte_strides_; }

  // ConcreteLayout implementation.

  absl::StatusOr<std::pair<int64_t, int64_t>> ByteBound(
      DType dtype, const Shape& shard_shape) const override;

  // Layout implementation.

  bool operator==(const Layout& other) const override;

  static char ID;  // NOLINT

 private:
  explicit ByteStridesLayout(ByteStrides byte_strides)
      : byte_strides_(std::move(byte_strides)) {}

  uint64_t hash() const override;
  std::string ToString() const override;

  ByteStrides byte_strides_;
};

// Options for deserializing layouts. `client` must remain valid during
// deserialization.
struct DeserializeLayoutOptions
    : llvm::RTTIExtends<DeserializeLayoutOptions, DeserializeOptions> {
  explicit DeserializeLayoutOptions(Client* client) : client(client) {}

  static char ID;  // NOLINT

  Client* client;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_LAYOUT_H_
