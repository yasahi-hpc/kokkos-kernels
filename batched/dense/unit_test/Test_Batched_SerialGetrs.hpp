//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Getrf.hpp>
#include <KokkosBatched_Getrs.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include "Test_Batched_DenseUtils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace Getrf {

template <typename T>
struct ParamTag {
  using trans = T;
};

template <typename DeviceType, typename AViewType, typename PivViewType, typename AlgoTagType>
struct Functor_BatchedSerialGetrf {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  PivViewType _ipiv;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGetrf(const AViewType &a, const PivViewType &ipiv) : _a(a), _ipiv(ipiv) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto aa   = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto ipiv = Kokkos::subview(_ipiv, k, Kokkos::ALL());

    KokkosBatched::SerialGetrf<AlgoTagType>::invoke(aa, ipiv);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGetrs");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::RangePolicy<execution_space> policy(0, _a.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename DeviceType, typename AViewType, typename PivViewType,
          typename BViewType, typename ParamTagType, typename AlgoTagType>
struct Functor_BatchedSerialGetrs {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  BViewType _b;
  PivViewType _ipiv;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGetrs(const AViewType &a, const PivViewType &ipiv,
                             const BViewType &b)
      : _a(a), _b(b), _ipiv(ipiv) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ParamTagType &, const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto ipiv = Kokkos::subview(_ipiv, k, Kokkos::ALL());
    auto bb   = Kokkos::subview(_b, k, Kokkos::ALL());

    KokkosBatched::SerialGetrs<typename ParamTagType::trans,
                               AlgoTagType>::invoke(aa, ipiv, bb);
  }

  inline int run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGetrs");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space, ParamTagType> policy(0, _b.extent(0));
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);
    Kokkos::Profiling::popRegion();
    return info_sum;
  }
};

template <typename DeviceType, typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
struct Functor_BatchedSerialGemm {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  BViewType _b;
  CViewType _c;
  ScalarType _alpha, _beta;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGemm(const ScalarType alpha, const AViewType &a, const BViewType &b, const ScalarType beta,
                            const CViewType &c)
      : _a(a), _b(b), _c(c), _alpha(alpha), _beta(beta) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
    auto cc = Kokkos::subview(_c, k, Kokkos::ALL(), Kokkos::ALL());

    KokkosBatched::SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked>::invoke(_alpha, aa, bb,
                                                                                                     _beta, cc);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGetrf");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::RangePolicy<execution_space> policy(0, _a.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename DeviceType, typename ScalarType, typename LayoutType,
          typename ParamTagType, typename AlgoTagType>
/// \brief Implementation details of batched getrs test
/// Confirm A * x = b, where
///        A: [[1, 1],
///            [1, -1]]
///        b: [2, 0]
///        x: [1, 1]
/// This corresponds to the following system of equations:
///        x0 + x1 = 2
///        x0 - x1 = 0
///
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_getrs_analytical(const int N) {
  using ats            = typename Kokkos::ArithTraits<ScalarType>;
  using RealType       = typename ats::mag_type;
  using View2DType    = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  constexpr int BlkSize = 2;
  View3DType A("A", N, BlkSize, BlkSize), ref("Ref", N, BlkSize, BlkSize);
  View3DType lu("lu", N, BlkSize, BlkSize);  // Factorized
  View2DType x("x", N, BlkSize), y("y", N, BlkSize),
      x_ref("x_ref", N, BlkSize);  // Solutions
  PivView2DType ipiv("ipiv", N, BlkSize);

  auto h_A     = Kokkos::create_mirror_view(A);
  auto h_x     = Kokkos::create_mirror_view(x);
  auto h_x_ref = Kokkos::create_mirror_view(x_ref);
  Kokkos::deep_copy(h_A, 1.0);
  for (std::size_t ib = 0; ib < N; ib++) {
    h_A(ib, 1, 1) = -1.0;

    h_x(ib, 0)     = 2;
    h_x(ib, 1)     = 0;
    h_x_ref(ib, 0) = 1;
    h_x_ref(ib, 1) = 1;
  }

  Kokkos::fence();

  Kokkos::deep_copy(A, h_A);
  Kokkos::deep_copy(x, h_x);

  // getrf to factorize matrix A = P * L * U
  Functor_BatchedSerialGetrf<DeviceType, View3DType, PivView2DType,
                             AlgoTagType>(A, ipiv)
      .run();

  // getrs (Note, LU is a factorized matrix of A)
  auto info = Functor_BatchedSerialGetrs<DeviceType, View3DType, PivView2DType, View2DType,
                             ParamTagType, AlgoTagType>(A, ipiv, x)
      .run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();

  // Check if x = [1, 1]
  Kokkos::deep_copy(h_x, x);
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      EXPECT_NEAR_KK(h_x(ib, i), h_x_ref(ib, i), eps);
    }
  }
}

template <typename DeviceType, typename ScalarType, typename LayoutType,
          typename ParamTagType, typename AlgoTagType>
/// \brief Implementation details of batched getrs test
///
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_getrs(const int N, const int BlkSize) {
  using ats = typename Kokkos::ArithTraits<ScalarType>;
  using RealType      = typename ats::mag_type;
  using View2DType    = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  View3DType A("A", N, BlkSize, BlkSize), ref("Ref", N, BlkSize, BlkSize);
  View3DType LU("LU", N, BlkSize, BlkSize);  // Factorized
  View2DType x("x", N, BlkSize), y("y", N, BlkSize),
      b("b", N, BlkSize);  // Solutions
  PivView2DType ipiv("ipiv", N, BlkSize);

  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  // Initialize A_reconst with random matrix
  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(A, rand_pool, randStart, randEnd);
  Kokkos::fill_random(x, rand_pool, randStart, randEnd);
  Kokkos::deep_copy(LU, A);
  Kokkos::deep_copy(b, x);

  // getrf to factorize matrix A = P * L * U
  Functor_BatchedSerialGetrf<DeviceType, View3DType, PivView2DType,
                             AlgoTagType>(LU, ipiv)
      .run();

  // getrs (Note, LU is a factorized matrix of A)
  auto info = Functor_BatchedSerialGetrs<DeviceType, View3DType, PivView2DType, View2DType,
                             ParamTagType, AlgoTagType>(LU, ipiv, x)
      .run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // Gemv to compute A*x, this should be identical to b
  Functor_BatchedSerialGemv<DeviceType, ScalarType, View3DType, View2DType,
                            View2DType, ParamTagType>(1.0, A, x, 0.0, y)
      .run();

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();

  // Check if A * x = b
  auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      EXPECT_NEAR_KK(h_y(ib, i), h_b(ib, i), eps);
    }
  }
}

}  // namespace Getrs
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename ParamTagType,  typename AlgoTagType>
int test_batched_getrs() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Getrs::impl_test_batched_getrs_analytical<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1);
    Test::Getrs::impl_test_batched_getrs_analytical<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(2);
    for (int i = 0; i < 10; i++) {
      Test::Getrs::impl_test_batched_getrs<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1, i);
      Test::Getrs::impl_test_batched_getrs<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(2, i);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Getrf::impl_test_batched_getrs_analytical<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1);
    Test::Getrf::impl_test_batched_getrs_analytical<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(2);
    for (int i = 0; i < 10; i++) {
      Test::Getrf::impl_test_batched_getrs<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1, i);
      Test::Getrf::impl_test_batched_getrs<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(2, i);
    }
  }
#endif

  return 0;
}
