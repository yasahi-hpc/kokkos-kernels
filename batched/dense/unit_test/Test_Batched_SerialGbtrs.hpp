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
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Gbtrf.hpp>
#include <KokkosBatched_Gbtrs.hpp>
#include "Test_Batched_DenseUtils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace Gbtrs {

template <typename T>
struct ParamTag {
  using trans = T;
};

template <typename DeviceType, typename ABViewType, typename PivViewType,
          typename AlgoTagType>
struct Functor_BatchedSerialGbtrf {
  using execution_space = typename DeviceType::execution_space;
  ABViewType _ab;
  PivViewType _ipiv;
  int _kl, _ku;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGbtrf(const ABViewType &ab, const PivViewType &ipiv,
                             int kl, int ku)
      : _ab(ab), _ipiv(ipiv), _kl(kl), _ku(ku) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto ab   = Kokkos::subview(_ab, k, Kokkos::ALL(), Kokkos::ALL());
    auto ipiv = Kokkos::subview(_ipiv, k, Kokkos::ALL());

    KokkosBatched::SerialGbtrf<AlgoTagType>::invoke(ab, ipiv, _kl, _ku);
  }

  inline void run() {
    using value_type = typename ABViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGbtrs");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::RangePolicy<execution_space> policy(0, _ab.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename DeviceType, typename AViewType, typename BViewType,
          typename PivViewType, typename ParamTagType, typename AlgoTagType>
struct Functor_BatchedSerialGbtrs {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  BViewType _b;
  PivViewType _ipiv;
  int _kl, _ku;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGbtrs(const AViewType &a, const BViewType &b,
                             const PivViewType &ipiv, int kl, int ku)
      : _a(a), _b(b), _ipiv(ipiv), _kl(kl), _ku(ku) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ParamTagType &, const int k, int &info) const {
    auto aa   = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb   = Kokkos::subview(_b, k, Kokkos::ALL());
    auto ipiv = Kokkos::subview(_ipiv, k, Kokkos::ALL());

    info += KokkosBatched::SerialGbtrs<typename ParamTagType::trans,
                               AlgoTagType>::invoke(aa, bb, ipiv, _kl, _ku);
  }

  inline int run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGbtrs");
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

template <typename DeviceType, typename ScalarType, typename AViewType,
          typename xViewType, typename yViewType, typename ParamTagType>
struct Functor_BatchedSerialGemv {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  xViewType _x;
  yViewType _y;
  ScalarType _alpha, _beta;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGemv(const ScalarType alpha, const AViewType &a,
                            const xViewType &x, const ScalarType beta,
                            const yViewType &y)
      : _alpha(alpha), _a(a), _x(x), _beta(beta), _y(y) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ParamTagType &, const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto xx = Kokkos::subview(_x, k, Kokkos::ALL());
    auto yy = Kokkos::subview(_y, k, Kokkos::ALL());

    KokkosBlas::SerialGemv<typename ParamTagType::trans,
                           Algo::Gemv::Unblocked>::invoke(_alpha, aa, xx, _beta,
                                                          yy);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGbtrs");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::RangePolicy<execution_space, ParamTagType> policy(0, _x.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename DeviceType, typename ScalarType, typename LayoutType,
          typename ParamTagType, typename AlgoTagType>
/// \brief Implementation details of batched gbtrs test
///        Confirm A * x = b, where
///        A: [[1, -3, -2,  0],
///            [-1, 1, -3, -2],
///            [2, -1,  1, -3],
///            [0,  2, -1,  1]]
///        b: [1, 1, 1, 1]
///        x: [67/81, 22/81, -40/81, -1/27] or [-1/27, -40/81, 22/81, 67/81]
///
///        This corresponds to the following system of equations:
///          x0 - 3 x1 - 2 x2        = 1
///        - x0 +   x1 - 3 x2 - 2 x3 = 1
///        2 x0 -   x1 +   x3 - 3 x3 = 1
///               2 x1 -   x2 +   x3 = 1
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_gbtrs_analytical(const int N) {
  using ats            = typename Kokkos::ArithTraits<ScalarType>;
  using RealType       = typename ats::mag_type;
  using View2DType  = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType  = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivViewType = Kokkos::View<int **, LayoutType, DeviceType>;

  constexpr int BlkSize = 4, kl = 2, ku = 2;
  constexpr int ldab = 2 * kl + ku + 1;
  View3DType A("A", N, BlkSize, BlkSize),
      AB("AB", N, ldab, BlkSize);                               // Banded matrix
  View2DType x0("x0", N, BlkSize), x_ref("x_ref", N, BlkSize);  // Solutions
  PivViewType piv("piv", N, BlkSize);

  using ArgTrans = typename ParamTagType::trans;
  auto h_A       = Kokkos::create_mirror_view(A);
  auto h_x_ref   = Kokkos::create_mirror_view(x_ref);
  for (std::size_t ib = 0; ib < N; ib++) {
    for (std::size_t i = 0; i < BlkSize; i++) {
      for (std::size_t j = 0; j < BlkSize; j++) {
        h_A(ib, i, j) = i == j       ? 1.0
                        : i == j - 1 ? -3.0
                        : i == j - 2 ? -2.0
                        : j == i - 1 ? -1.0
                        : j == i - 2 ? 2.0
                                     : 0.0;
      }
    }
    if constexpr (std::is_same_v<ArgTrans, KokkosBatched::Trans::NoTranspose>) {
      h_x_ref(ib, 0) = 67.0 / 81.0;
      h_x_ref(ib, 1) = 22.0 / 81.0;
      h_x_ref(ib, 2) = -40.0 / 81.0;
      h_x_ref(ib, 3) = -1.0 / 27.0;
    } else if constexpr (std::is_same_v<ArgTrans,
                                        KokkosBatched::Trans::Transpose>) {
      h_x_ref(ib, 0) = -1.0 / 27.0;
      h_x_ref(ib, 1) = -40.0 / 81.0;
      h_x_ref(ib, 2) = 22.0 / 81.0;
      h_x_ref(ib, 3) = 67.0 / 81.0;
    }
  }

  Kokkos::fence();
  Kokkos::deep_copy(x0, ScalarType(1.0));
  Kokkos::deep_copy(x_ref, h_x_ref);
  Kokkos::deep_copy(A, h_A);

  full_to_banded(A, AB, kl, ku);

  // gbtrf to factorize matrix A = P * L * U
  Functor_BatchedSerialGbtrf<DeviceType, View3DType, PivViewType, AlgoTagType>(
      AB, piv, kl, ku)
      .run();

  // gbtrs (Note, Ab is a factorized matrix of A)
  auto info = Functor_BatchedSerialGbtrs<DeviceType, View3DType, View2DType, PivViewType,
                             ParamTagType, AlgoTagType>(AB, x0, piv, kl, ku)
      .run();

  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();
  auto h_x0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x0);

  // Check x0 = [67/81, 22/81, -40/81, -1/27] or [-1/27, -40/81, 22/81, 67/81]
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      EXPECT_NEAR_KK(h_x0(ib, i), h_x_ref(ib, i), eps);
    }
  }
}

template <typename DeviceType, typename ScalarType, typename LayoutType,
          typename ParamTagType, typename AlgoTagType>
/// \brief Implementation details of batched gbtrs test
///        Confirm A * x = b, where
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_gbtrs(const int N, const int k, const int BlkSize) {
  using ats            = typename Kokkos::ArithTraits<ScalarType>;
  using RealType       = typename ats::mag_type;
  using View2DType  = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType  = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivViewType = Kokkos::View<int **, LayoutType, DeviceType>;

  const int kl = k, ku = k;
  const int ldab = 2 * kl + ku + 1;
  View3DType A("A", N, BlkSize, BlkSize), tmp_A("tmp_A", N, BlkSize, BlkSize),
      AB("AB", N, ldab, BlkSize);  // Banded matrix
  View2DType x0("x0", N, BlkSize), x_ref("x_ref", N, BlkSize),
      y0("y0", N, BlkSize);  // Solutions
  PivViewType piv("piv", N, BlkSize);

  // Create a random matrix A and make it Positive Definite Symmetric
  using execution_space = typename DeviceType::execution_space;
  Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(13718);
  ScalarType randStart, randEnd;

  // Initialize tmp_A with random matrix
  KokkosKernels::Impl::getRandomBounds(1.0, randStart, randEnd);
  Kokkos::fill_random(A, rand_pool, randStart, randEnd);

  // Make the matrix Positive Definite Symmetric and Diagonal dominant
  random_to_pds(A, tmp_A);
  Kokkos::deep_copy(A, ScalarType(0.0));

  full_to_banded(tmp_A, AB, kl, ku);  // In banded storage
  banded_to_full(AB, A, kl, ku);      // In full storage

  Kokkos::fence();

  // Create an initial solution vector x0 = [1, 1, 1, ...]
  Kokkos::deep_copy(x0, ScalarType(1.0));
  auto h_x_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x0);

  // gbtrf to factorize matrix A = P * L * U
  Functor_BatchedSerialGbtrf<DeviceType, View3DType, PivViewType, AlgoTagType>(
      AB, piv, kl, ku)
      .run();

  // gbtrs (Note, Ab is a factorized matrix of A)
  auto info = Functor_BatchedSerialGbtrs<DeviceType, View3DType, View2DType, PivViewType,
                             ParamTagType, AlgoTagType>(AB, x0, piv, kl, ku)
      .run();
  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // Gemv to compute A*x0, this should be identical to x_ref
  Functor_BatchedSerialGemv<DeviceType, ScalarType, View3DType, View2DType,
                            View2DType, ParamTagType>(1.0, A, x0, 0.0, y0)
      .run();

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();

  // Check A * x0 = x_ref
  auto h_y0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y0);
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      EXPECT_NEAR_KK(h_y0(ib, i), h_x_ref(ib, i), eps);
    }
  }
}

}  // namespace Gbtrs
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename ParamTagType,  typename AlgoTagType>
int test_batched_gbtrs() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Gbtrs::impl_test_batched_gbtrs_analytical<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1);
    Test::Gbtrs::impl_test_batched_gbtrs_analytical<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(2);
    for (int i = 0; i < 10; i++) {
      int k = 1;
      Test::Gbtrs::impl_test_batched_gbtrs<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1, k, i);
      Test::Gbtrs::impl_test_batched_gbtrs<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(2, k, i);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Gbtrs::impl_test_batched_gbtrs_analytical<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1);
    Test::Gbtrs::impl_test_batched_gbtrs_analytical<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(2);
    for (int i = 0; i < 10; i++) {
      int k = 1;
      Test::Gbtrs::impl_test_batched_gbtrs<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1, k, i);
      Test::Gbtrs::impl_test_batched_gbtrs<DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(2, k, i);
    }
  }
#endif

  return 0;
}
