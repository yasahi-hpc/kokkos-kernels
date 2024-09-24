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
#include <KokkosBatched_Gbtrf.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include "Test_Batched_DenseUtils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace Gbtrf {

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
  void operator()(const int k, int &info) const {
    auto ab   = Kokkos::subview(_ab, k, Kokkos::ALL(), Kokkos::ALL());
    auto ipiv = Kokkos::subview(_ipiv, k, Kokkos::ALL());

    info += KokkosBatched::SerialGbtrf<AlgoTagType>::invoke(ab, ipiv, _kl, _ku);
  }

  inline int run() {
    using value_type = typename ABViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGbtrf");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    int info_sum                      = 0;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space> policy(0, _ab.extent(0));
    Kokkos::parallel_reduce(name.c_str(), policy, *this, info_sum);
    Kokkos::Profiling::popRegion();
    return info_sum;
  }
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
    std::string name_region("KokkosBatched::Test::SerialGbtrf");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::RangePolicy<execution_space> policy(0, _a.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename DeviceType, typename ScalarType, typename LayoutType,
          typename AlgoTagType>
/// \brief Implementation details of batched gbtrf test
///
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_gbtrf(const int N) {
  using ats           = typename Kokkos::ArithTraits<ScalarType>;
  using RealType      = typename ats::mag_type;
  using View2DType    = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType    = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivView2DType = Kokkos::View<int **, LayoutType, DeviceType>;

  constexpr int BlkSize = 10, kl = 2, ku = 2;
  constexpr int ldab = 2 * kl + ku + 1;
  View3DType A("A", N, BlkSize, BlkSize),
      A_reconst("A_reconst", N, BlkSize, BlkSize),
      NL("NL", N, BlkSize, BlkSize), L("L", N, BlkSize, BlkSize),
      U("U", N, BlkSize, BlkSize), U_ref("U_ref", N, BlkSize, BlkSize),
      I("I", N, BlkSize, BlkSize);

  View3DType AB("AB", N, ldab, BlkSize),
      AB_upper("AB_upper", N, kl + ku + 1, BlkSize);
  View2DType ones(Kokkos::view_alloc("ones", Kokkos::WithoutInitializing), N,
                  BlkSize);
  PivView2DType ipiv("ipiv", N, BlkSize), ipiv_ref("ipiv_ref", N, BlkSize);

  auto h_A = Kokkos::create_mirror_view(A);
  for (std::size_t ib = 0; ib < N; ib++) {
    for (std::size_t i = 0; i < BlkSize; i++) {
      for (std::size_t j = 0; j < BlkSize; j++) {
        h_A(ib, i, j) = i == j         ? 1.0
                        : i == (j - 1) ? -3.0
                        : i == (j + 1) ? -1.0
                        : i == (j - 2) ? -2.0
                        : i == (j + 2) ? 2.0
                                       : 0.0;
      }
    }
  }

  Kokkos::deep_copy(A, h_A);
  Kokkos::deep_copy(ones, RealType(1.0));

  full_to_banded(A, AB, kl, ku);
  create_diagonal_matrix(ones, I);
  Kokkos::fence();

  // gbtrf to factorize matrix A = P * L * U
  auto info = Functor_BatchedSerialGbtrf<DeviceType, View3DType, PivView2DType,
                             AlgoTagType>(AB, ipiv, kl, ku)
      .run();
  
  Kokkos::fence();
  EXPECT_EQ(info, 0);

  // extract matrix U from AB
  // first take subview, note u is stored at 0:kl+ku+1
  auto sub_AB = Kokkos::subview(
      AB, Kokkos::ALL(), Kokkos::pair<int, int>(0, kl + ku + 1), Kokkos::ALL());
  Kokkos::deep_copy(AB_upper, sub_AB);

  banded_to_full<View3DType, View3DType, KokkosBatched::Uplo::Upper>(
      AB_upper, U, kl + ku);

  // Reference is made from getrf
  // getrf to factorize matrix A = P * L * U
  Functor_BatchedSerialGetrf<DeviceType, View3DType, PivView2DType,
                             AlgoTagType>(A, ipiv_ref)
      .run();

  // Copy upper triangular components to U_ref
  create_triangular_matrix<View3DType, View3DType, KokkosBatched::Uplo::Upper>(
      A, U_ref);

  // this eps is about 10^-14
  RealType eps = 1.0e3 * ats::epsilon();

  auto h_U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U);
  auto h_U_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U_ref);

  // Check if U (gbtrf) == U_ref (getrf)
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      for (int j = 0; i < BlkSize; j++) {
        EXPECT_NEAR_KK(h_U(ib, i, j), h_U_ref(ib, i, j), eps);
      }
    }
  }
}

}  // namespace Gbtrf
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename AlgoTagType>
int test_batched_gbtrf() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Gbtrf::impl_test_batched_gbtrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(1);
    Test::Gbtrf::impl_test_batched_gbtrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(2);
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Gbtrf::impl_test_batched_gbtrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(1);
    Test::Gbtrf::impl_test_batched_gbtrf<DeviceType, ScalarType, LayoutType, AlgoTagType>(2);
  }
#endif

  return 0;
}
