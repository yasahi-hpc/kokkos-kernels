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

#ifndef KOKKOSBATCHED_GETRS_SERIAL_IMPL_HPP_
#define KOKKOSBATCHED_GETRS_SERIAL_IMPL_HPP_

#include <KokkosBatched_Util.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Laswp.hpp>

namespace KokkosBatched {

template <typename AViewType, typename BViewType>
KOKKOS_INLINE_FUNCTION static int checkGetrsInput([[maybe_unused]] const AViewType &A,
                                                  [[maybe_unused]] const BViewType &b) {
  static_assert(Kokkos::is_view<AViewType>::value, "KokkosBatched::getrs: AViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<BViewType>::value, "KokkosBatched::getrs: BViewType is not a Kokkos::View.");
  static_assert(AViewType::rank == 2, "KokkosBatched::getrs: AViewType must have rank 2.");
  static_assert(BViewType::rank == 1, "KokkosBatched::getrs: BViewType must have rank 1.");
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  const int lda = A.extent(0), n = A.extent(1);
  if (lda < Kokkos::max(1, n)) {
    Kokkos::printf(
        "KokkosBatched::getrs: the leading dimension of the array A must "
        "satisfy lda >= max(1, n): A: "
        "%d "
        "x %d \n",
        lda, n);
    return 1;
  }

  const int ldb = b.extent(0);
  if (ldb < Kokkos::max(1, n)) {
    Kokkos::printf(
        "KokkosBatched::getrs: the leading dimension of the array b must "
        "satisfy ldb >= max(1, n): b: %d, A: "
        "%d "
        "x %d \n",
        ldb, lda, n);
    return 1;
  }
#endif
  return 0;
}

//// Non-transpose ////
template <>
struct SerialGetrs<Trans::NoTranspose, Algo::Getrs::Unblocked> {
  template <typename AViewType, typename PivViewType, typename BViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A, const PivViewType &piv, const BViewType &b) {
    // quick return if possible
    if (A.extent(1) == 0) return 0;

    auto info = checkGetrsInput(A, b);
    if (info) return info;

    [[maybe_unused]] auto info_laswp = KokkosBatched::SerialLaswp<Side::Left, Direct::Forward>::invoke(piv, b);

    auto info_trsm = KokkosBatched::SerialTrsm<Side::Left, Uplo::Lower, Trans::NoTranspose, Diag::Unit,
                                               Algo::Trsm::Unblocked>::invoke(1.0, A, b);
    info_trsm      = KokkosBatched::SerialTrsm<Side::Left, Uplo::Upper, Trans::NoTranspose, Diag::NonUnit,
                                          Algo::Trsm::Unblocked>::invoke(1.0, A, b);

    return 0;
  }
};

//// Transpose ////
template <>
struct SerialGetrs<Trans::Transpose, Algo::Getrs::Unblocked> {
  template <typename AViewType, typename PivViewType, typename BViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A, const PivViewType &piv, const BViewType &b) {
    // quick return if possible
    if (A.extent(1) == 0) return 0;

    auto info = checkGetrsInput(A, b);
    if (info) return info;

    auto info_trsm = KokkosBatched::SerialTrsm<Side::Left, Uplo::Upper, Trans::Transpose, Diag::NonUnit,
                                               Algo::Trsm::Unblocked>::invoke(1.0, A, b);
    info_trsm =
        KokkosBatched::SerialTrsm<Side::Left, Uplo::Lower, Trans::Transpose, Diag::Unit, Algo::Trsm::Unblocked>::invoke(
            1.0, A, b);

    [[maybe_unused]] auto info_laswp = KokkosBatched::SerialLaswp<Side::Left, Direct::Backward>::invoke(piv, b);

    return 0;
  }
};
}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_GETRS_SERIAL_IMPL_HPP_
