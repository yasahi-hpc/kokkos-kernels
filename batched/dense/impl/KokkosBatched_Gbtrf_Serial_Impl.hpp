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

#ifndef KOKKOSBATCHED_GBTRF_SERIAL_IMPL_HPP_
#define KOKKOSBATCHED_GBTRF_SERIAL_IMPL_HPP_

#include <Kokkos_Swap.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBatched_Iamax.hpp>

namespace KokkosBatched {
template <typename ABViewType>
KOKKOS_INLINE_FUNCTION static int checkGbtrfInput(
    [[maybe_unused]] const ABViewType &AB, [[maybe_unused]] const int kl,
    [[maybe_unused]] const int ku, [[maybe_unused]] const int m) {
  static_assert(Kokkos::is_view_v<ABViewType>,
                "KokkosBatched::gbtrf: ABViewType is not a Kokkos::View.");
  static_assert(ABViewType::rank == 2,
                "KokkosBatched::gbtrs: ABViewType must have rank 2.");
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  if (m < 0) {
    Kokkos::printf(
        "KokkosBatched::gbtrf: input parameter m must not be less than 0: m "
        "= "
        "%d\n",
        m);
    return 1;
  }

  if (kl < 0) {
    Kokkos::printf(
        "KokkosBatched::gbtrf: input parameter kl must not be less than 0: kl "
        "= "
        "%d\n",
        kl);
    return 1;
  }

  if (ku < 0) {
    Kokkos::printf(
        "KokkosBatched::gbtrf: input parameter ku must not be less than 0: ku "
        "= "
        "%d\n",
        ku);
    return 1;
  }

  const int lda = AB.extent(0), n = AB.extent(1);
  if (lda < (2 * kl + ku + 1)) {
    Kokkos::printf(
        "KokkosBatched::gbtrs: leading dimension of A must be smaller than 2 * "
        "kl + ku + 1: "
        "lda = %d, kl = %d, ku = %d\n",
        lda, kl, ku);
    return 1;
  }
#endif
  return 0;
}

template <>
struct SerialGbtrf<Algo::Trsv::Unblocked> {
  template <typename ABViewType, typename PivViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const ABViewType &AB, const PivViewType &piv, const int kl, const int ku,
      const int _m = -1) {
    using ScalarType = typename ABViewType::non_const_value_type;
    int m            = _m > 0 ? _m: AB.extent(1);

    auto info = checkGbtrfInput(AB, kl, ku, m);
    if (info) return info;

    // Quick return if possible
    int ldab = AB.extent(0), n = AB.extent(1);
    if (m == 0 || n == 0) return 0;

    // Upper bandwidth of U factor
    const int kv = ku + kl;

    // Gaussian elimination with partial pivoting
    // Set fill-in elements in columns KU+2 to KV to zero
    for (int j = ku + 1; j < Kokkos::min(kv, n); ++j) {
      for (int i = kv - j + 1; i < kl; ++i) {
        AB(i, j) = 0;
      }
    }

    // JU is the index of the last column affected by the current stage of
    // the factorization
    int ju = 0;

    for (int j = 0; j < Kokkos::min(m, n); ++j) {
      // Set fill-in elements in column J+KV to zero
      if (j + kv < n) {
        for (int i = 0; i < kl; ++i) {
          AB(i, j + kv) = 0;
        }
      }

      // Find pivot and test for singularity. KM is the number of subdiagonals
      // elements in the current column.
      int km = Kokkos::min(kl, m - j - 1);
      auto sub_col_AB =
          Kokkos::subview(AB, Kokkos::pair<int, int>(kv, kv + km + 1), j);
      int jp = SerialIamax::invoke(sub_col_AB);
      piv(j) = jp + j;

      if (AB(kv + jp, j) == 0) {
        // If pivot is zero, set INFO to the index of the pivot unless a
        // zero pivot has already been found.
        if (info == 0) info = j + 1;
      } else {
        ju = Kokkos::max(ju, Kokkos::min(j + ku + jp, n - 1));

        // Apply the interchange to columns J to JU
        if (jp != 0) {
          for (int k = 0; k < ju - j + 1; ++k) {
            Kokkos::kokkos_swap(AB(kv + jp - k, j + k), AB(kv - k, j + k));
          }
        }

        if (km > 0) {
          // Compute multipliers
          const ScalarType alpha = 1.0 / AB(kv, j);
          auto sub_col_AB        = Kokkos::subview(
              AB, Kokkos::pair<int, int>(kv + 1, kv + km + 1), j);
          [[maybe_unused]] auto info_scal =
              KokkosBlas::SerialScale::invoke(alpha, sub_col_AB);

          // Update trailing submatrix within the band
          if (ju > j) {
            auto x = Kokkos::subview(
                AB, Kokkos::pair<int, int>(kv + 1, kv + km + 1), j);

            // dger with alpha = -1.0
            for (int k = 0; k < ju - j; ++k) {
              auto y_k = AB(kv - 1 - k, j + k + 1);
              if (y_k != 0) {
                auto temp = -1.0 * y_k;
                for (int i = 0; i < km; ++i) {
                  AB(kv + i - k, j + k + 1) += x(i) * temp;
                }
              }
            }
          }
        }
      }
    }
    return 0;
  }
};

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_GBTRF_SERIAL_IMPL_HPP_
