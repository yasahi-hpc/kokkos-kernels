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

#ifndef KOKKOSBATCHED_PTTS2_SERIAL_IMPL_HPP_
#define KOKKOSBATCHED_PTTS2_SERIAL_IMPL_HPP_

#include <KokkosBlas1_scal.hpp>

namespace KokkosBatched {
template <typename AlgoType>
struct SerialPtts2Internal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const int n, const ValueType *KOKKOS_RESTRICT d, const int ds0,
      const ValueType *KOKKOS_RESTRICT e, const int es0,
      ValueType *KOKKOS_RESTRICT b, const int bs0, const int ldb);
};

template <>
template <typename ValueType>
KOKKOS_INLINE_FUNCTION int SerialPtts2Internal<Algo::Ptts2::Unblocked>::invoke(
    const int n, const ValueType *KOKKOS_RESTRICT d, const int ds0,
    const ValueType *KOKKOS_RESTRICT e, const int es0,
    ValueType *KOKKOS_RESTRICT b, const int bs0, const int ldb) {
  // Solve L * x = b
  for (int i = 1; i < n; i++) {
    b[i * bs0] -= e[(i - 1) * es0] * b[(i - 1) * bs0];
  }

  b[(n - 1) * bs0] /= d[(n - 1) * ds0];

  for (int i = n - 2; i >= 0; i--) {
    b[i * bs0] = b[i * bs0] / d[i * ds0] - b[(i + 1) * bs0] * e[i * es0];
  }

  return 0;
}

template <>
struct SerialPtts2<Algo::Ptts2::Unblocked> {
  template <typename DViewType, typename EViewType, typename BViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const DViewType &d,
                                           const EViewType &e,
                                           const BViewType &b) {
    using ScalarType = typename DViewType::non_const_value_type;
    int n            = d.extent(0);
    int ldb          = b.extent(0);

    if (n == 1) {
      const ScalarType alpha = 1.0 / d(0);
      return KokkosBlas::SerialScale::invoke(alpha, b);
    }

    // Solve A * X = B using the factorization A = L*D*L**T,
    // overwriting each right hand side vector with its solution.
    return SerialPtts2Internal<Algo::Ptts2::Unblocked>::invoke(
        n, d.data(), d.stride(0), e.data(), e.stride(0), b.data(), b.stride(0),
        ldb);
  }
};
}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_PTTS2_SERIAL_IMPL_HPP_