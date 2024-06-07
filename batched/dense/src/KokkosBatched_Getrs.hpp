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
#ifndef KOKKOSBATCHED_GETRS_HPP_
#define KOKKOSBATCHED_GETRS_HPP_

#include <KokkosBatched_Util.hpp>

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

namespace KokkosBatched {

/// \brief Serial Batched Getrs:
///
/// Solve A_l x_l = b_l or A**T_l x_l = b_l for all l = 0, ..., N
///   with a general N-by-N matrix A using the LU factorization computed
///   by getrf.
///
/// \tparam AViewType: Input type for the matrix, needs to be a 2D view
/// \tparam PivViewType: Integer type for pivot indices, needs to be a 1D view
/// \tparam BViewType: Input type for the right-hand side and the solution,
/// needs to be a 1D view
///
/// \param A [in]: A is a lda by n banded matrix.
/// The factors L and U from the factorization A = P*L*U
/// as computed by getrf.
/// \param b [inout]: right-hand side and the solution
/// \param piv [in]: The pivot indices; for 0 <= i < N, row i of the matrix
/// was interchanged with row piv(i).
///
/// No nested parallel_for is used inside of the function.
///

template <typename ArgTrans, typename ArgAlgo>
struct SerialGetrs {
  template <typename AViewType, typename PivViewType, typename BViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A,
                                           const PivViewType &piv,
                                           const BViewType &b);
};
}  // namespace KokkosBatched

#include "KokkosBatched_Getrs_Serial_Impl.hpp"

#endif  // KOKKOSBATCHED_GETRS_HPP_