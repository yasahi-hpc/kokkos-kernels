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
#ifndef KOKKOSBATCHED_PBTRS_HPP_
#define KOKKOSBATCHED_PBTRS_HPP_

#include <KokkosBatched_Util.hpp>

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

namespace KokkosBatched {

/// \brief Serial Batched Pbtrs:
///
/// Solve Ab_l x_l = b_l for all l = 0, ..., N
///   with a symmetric positive definite band matrix A using the Cholesky
///   factorization A = U**T*U or A = L*L**T computed by pbtrf.
/// \tparam AViewType: Input type for the matrix, needs to be a 2D view
/// \tparam BViewType: Input type for the right-hand side and the solution,
/// needs to be a 1D view
///
/// \param A [in]: A is a ldab by n banded matrix
/// The triangular factor U or L from the Cholesky factorization
/// A = U**T*U or A = L*L**T of the band matrix A, stored in the
/// first KD+1 rows of the array.  The j-th column of U or L is
/// stored in the j-th column of the array AB as follows:
/// if UPLO ='U', AB(kd+1+i-j,j) = U(i,j) for max(1,j-kd)<=i<=j;
/// if UPLO ='L', AB(1+i-j,j)    = L(i,j) for j<=i<=min(n,j+kd).
/// \param b [inout]: right-hand side and the solution, a rank 1 view
///
/// No nested parallel_for is used inside of the function.
///

template <typename ArgUplo, typename ArgAlgo>
struct SerialPbtrs {
  template <typename ABViewType, typename BViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const ABViewType &Ab,
                                           const BViewType &b);
};
}  // namespace KokkosBatched

#include "KokkosBatched_Pbtrs_Serial_Impl.hpp"

#endif  // KOKKOSBATCHED_PBTRS_HPP_