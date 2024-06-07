#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas2_gemv.hpp>
#include "KokkosBatched_Gbtrs.hpp"
#include "../impl/KokkosBatched_Gbtrs_Serial_Impl.hpp"
#include "Test_Utils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace Gbtrs {

template <typename T>
struct ParamTag {
  using trans = T;
};

template <typename DeviceType, typename AViewType, typename BViewType,
          typename PivViewType, typename ParamTagType, typename AlgoTagType>
struct Functor_BatchedSerialGbtrs {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  BViewType _b;
  PivViewType _piv;
  int _kl, _ku;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGbtrs(const AViewType &a, const BViewType &b,
                             const PivViewType &piv, int kl, int ku)
      : _a(a), _b(b), _piv(piv), _kl(kl), _ku(ku) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ParamTagType &, const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());

    // We need to make bb two dimensional
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL());

    KokkosBatched::SerialGbtrs<typename ParamTagType::trans,
                               AlgoTagType>::invoke(aa, bb, _piv, _kl, _ku);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGbtrs");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space, ParamTagType> policy(0, _b.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
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
    std::string name_region("KokkosBatched::Test::SerialGemv");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space, ParamTagType> policy(0, _x.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename ScalarType, typename LayoutType,
          typename ParamTagType, typename AlgoTagType>
/// \brief Implementation details of batched pbtrs test
///
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_gbtrs_analytical(const int N) {
  using View2DType  = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType  = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;
  using PivViewType = Kokkos::View<int *, LayoutType, DeviceType>;

  constexpr int BlkSize = 10, kl = 2, ku = 2;
  constexpr int ldab = 2 * kl + ku + 1;
  View3DType A("A", N, BlkSize, BlkSize), ref("Ref", N, BlkSize, BlkSize);
  View3DType Ab("Ab", N, ldab, BlkSize);  // Banded matrix
  View2DType x0("x0", N, BlkSize), x_ref("x_ref", N, BlkSize),
      y0("y0", N, BlkSize);  // Solutions
  PivViewType piv("piv", BlkSize);

  auto h_A     = Kokkos::create_mirror_view(A);
  auto h_Ab    = Kokkos::create_mirror_view(Ab);
  auto h_ref   = Kokkos::create_mirror_view(ref);
  auto h_x0    = Kokkos::create_mirror_view(x0);
  auto h_x_ref = Kokkos::create_mirror_view(x_ref);
  auto h_piv   = Kokkos::create_mirror_view(piv);

  // FIXME before implementing gbtrf, we need to have a reference factorized
  // matrix and pivot
  std::vector<ScalarType> banded_ref = {
      0.0,         0.0,         0.0,         0.0,         -2.0,
      0.0,         0.0,         0.0,         -2.0,        0.0,
      0.0,         0.0,         0.0,         -3.0,        1.0,
      0.0,         0.0,         -3.0,        -0.34567901, 0.0,
      0.0,         0.0,         1.0,         1.5,         -0.8,
      -2.0,        1.0,         -0.51851852, -0.6011396,  -3.0,
      0.0,         -1.0,        -2.5,        -3.2,        -1.4,
      -1.0,        -1.82716049, -2.9017094,  0.64552709,  -4.58683191,
      2.0,         -2.5,        -3.0,        5.4,         2.0,
      -4.33333333, -3.39173789, -2.71104578, -2.27621998, -0.15769414,
      -0.5,        -0.2,        1.0,         -0.58024691, -0.24074074,
      -0.34615385, 0.54346913,  -0.8622773,  0.23010958,  0.0,
      0.5,         -0.8,        -0.66666667, 0.37037037,  -0.17283951,
      -0.46153846, -0.58966821, -0.7377227,  0.0,         0.0};
  std::vector<int> ipiv_ref = {2, 2, 2, 3, 6, 6, 6, 8, 8, 9};

  for (std::size_t ib = 0; ib < N; ib++) {
    for (std::size_t i = 0; i < BlkSize; i++) {
      for (std::size_t j = 0; j < BlkSize; j++) {
        h_ref(ib, i, j) = i == j       ? 1.0
                          : i == j - 1 ? -3.0
                          : i == j - 2 ? -2.0
                          : j == i - 1 ? -1.0
                          : j == i - 2 ? 2.0
                                       : 0.0;
      }
    }

    for (std::size_t n = 0; n < BlkSize; n++) {
      h_x0(ib, n)    = 1;
      h_x_ref(ib, n) = 1;
    }

    for (std::size_t i = 0; i < ldab; i++) {
      for (std::size_t j = 0; j < BlkSize; j++) {
        std::size_t idx = j + i * BlkSize;
        h_Ab(ib, i, j)  = static_cast<ScalarType>(banded_ref.at(idx));
      }
    }
  }

  for (std::size_t j = 0; j < BlkSize; j++) {
    h_piv(j) = ipiv_ref.at(j);
  }

  Kokkos::fence();

  Kokkos::deep_copy(ref, h_ref);
  Kokkos::deep_copy(A, h_ref);
  Kokkos::deep_copy(x0, h_x0);
  Kokkos::deep_copy(x_ref, h_x_ref);
  Kokkos::deep_copy(Ab, h_Ab);
  Kokkos::deep_copy(piv, h_piv);

  // gbtrs (Note, Ab is a factorized matrix of A)
  Functor_BatchedSerialGbtrs<DeviceType, View3DType, View2DType, PivViewType,
                             ParamTagType, AlgoTagType>(Ab, x0, piv, kl, ku)
      .run();

  // Gemv to compute A*x0, this should be identical to x_ref
  Functor_BatchedSerialGemv<DeviceType, ScalarType, View3DType, View2DType,
                            View2DType, ParamTagType>(1.0, A, x0, 0.0, y0)
      .run();

  // this eps is about 10^-14
  using ats      = typename Kokkos::ArithTraits<ScalarType>;
  using mag_type = typename ats::mag_type;
  mag_type eps   = 1.0e3 * ats::epsilon();

  // Check x0 = x1
  using execution_space = typename DeviceType::execution_space;
  EXPECT_TRUE(allclose<execution_space>(y0, x_ref, 1.e-4, eps));
}

}  // namespace Gbtrs
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename ParamTagType,
          typename AlgoTagType>
int test_batched_gbtrs() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Gbtrs::impl_test_batched_gbtrs_analytical<
        DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1);
  }
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Gbtrs::impl_test_batched_gbtrs_analytical<
        DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1);
  }
#endif
  return 0;
}