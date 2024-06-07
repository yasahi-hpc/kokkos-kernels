#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas2_gemv.hpp>
#include "KokkosBatched_Pbtrs.hpp"
#include "Test_Utils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace Pbtrs {

template <typename U>
struct ParamTag {
  using uplo = U;
};

template <typename DeviceType, typename AViewType, typename BViewType,
          typename ParamTagType, typename AlgoTagType>
struct Functor_BatchedSerialPbtrs {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  BViewType _b;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialPbtrs(const AViewType &a, const BViewType &b)
      : _a(a), _b(b) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ParamTagType &, const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL());

    KokkosBatched::SerialPbtrs<typename ParamTagType::uplo,
                               AlgoTagType>::invoke(aa, bb);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialPbtrs");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space, ParamTagType> policy(0, _b.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename ScalarType, typename AViewType,
          typename xViewType, typename yViewType>
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
  void operator()(const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto xx = Kokkos::subview(_x, k, Kokkos::ALL());
    auto yy = Kokkos::subview(_y, k, Kokkos::ALL());

    KokkosBlas::SerialGemv<Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
        _alpha, aa, xx, _beta, yy);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialGemv");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space> policy(0, _x.extent(0));
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
void impl_test_batched_pbtrs_analytical(const int N) {
  using View2DType = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;

  constexpr int BlkSize = 5, k = 1;
  View3DType A("A", N, BlkSize, BlkSize), ref("Ref", N, BlkSize, BlkSize);
  View3DType Ab("Ab", N, k + 1, BlkSize);  // Banded matrix
  View2DType x0("x0", N, BlkSize), x_ref("x_ref", N, BlkSize),
      y0("y0", N, BlkSize);  // Solutions

  auto h_A     = Kokkos::create_mirror_view(A);
  auto h_Ab    = Kokkos::create_mirror_view(Ab);
  auto h_ref   = Kokkos::create_mirror_view(ref);
  auto h_x0    = Kokkos::create_mirror_view(x0);
  auto h_x_ref = Kokkos::create_mirror_view(x_ref);

  // FIXME before implementing pbtrf, we need to have a reference factorized
  // matrix
  std::vector<ScalarType> banded_ref = {
      1.41421356, 1.22474487, 1.15470054, 1.11803399, 1.09544512,
      0.70710678, 0.81649658, 0.8660254,  0.89442719, 0.};
  std::vector<ScalarType> banded_ref_u = {
      0.,         0.70710678, 0.81649658, 0.8660254,  0.89442719,
      1.41421356, 1.22474487, 1.15470054, 1.11803399, 1.09544512};

  for (std::size_t ib = 0; ib < N; ib++) {
    for (std::size_t i = 0; i < BlkSize; i++) {
      for (std::size_t j = 0; j < BlkSize; j++) {
        h_ref(ib, i, j) = i == j ? 2.0 : 1.0;
      }
    }

    for (std::size_t n = 0; n < BlkSize; n++) {
      h_x0(ib, n)    = 1;
      h_x_ref(ib, n) = 1;
    }

    for (std::size_t i = 0; i < k + 1; i++) {
      for (std::size_t j = 0; j < BlkSize; j++) {
        if constexpr (std::is_same_v<typename ParamTagType::uplo,
                                     KokkosBatched::Uplo::Lower>) {
          std::size_t idx = j + i * BlkSize;
          h_Ab(ib, i, j)  = static_cast<ScalarType>(banded_ref.at(idx));
        } else {
          std::size_t idx = j + i * BlkSize;
          h_Ab(ib, i, j)  = static_cast<ScalarType>(banded_ref_u.at(idx));
        }
      }
    }
  }

  Kokkos::fence();

  Kokkos::deep_copy(ref, h_ref);
  Kokkos::deep_copy(x0, h_x0);
  Kokkos::deep_copy(x_ref, h_x_ref);
  Kokkos::deep_copy(Ab, h_Ab);

  create_banded_pds_matrix<View3DType, View3DType, typename ParamTagType::uplo>(
      ref, A, k, false);

  // pbtrs (Note, Ab is a factorized matrix of A)
  Functor_BatchedSerialPbtrs<DeviceType, View3DType, View2DType, ParamTagType,
                             AlgoTagType>(Ab, x0)
      .run();

  // Gemv to compute A*x0, this should be identical to x_ref
  Functor_BatchedSerialGemv<DeviceType, ScalarType, View3DType, View2DType,
                            View2DType>(1.0, A, x0, 0.0, y0)
      .run();

  // this eps is about 10^-14
  using ats      = typename Kokkos::ArithTraits<ScalarType>;
  using mag_type = typename ats::mag_type;
  mag_type eps   = 1.0e3 * ats::epsilon();

  // Check x0 = x1
  using execution_space = typename DeviceType::execution_space;
  EXPECT_TRUE(allclose<execution_space>(y0, x_ref, 1.e-4, eps));
}

}  // namespace Pbtrs
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename ParamTagType,
          typename AlgoTagType>
int test_batched_pbtrs() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Pbtrs::impl_test_batched_pbtrs_analytical<
        DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1);
  }
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Pbtrs::impl_test_batched_pbtrs_analytical<
        DeviceType, ScalarType, LayoutType, ParamTagType, AlgoTagType>(1);
  }
#endif
  return 0;
}