#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas2_gemv.hpp>
#include "KokkosBatched_Ptts2.hpp"
#include "KokkosBatched_Pttrs.hpp"
#include "Test_Utils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace Pttrs {

template <typename DeviceType, typename DViewType, typename EViewType,
          typename BViewType, typename AlgoTagType>
struct Functor_BatchedSerialPtts2 {
  using execution_space = typename DeviceType::execution_space;
  DViewType _d;
  EViewType _e;
  BViewType _b;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialPtts2(const DViewType &d, const EViewType &e,
                             const BViewType &b)
      : _d(d), _e(e), _b(b) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto dd = Kokkos::subview(_d, k, Kokkos::ALL());
    auto ee = Kokkos::subview(_e, k, Kokkos::ALL());
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL());

    KokkosBatched::SerialPtts2<AlgoTagType>::invoke(dd, ee, bb);
  }

  inline void run() {
    using value_type = typename BViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialPtts2");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space> policy(0, _b.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename DViewType, typename EViewType,
          typename BViewType, typename AlgoTagType>
struct Functor_BatchedSerialPttrs {
  using execution_space = typename DeviceType::execution_space;
  DViewType _d;
  EViewType _e;
  BViewType _b;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialPttrs(const DViewType &d, const EViewType &e,
                             const BViewType &b)
      : _d(d), _e(e), _b(b) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto dd = Kokkos::subview(_d, k, Kokkos::ALL());
    auto ee = Kokkos::subview(_e, k, Kokkos::ALL());
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL());

    KokkosBatched::SerialPttrs<AlgoTagType>::invoke(dd, ee, bb);
  }

  inline void run() {
    using value_type = typename BViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialPttrs");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space> policy(0, _b.extent(0));
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
          typename AlgoTagType>
/// \brief Implementation details of batched pbtrs test
///
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_ptts2_analytical(const int N) {
  using View2DType = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;

  constexpr int BlkSize = 5, k = 1;
  View3DType A("A", N, BlkSize, BlkSize), ref("Ref", N, BlkSize, BlkSize);
  View2DType d("d", N, BlkSize), e("e", N, BlkSize - 1), x0("x0", N, BlkSize),
      x_ref("x_ref", N, BlkSize), y0("y0", N, BlkSize);  // Solutions

  auto h_A     = Kokkos::create_mirror_view(A);
  auto h_d     = Kokkos::create_mirror_view(d);
  auto h_e     = Kokkos::create_mirror_view(e);
  auto h_ref   = Kokkos::create_mirror_view(ref);
  auto h_x0    = Kokkos::create_mirror_view(x0);
  auto h_x_ref = Kokkos::create_mirror_view(x_ref);

  // FIXME before implementing pttrf, we need to have a reference factorized
  // matrix
  std::vector<ScalarType> d_fact_ref = {4.0, 3.75, 3.7333333, 3.732143,
                                        3.7320573};
  std::vector<ScalarType> e_fact_ref = {0.25, 0.26666667, 0.26785713,
                                        0.26794258};

  for (std::size_t ib = 0; ib < N; ib++) {
    for (std::size_t i = 0; i < BlkSize; i++) {
      for (std::size_t j = 0; j < BlkSize; j++) {
        h_ref(ib, i, j) = i == j ? 4.0 : 1.0;
      }
    }

    for (std::size_t n = 0; n < BlkSize; n++) {
      h_x0(ib, n)    = 1;
      h_x_ref(ib, n) = 1;
    }

    for (std::size_t i = 0; i < BlkSize; i++) {
      h_d(ib, i) = d_fact_ref[i];
    }

    for (std::size_t i = 0; i < BlkSize - 1; i++) {
      h_e(ib, i) = e_fact_ref[i];
    }
  }

  Kokkos::fence();

  Kokkos::deep_copy(ref, h_ref);
  Kokkos::deep_copy(x0, h_x0);
  Kokkos::deep_copy(x_ref, h_x_ref);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  create_banded_pds_matrix<View3DType, View3DType, KokkosBatched::Uplo::Upper>(
      ref, A, k, false);

  // pbtrs (Note, d and e is a diagonal and non-diagnoal part of factorized
  // matrix of A)
  Functor_BatchedSerialPtts2<DeviceType, View2DType, View2DType, View2DType,
                             AlgoTagType>(d, e, x0)
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
  EXPECT_TRUE(allclose<execution_space>(y0, x_ref, 1.e-5, 1.e-12));
}

template <typename DeviceType, typename ScalarType, typename LayoutType,
          typename AlgoTagType>
/// \brief Implementation details of batched pbtrs test
///
/// \param N [in] Batch size of RHS (banded matrix can also be batched matrix)
/// \param k [in] Number of superdiagonals or subdiagonals of matrix A
/// \param BlkSize [in] Block size of matrix A
void impl_test_batched_pttrs_analytical(const int N) {
  using View2DType = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;

  constexpr int BlkSize = 5, k = 1;
  View3DType A("A", N, BlkSize, BlkSize), ref("Ref", N, BlkSize, BlkSize);
  View2DType d("d", N, BlkSize), e("e", N, BlkSize - 1), x0("x0", N, BlkSize),
      x_ref("x_ref", N, BlkSize), y0("y0", N, BlkSize);  // Solutions

  auto h_A     = Kokkos::create_mirror_view(A);
  auto h_d     = Kokkos::create_mirror_view(d);
  auto h_e     = Kokkos::create_mirror_view(e);
  auto h_ref   = Kokkos::create_mirror_view(ref);
  auto h_x0    = Kokkos::create_mirror_view(x0);
  auto h_x_ref = Kokkos::create_mirror_view(x_ref);

  std::vector<ScalarType> d_fact_ref = {4.0, 3.75, 3.7333333, 3.732143,
                                        3.7320573};
  std::vector<ScalarType> e_fact_ref = {0.25, 0.26666667, 0.26785713,
                                        0.26794258};

  for (std::size_t ib = 0; ib < N; ib++) {
    for (std::size_t i = 0; i < BlkSize; i++) {
      for (std::size_t j = 0; j < BlkSize; j++) {
        h_ref(ib, i, j) = i == j ? 4.0 : 1.0;
      }
    }

    for (std::size_t n = 0; n < BlkSize; n++) {
      h_x0(ib, n)    = 1;
      h_x_ref(ib, n) = 1;
    }

    for (std::size_t i = 0; i < BlkSize; i++) {
      h_d(ib, i) = d_fact_ref[i];
    }

    for (std::size_t i = 0; i < BlkSize - 1; i++) {
      h_e(ib, i) = e_fact_ref[i];
    }
  }

  Kokkos::fence();

  Kokkos::deep_copy(ref, h_ref);
  Kokkos::deep_copy(x0, h_x0);
  Kokkos::deep_copy(x_ref, h_x_ref);
  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);

  create_banded_pds_matrix<View3DType, View3DType, KokkosBatched::Uplo::Upper>(
      ref, A, k, false);

  // pbtrs (Note, d and e is a diagonal and non-diagnoal part of factorized
  // matrix of A)
  Functor_BatchedSerialPttrs<DeviceType, View2DType, View2DType, View2DType,
                             AlgoTagType>(d, e, x0)
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
  EXPECT_TRUE(allclose<execution_space>(y0, x_ref, 1.e-5, 1.e-12));
}

}  // namespace Pttrs
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename AlgoTagType>
int test_batched_pttrs() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    Test::Pttrs::impl_test_batched_ptts2_analytical<DeviceType, ScalarType,
                                                    LayoutType, AlgoTagType>(1);
    Test::Pttrs::impl_test_batched_pttrs_analytical<DeviceType, ScalarType,
                                                    LayoutType, AlgoTagType>(1);
  }
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    Test::Pttrs::impl_test_batched_ptts2_analytical<DeviceType, ScalarType,
                                                    LayoutType, AlgoTagType>(1);
    Test::Pttrs::impl_test_batched_pttrs_analytical<DeviceType, ScalarType,
                                                    LayoutType, AlgoTagType>(1);
  }
#endif
  return 0;
}