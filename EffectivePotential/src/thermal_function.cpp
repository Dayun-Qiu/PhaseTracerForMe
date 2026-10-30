// ====================================================================
// This file is part of PhaseTracer

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// ====================================================================

#include "thermal_function.hpp"
#include "thermal_function_tables.hpp"

#include <thermal_funcs.h>
#include <derivatives.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>

#include <memory>
#include <stdexcept>
#include <cmath>
#include <string>

namespace EffectivePotential {

// ===================== OLD ALGIB CODE (commented out) =====================

alglib::spline1dinterpolant make_cubic_spline(alglib::real_1d_array x, alglib::real_1d_array y) {
  alglib::spline1dinterpolant spline;
  alglib::spline1dbuildcubic(x, y, spline);
  return spline;
}

double J_B(double x) {
  constexpr double min_x = -3.72402637;
  constexpr double max_x = 1.41e3;

  static const auto spline = make_cubic_spline(J_B_X_DATA, J_B_Y_DATA);
  static const double spline_at_min_x = alglib::spline1dcalc(spline, min_x);

  if (x < min_x) {
    return spline_at_min_x;
  } else if (x > max_x) {
    return 0.;
  } else {
    return alglib::spline1dcalc(spline, x);
  }
}

double J_F(double x) {
  constexpr double min_x = -6.82200203;
  constexpr double max_x = 1.35e3;

  static const auto spline = make_cubic_spline(J_F_X_DATA, J_F_Y_DATA);
  static const double spline_at_min_x = alglib::spline1dcalc(spline, min_x);

  if (x < min_x) {
    return spline_at_min_x;
  } else if (x > max_x) {
    return 0.;
  } else {
    return alglib::spline1dcalc(spline, x);
  }
}
//
double J_B_algb(double x, int deriv_order) {
    constexpr double min_x = -3.72402637;
    constexpr double max_x = 1.41e3;
    static const auto spline = make_cubic_spline(J_B_X_DATA, J_B_Y_DATA);
    static const double spline_at_min_x_val = alglib::spline1dcalc(spline, min_x);
    if (x < min_x) {
        if (deriv_order == 0) return spline_at_min_x_val;
        else return 0.0;
    }
    else if (x > max_x) {
        return 0.0;
    }
    else {
        double val, der1, der2;
        alglib::spline1ddiff(spline, x, val, der1, der2);
        switch (deriv_order) {
            case 0: return val;
            case 1: return der1;
            case 2: return der2;
            default: throw std::invalid_argument("Invalid derivative order");
        }
    }
}
//
double J_F_algb(double x, int deriv_order) {
    constexpr double min_x = -6.82200203;
    constexpr double max_x = 1.35e3;
    static const auto spline = make_cubic_spline(J_F_X_DATA, J_F_Y_DATA);
    static const double spline_at_min_x_val = alglib::spline1dcalc(spline, min_x);
    if (x < min_x) {
        if (deriv_order == 0) return spline_at_min_x_val;
        else return 0.0;
    }
    else if (x > max_x) {
        return 0.0;
    }
    else {
        double val, der1, der2;
        alglib::spline1ddiff(spline, x, val, der1, der2);
        switch (deriv_order) {
            case 0: return val;
            case 1: return der1;
            case 2: return der2;
            default: throw std::invalid_argument("Invalid derivative order");
        }
    }
}

double J_B0(double x, int deriv_order) {
    constexpr double min_x = -3.72402637;
    constexpr double max_x = 1.41e3;
    static const auto spline = make_cubic_spline(J_B0_X_DATA, J_B0_Y_DATA);
    static const double spline_at_min_x_val = alglib::spline1dcalc(spline, min_x);
    if (x < min_x) {
        if (deriv_order == 0) return spline_at_min_x_val;
        else return 0.0;
    }
    else if (x > max_x) {
        switch (deriv_order) {
            case 0: return M_PI * std::pow(x,1.5)/6.;
            case 1: return M_PI * std::pow(x,0.5)/4.;
            case 2: return M_PI * std::pow(x,-0.5)/8.;
            default: throw std::invalid_argument("Invalid derivative order");
        } 
    }
    else {
        double val, der1, der2;
        alglib::spline1ddiff(spline, x, val, der1, der2);
        switch (deriv_order) {
            case 0: return val;
            case 1: return der1;
            case 2: return der2;
            default: throw std::invalid_argument("Invalid derivative order");
        }
    }
}



// ============================================================
// GSL cubic B-spline
// ============================================================

struct GSLSpline {

    gsl_bspline_workspace* work = nullptr;
    gsl_vector* coeffs = nullptr;

    double x_min = 0.0;
    double x_max = 0.0;

    size_t ncontrol = 0;

    // Pre-computed values at x_min for derivative orders 0..3
    // This avoids re-evaluating the spline on every boundary call
    double boundary_val[4] = {0.0, 0.0, 0.0, 0.0};

    GSLSpline() = default;

    ~GSLSpline()
    {
        if (coeffs)
            gsl_vector_free(coeffs);

        if (work)
            gsl_bspline_free(work);
    }

    GSLSpline(const GSLSpline&) = delete;
    GSLSpline& operator=(const GSLSpline&) = delete;

    GSLSpline(GSLSpline&& other) noexcept
        : work(other.work),
          coeffs(other.coeffs),
          x_min(other.x_min),
          x_max(other.x_max),
          ncontrol(other.ncontrol)
    {
        for (int i = 0; i < 4; ++i)
            boundary_val[i] = other.boundary_val[i];
        other.work = nullptr;
        other.coeffs = nullptr;
    }

    GSLSpline& operator=(GSLSpline&& other) noexcept
    {
        if (this != &other) {

            if (coeffs)
                gsl_vector_free(coeffs);

            if (work)
                gsl_bspline_free(work);

            work = other.work;
            coeffs = other.coeffs;

            x_min = other.x_min;
            x_max = other.x_max;
            ncontrol = other.ncontrol;

            for (int i = 0; i < 4; ++i)
                boundary_val[i] = other.boundary_val[i];

            other.work = nullptr;
            other.coeffs = nullptr;
        }

        return *this;
    }
};


// ============================================================
// Construct cubic B-spline interpolation
//
// Equivalent in purpose to:
//
//     scipy.interpolate.splrep(x, y, k=3, s=0)
//
// ============================================================

static std::unique_ptr<GSLSpline>
make_gsl_spline(
    const alglib::real_1d_array& x_data,
    const alglib::real_1d_array& y_data)
{
    const size_t n = x_data.length();

    if (n != y_data.length()) {
        throw std::runtime_error(
            "make_gsl_spline: x/y size mismatch"
        );
    }

    if (n < 4) {
        throw std::runtime_error(
            "make_gsl_spline: need at least 4 points"
        );
    }

    // cubic
    constexpr size_t k = 4;

    auto sp = std::make_unique<GSLSpline>();

    sp->x_min = x_data[0];
    sp->x_max = x_data[n - 1];

    // --------------------------------------------------------
    // Check data
    // --------------------------------------------------------

    for (size_t i = 0; i < n; ++i) {

        if (!std::isfinite(x_data[i]) ||
            !std::isfinite(y_data[i])) {

            throw std::runtime_error(
                "make_gsl_spline: "
                "NaN/Inf in input data"
            );
        }

        if (i > 0 &&
            x_data[i] <= x_data[i - 1]) {

            throw std::runtime_error(
                "make_gsl_spline: "
                "x_data must be strictly increasing"
            );
        }
    }

    // --------------------------------------------------------
    // Allocate B-spline
    // --------------------------------------------------------

    sp->work =
        gsl_bspline_alloc_ncontrol(k, n);

    if (!sp->work) {
        throw std::runtime_error(
            "gsl_bspline_alloc_ncontrol failed"
        );
    }

    sp->ncontrol =
        gsl_bspline_ncontrol(sp->work);

    sp->coeffs =
        gsl_vector_alloc(sp->ncontrol);

    if (!sp->coeffs) {
        throw std::runtime_error(
            "gsl_vector_alloc failed"
        );
    }

    // --------------------------------------------------------
    // x/y views
    // --------------------------------------------------------

    gsl_vector_const_view xv =
        gsl_vector_const_view_array(
            x_data.getcontent(),
            n
        );

    gsl_vector_const_view yv =
        gsl_vector_const_view_array(
            y_data.getcontent(),
            n
        );

    // --------------------------------------------------------
    // Interpolation knots
    // --------------------------------------------------------

    const int status_interp =
        gsl_bspline_init_interp(
            &xv.vector,
            sp->work
        );

    if (status_interp != GSL_SUCCESS) {

        throw std::runtime_error(
            std::string(
                "gsl_bspline_init_interp failed: "
            )
            + gsl_strerror(status_interp)
        );
    }

    // --------------------------------------------------------
    // Build interpolation matrix
    //
    // A_ij = B_j(x_i)
    // --------------------------------------------------------

    gsl_matrix* A =
        gsl_matrix_calloc(
            n,
            sp->ncontrol
        );

    if (!A) {
        throw std::runtime_error(
            "failed to allocate matrix A"
        );
    }

    gsl_vector* B =
        gsl_vector_alloc(k);

    if (!B) {

        gsl_matrix_free(A);

        throw std::runtime_error(
            "failed to allocate B"
        );
    }

    for (size_t i = 0; i < n; ++i) {

        size_t istart = 0;

        const int status =
            gsl_bspline_basis(
                x_data[i],
                B,
                &istart,
                sp->work
            );

        if (status != GSL_SUCCESS) {

            gsl_vector_free(B);
            gsl_matrix_free(A);

            throw std::runtime_error(
                std::string(
                    "gsl_bspline_basis failed: "
                )
                + gsl_strerror(status)
            );
        }

        for (size_t j = 0; j < k; ++j) {

            const size_t col =
                istart + j;

            if (col < sp->ncontrol) {

                gsl_matrix_set(
                    A,
                    i,
                    col,
                    gsl_vector_get(B, j)
                );
            }
        }
    }

    gsl_vector_free(B);

    // --------------------------------------------------------
    // QR solve
    //
    // A * coeffs = y
    // --------------------------------------------------------

    gsl_vector* tau =
        gsl_vector_alloc(
            std::min(n, sp->ncontrol)
        );

    if (!tau) {

        gsl_matrix_free(A);

        throw std::runtime_error(
            "failed to allocate tau"
        );
    }

    int status =
        gsl_linalg_QR_decomp(
            A,
            tau
        );

    if (status != GSL_SUCCESS) {

        gsl_vector_free(tau);
        gsl_matrix_free(A);

        throw std::runtime_error(
            std::string(
                "gsl_linalg_QR_decomp failed: "
            )
            + gsl_strerror(status)
        );
    }

    gsl_vector* residual =
        gsl_vector_alloc(n);

    if (!residual) {

        gsl_vector_free(tau);
        gsl_matrix_free(A);

        throw std::runtime_error(
            "failed to allocate residual"
        );
    }

    status =
        gsl_linalg_QR_lssolve(
            A,
            tau,
            &yv.vector,
            sp->coeffs,
            residual
        );

    if (status != GSL_SUCCESS) {

        gsl_vector_free(residual);
        gsl_vector_free(tau);
        gsl_matrix_free(A);

        throw std::runtime_error(
            std::string(
                "gsl_linalg_QR_lssolve failed: "
            )
            + gsl_strerror(status)
        );
    }

    // --------------------------------------------------------
    // Check coefficients
    // --------------------------------------------------------

    for (size_t i = 0;
         i < sp->ncontrol;
         ++i)
    {
        const double c =
            gsl_vector_get(
                sp->coeffs,
                i
            );

        if (!std::isfinite(c)) {

            gsl_vector_free(residual);
            gsl_vector_free(tau);
            gsl_matrix_free(A);

            throw std::runtime_error(
                "spline coefficient is NaN/Inf"
            );
        }
    }

    gsl_vector_free(residual);
    gsl_vector_free(tau);
    gsl_matrix_free(A);

    // --------------------------------------------------------
    // Pre-compute boundary values at x_min
    //
    // For x < x_min, the spline returns the value at x_min
    // (constant extrapolation). Computing these once avoids
    // re-evaluating the spline on every boundary call.
    // --------------------------------------------------------

    for (int n = 0; n < 4; ++n) {

        double result = 0.0;

        const int status =
            gsl_bspline_calc_deriv(
                sp->x_min,
                sp->coeffs,
                static_cast<size_t>(n),
                &result,
                sp->work
            );

        if (status != GSL_SUCCESS) {

            throw std::runtime_error(
                std::string(
                    "gsl_bspline_calc_deriv failed during "
                    "boundary pre-computation: "
                )
                + gsl_strerror(status)
            );
        }

        sp->boundary_val[n] = result;
    }

    return sp;
}


// ============================================================
// Evaluate spline / derivatives (HOT PATH)
//
// n = 0 -> J_B
// n = 1 -> J_B'
// n = 2 -> J_B''
// n = 3 -> J_B'''
//
// Inlined aggressively and stripped of runtime checks for
// maximum performance in the hot loop.
// ============================================================

__attribute__((always_inline))
static inline double
gsl_spline_eval(
    const GSLSpline& sp,
    double x,
    int n)
{
    // Cubic spline: fourth and higher derivatives vanish
    if (n >= 4)
        return 0.0;

    double result = 0.0;

    // In practice gsl_bspline_calc_deriv never fails for
    // valid inputs, so we skip the status check to avoid
    // branch prediction overhead in the hot loop.
    gsl_bspline_calc_deriv(
        x,
        sp.coeffs,
        static_cast<size_t>(n),
        &result,
        sp.work
    );

    return result;
}


// ============================================================
// J_B
//
// Optimized: boundary values pre-computed at construction time
// ============================================================

double J_B(
    double x,
    int n)
{

    // if (std::abs(x) < 0.1) {
    //     switch (n) {
    //         case 0:
    //             return J_B_taylor(x);
    //         case 1:
    //             return D1_J_B_taylor(x);
    //         case 2:
    //             return D2_J_B_taylor(x);
    //         case 3:
    //             return D3_J_B_taylor(x);
    //         default:
    //             throw std::invalid_argument("n must be 0, 1, 2, or 3");
    //     }
    // }

    if (n <= 2) {
        return J_B_algb(x, n);
    }
    static const auto spline =
        make_gsl_spline(
            J_B_X_DATA,
            J_B_Y_DATA
        );

    // x > xbmax -> always 0
    if (x > 1.41e3)
        return 0.0;

    // x < xbmin -> return pre-computed boundary value
    if (x < -3.72402637) {
        if (n >= 4)
            return 0.0;
        return spline->boundary_val[n];
    }

    return gsl_spline_eval(
        *spline,
        x,
        n
    );
}


// ============================================================
// J_F
//
// Optimized: boundary values pre-computed at construction time
// ============================================================

double J_F(
    double x,
    int n)
{
    if (n <= 2) {
        return J_F_algb(x, n);
    }
    static const auto spline =
        make_gsl_spline(
            J_F_X_DATA,
            J_F_Y_DATA
        );

    // x > xfmax -> always 0
    if (x > 1.35e3)
        return 0.0;

    // x < xfmin -> return pre-computed boundary value
    if (x < -6.82200203) {
        if (n >= 4)
            return 0.0;
        return spline->boundary_val[n];
    }

    return gsl_spline_eval(
        *spline,
        x,
        n
    );
}



} // namespace EffectivePotential
