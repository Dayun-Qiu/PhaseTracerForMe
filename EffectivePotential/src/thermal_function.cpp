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

namespace EffectivePotential {

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


// Add overloaded functions to compute first or second derivatives.
double J_B(double x, int deriv_order) {
    constexpr double min_x = -3.72402637;
    constexpr double max_x = 1.41e3;

    static const auto spline = make_cubic_spline(J_B_X_DATA, J_B_Y_DATA);
    
    static const double spline_at_min_x_val = alglib::spline1dcalc(spline, min_x);
    static const double spline_at_min_x_der1 = 0.0;  
    static const double spline_at_min_x_der2 = 0.0;   

    if (x < min_x) {
        if (deriv_order == 0) return spline_at_min_x_val;
        else return 0.0;   
    } 
    else if (x > max_x) {
        return 0.0;
    } 
    else {
        double val, der1, der2;
        // spline1ddiff computes value, first, and second derivatives in one call
        alglib::spline1ddiff(spline, x, val, der1, der2);
        switch (deriv_order) {
            case 0: return val;
            case 1: return der1;
            case 2: return der2;
            default: throw std::invalid_argument("Invalid derivative order");
        }
    }
}


double J_F(double x, int deriv_order) {
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


} // namespace EffectivePotential
