#include "SelfEnergy.hpp"
#include "thermal_function.hpp"

namespace EffectivePotential {  

    // factorial helper
    double factorial(int n) {
        if (n < 0) {
            throw std::invalid_argument("Negative argument to factorial");
        }
        double result = 1.0;
        for (int i = 2; i <= n; ++i) {
            result *= i;
        }
        return result;
    }

    // helper function
    /*
        parameters:
            m2: mass squared
            T: temperature 
        return:
            bosons thermal function I0b(m2, T) = -1/(32 pi^2) * m2^2 * (log(m2/RG_scale^2) - 3/2) - T^4/pi^2 * J_B(m2/T^2) 
    */
    double I0b(double m2, double T) {
        double x_0 = m2/square(RG_scale);
        double zero_temp_part = -(m2 / (32 * square(M_PI))) * (square(RG_scale)*xlogx(x_0) - 1.5* m2);
        if (T <= std::numeric_limits<double>::min()) {
            return zero_temp_part;
        }
        else {
            double x_T = m2/square(T);
            double finite_temp_part = - (pow_4(T) / square(M_PI)) * J_B(x_T, 0);
            return zero_temp_part + finite_temp_part;
        }
    }

    // DTI0b,  T*T *dI0b/dT2
    double DTI0b(double m2, double T) {
        if (T <= std::numeric_limits<double>::min()) {
            return 0.;
        }
        else {
            double x_T = m2/square(T);
            double result = - (2*pow_4(T) / square(M_PI)) * J_B(x_T, 0) + (m2 * square(T)/ square(M_PI)) * J_B(x_T, 1);
            return result;
        }
    }

    // fermions thermal function I0f(m2, T) = -1/(32 pi^2) * m2^2 * (log(m2/RG_scale^2) - 3/2) + T^4/pi^2 * J_f(m2/T^2)
    double I0f(double m2, double T) {
        double x_0 = m2/square(RG_scale);
        double zero_temp_part = -(m2 / (32 * square(M_PI))) * (square(RG_scale)*xlogx(x_0) - 1.5* m2);
        if (T <= std::numeric_limits<double>::min()) {
            return zero_temp_part;
        }
        else {
            double x_T = m2/square(T);
            double finite_temp_part = (pow_4(T) / square(M_PI)) * J_F(x_T, 0);
            return zero_temp_part + finite_temp_part;
        }
    }

    // DTI0f,  T*T *dI0f/dT2
    double DTI0f(double m2, double T) {
        if (T <= std::numeric_limits<double>::min()) {
            return 0.;
        }
        else {
            double x_T = m2/square(T);
            double result = (2*pow_4(T) / square(M_PI)) * J_F(x_T, 0) - (m2 * square(T)/ square(M_PI)) * J_F(x_T, 1);
            return result;
        }
    }

    // UV term in dimensional regularization, which is proportional to epsilon, the coefficient of 1/epsilon divergence in DR. j is the order of derivative, corresponding to the Ijb/Ijf.
    double UV_term(double m2, double eps, int j) {
        switch (j)
        {
        case 0:
            return eps * square(m2) / (32 * square(M_PI));
        case 1:
            return -eps * 2 * m2 / (32 * square(M_PI));
        case 2:
            return eps * 2 / (32 * square(M_PI));
        default:
            return 0.0;
        }
    }

    // zeroT_term jth derivative
    double zeroT_term_deriv(double m2, int j) {
        double x_0 = m2/square(RG_scale);
        switch (j)
        {
            case 0:
                return -(m2 / (32 * square(M_PI))) * (square(RG_scale)*xlogx(x_0) - 1.5* m2);
            case 1:
                return (1 / (16 * square(M_PI))) * (m2 - square(RG_scale)*xlogx(x_0)); 
            case 2:{
                if (std::abs(x_0) < 1e-6) {
                    if (x_0 < 0.0) x_0 = std::min(-1e-6, x_0);
                    if (x_0 >= 0.0) x_0 = std::max(1e-6, x_0);
                }
                return - (xlogx(x_0)/x_0) / (16 * square(M_PI));
            }
            case 3: {
                if (std::abs(x_0) < 1e-6) {
                    if (x_0 < 0.0) x_0 = std::min(-1e-6, x_0);
                    if (x_0 >= 0.0) x_0 = std::max(1e-6, x_0);
                }
                return - 1 / (16 * square(M_PI) * m2);
            }
            default:
                throw std::invalid_argument("Invalid derivative order in zeroT_term_deriv");
        }
    }

    // Ijb, the jth derivative of I0b with respect to T^2. Note that the coefficient 1/((j-1)!) * (-1)^j is included here.
    double Ijb(double m2, double T, int j) {
        if (j == 0) {
            return I0b(m2, T);
        }
        else {
            double coeff = std::pow(-1, j) / factorial(j - 1);
            double zero_temp_part = zeroT_term_deriv(m2, j);
            if (T <= std::numeric_limits<double>::min()) {
                return coeff *zero_temp_part;
            }
            else {
                double x_T = m2 / square(T);
                double finite_temp_part = - (std::pow(T, 4 - 2*j) / square(M_PI)) * J_B(x_T, j);
                return coeff *(zero_temp_part + finite_temp_part);
            }
        }
    }

    double DTIjb(double m2, double T, int j) {
        if (j == 0) {
            return DTI0b(m2, T);
        }
        else {
            if (T <= std::numeric_limits<double>::min()) {
                return 0.;
            }
            else {
                double coeff = std::pow(-1, j) / factorial(j - 1);
                double x_T = m2 / square(T);
                double result = - (2-j) * (std::pow(T, 4-2*j)/ square(M_PI)) * J_B(x_T, j) + (m2 * std::pow(T, 2-2*j)/ square(M_PI)) * J_B(x_T, j+1);
                return coeff * result;
            }
        }
    }

    double Ijf(double m2, double T, int j) {
        if (j == 0) {
            return I0f(m2, T);
        }
        else {
            double coeff = std::pow(-1, j) / factorial(j - 1);
            double zero_temp_part = zeroT_term_deriv(m2, j);
            if (T <= std::numeric_limits<double>::min()) {
                return coeff *zero_temp_part;
            }
            else {
                double x_T = m2 / square(T);
                double finite_temp_part = (std::pow(T, 4 - 2*j) / square(M_PI)) * J_F(x_T, j);
                return coeff *(zero_temp_part + finite_temp_part);
            }
        }
    }

    double DTIjf(double m2, double T, int j) {
        if (j == 0) {
            return DTI0f(m2, T);
        }
        else {
            if (T <= std::numeric_limits<double>::min()) {
                return 0.;
            }
            else {
                double coeff = std::pow(-1, j) / factorial(j - 1);
                double x_T = m2 / square(T);
                double result = (2-j) * (std::pow(T, 4-2*j)/ square(M_PI)) * J_F(x_T, j) - (m2 * std::pow(T, 2-2*j)/ square(M_PI)) * J_F(x_T, j+1);
                return coeff * result;
            }
        }
    }

    // mixing self-energy
    double Imix(double m12, double m22, double T) {
        double deltaM2 = m22 - m12;
        if (std::abs(deltaM2) < 1e-4) { // if the mass difference is small, use the derivative to avoid numerical instability
            double arg = (m12 + m22) / 2; 
            return Ijb(arg, T, 2);
        }
        else {
            return ( Ijb(m12, T, 1) - Ijb(m22, T, 1) ) / deltaM2;
        }
    }

} // namespace EffectivePotential
