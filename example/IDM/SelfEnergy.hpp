#ifndef SelfEnergy_HPP_
#define SelfEnergy_HPP_

#include <cmath>
#include "thermal_function.hpp"
#include "pow.hpp"
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <functional>
#include <array>

#define RG_scale 91.1876

namespace EffectivePotential {  

    double xlogx(double);

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
                if (std::abs(x_0) <= std::numeric_limits<double>::min()) {
                    throw std::runtime_error("x_0 is too small in zeroT_term_deriv, which may cause infared divergence.");
                }
                return - xlogx(x_0)/x_0 / (16 * square(M_PI));
            }
            case 3: {
                if (std::abs(x_0) <= std::numeric_limits<double>::min()) {
                    throw std::runtime_error("x_0 is too small in zeroT_term_deriv, which may cause infared divergence.");
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
            double coeff = pow(-1, j) / factorial(j - 1);
            double zero_temp_part = zeroT_term_deriv(m2, j);
            if (T <= std::numeric_limits<double>::min()) {
                return coeff *zero_temp_part;
            }
            else {
                double x_T = m2 / square(T);
                double finite_temp_part = - (pow(T, 4 - 2*j) / square(M_PI)) * J_B(x_T, j);
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
                double coeff = pow(-1, j) / factorial(j - 1);
                double x_T = m2 / square(T);
                double result = - (2-j) * (pow(T, 4-2*j)/ square(M_PI)) * J_B(x_T, j) + (m2 * pow(T, 2-2*j)/ square(M_PI)) * J_B(x_T, j+1);
                return coeff * result;
            }
        }
    }

    double Ijf(double m2, double T, int j) {
        if (j == 0) {
            return I0f(m2, T);
        }
        else {
            double coeff = pow(-1, j) / factorial(j - 1);
            double zero_temp_part = zeroT_term_deriv(m2, j);
            if (T <= std::numeric_limits<double>::min()) {
                return coeff *zero_temp_part;
            }
            else {
                double x_T = m2 / square(T);
                double finite_temp_part = (pow(T, 4 - 2*j) / square(M_PI)) * J_F(x_T, j);
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
                double coeff = pow(-1, j) / factorial(j - 1);
                double x_T = m2 / square(T);
                double result = (2-j) * (pow(T, 4-2*j)/ square(M_PI)) * J_F(x_T, j) - (m2 * pow(T, 2-2*j)/ square(M_PI)) * J_F(x_T, j+1);
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


    class SelfEnergy {
        private:
            // constants
            const double v0 = 246.22; // GeV
            const double mh = 125.10; // GeV
            const double mt = 172.76; // GeV
            const double yt = sqrt(2.0)*mt/v0;
            const double g = 0.65175;
            const double gp = 0.35742;
            const double gstar = 110.75;
            // input parameters
            double lam2 = 0;
            double lamL = 0;
            double mA = 0;
            double mH = 0;
            double mHpm = 0;
            //others
            double lam1 = 0;
            double mu1sq = 0;
            double mu2sq = 0;
            double lamm = 0;
            double lamp = 0;
            double lam3 = 0;
            double lam4 = 0;
            double lam5 = 0;

            double delta_lam1 = 0;
            double delta_mu1sq = 0;
            double delta_mu2sq = 0;
            double delta_lamm = 0;
            double delta_lamp = 0;
            double delta_lam3 = 0;



        public:
            SelfEnergy(double lam2_, double lamL_, double mA_, double mH_, double mHpm_){
                lam2 = lam2_;
                lamL = lamL_;
                mA = mA_;
                mH = mH_;
                mHpm = mHpm_;
                lam1 = square(mh/v0);
                mu1sq = - 0.5 * square(mh);

                lam4 = (square(mA) + square(mH) - 2 * square(mHpm))/ square(v0);
                lam5 = (square(mH) - square(mA))/ square(v0);
                lam3 = 2 * lamL - lam4 - lam5;
                mu2sq = 0.5* (2 * square(mHpm) - lam3 * square(v0));
                lamp = lam3 + lam4 + lam5;
                lamm = lam3 + lam4 - lam5;
            }

            // Higgs three-points vertex self-energy functions
    
            // Top quark contribution: Pihtt
            double Pihtt(double mt2, double T) {
                return 6.0 * square(yt) * (-Ijf(mt2, T, 1) + 2.0 * mt2 * Ijf(mt2, T, 2));
            }

            double DPihtt_Dm2(double mt2) {
                return 6.0 * square(yt) * (3.0 * zeroT_term_deriv(mt2, 2) + 2.0 * mt2 * zeroT_term_deriv(mt2, 3));
            }

            // h particle contribution: Pihhh
            double Pihhh(double mh2, double T, double phi) {
                return -4.5 * square(lam1) * square(phi) * Ijb(mh2, T, 2);
            }
            
            double DPihhh_Dm2(double mh2) {
                return -4.5 * square(lam1) * square(v0) * zeroT_term_deriv(mh2, 3);
            }

            // A particle contribution: PihAA
            double PihAA(double mA2, double T, double phi) {
                return -0.5 * square(lamm) * square(phi) * Ijb(mA2, T, 2);
            }

            double DPihAA_Dm2(double mA2) {
                return -0.5 * square(lamm) * square(v0) * zeroT_term_deriv(mA2, 3);
            }

            // H particle contribution: PihHH
            double PihHH(double mH2, double T, double phi) {
                return -0.5 * square(lamp) * square(phi) * Ijb(mH2, T, 2);
            }
            
            double DPihHH_Dm2(double mH2) {
                return -0.5 * square(lamp) * square(v0) * zeroT_term_deriv(mH2, 3);
            }

            // H± particle contribution: PihHpHm
            double PihHpHm(double mHpm2, double T, double phi) {
                return -square(lam3) * square(phi) * Ijb(mHpm2, T, 2);
            }
            
            double DPihHpHm_Dm2(double mHpm2) {
                return -square(lam3) * square(v0) * zeroT_term_deriv(mHpm2, 3);
            }

            // G particle contribution: PihGG
            double PihGG(double mG2, double T, double phi) {
                return -1.5 * square(lam1) * square(phi) * Ijb(mG2, T, 2);
            }
            
            double DPihGG_Dm2(double mG2) {
                return -1.5 * square(lam1) * square(v0) * zeroT_term_deriv(mG2, 3);
            }

            // Z boson contribution: PihZZ
            double PihZZ(double mZ2L, double mZ2T, double T, double swL, double swT, double phi) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double L_term = -0.125 * pow(g * cwL + gp * swL, 4) * square(phi) * Ijb(mZ2L, T, 2);
                double T_term = -0.125 * pow(g * cwT + gp * swT, 4) * square(phi) * (2.0 * Ijb(mZ2T, T, 2) + UV_term(mZ2T, -2.0, 2));
                return L_term + T_term;
            }

            double DPihZZ_Dm2(double mZ2, double sw) {
                double cw = std::sqrt(1.0 - square(sw));
                return -0.125 * pow(g * cw + gp * sw, 4) * square(v0) * (3.0 * zeroT_term_deriv(mZ2, 3) + UV_term(mZ2, -2.0, 3));
            }

            // Photon (gamma) contribution: Pihgaga
            double Pihgaga(double mga2L, double mga2T, double T, double swL, double swT, double phi) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double L_term = -0.125 * pow(g * swL - gp * cwL, 4) * square(phi) * Ijb(mga2L, T, 2);
                double T_term = -0.125 * pow(g * swT - gp * cwT, 4) * square(phi) * (2.0 * Ijb(mga2T, T, 2) + UV_term(mga2T, -2.0, 2));
                return L_term + T_term;
            }

            // gamma-Z mixing contribution: PihgaZ
            double PihgaZ(double mga2L, double mga2T, double mZ2L, double mZ2T, double T, double swL, double swT, double phi) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double L_term = -0.125 * square(g * swL - gp * cwL) * square(g * cwL + gp * swL) * square(phi) * Imix(mga2L, mZ2L, T);
                
                double deltaM2 = mZ2T - mga2T;
                double T_term;
                if (std::abs(deltaM2) < 1e-4) { // if the mass difference is small, use the derivative to avoid numerical instability
                    double arg = (mga2T + mZ2T) / 2.0;
                    T_term = -0.125 * square(g * swT - gp * cwT) * square(g * cwT + gp * swT) * square(phi) * (2.0 * Ijb(arg, T, 2) + UV_term(arg, -2.0, 2));
                } else {
                    T_term = -0.125 * square(g * swT - gp * cwT) * square(g * cwT + gp * swT) * square(phi) * (2.0 * (Ijb(mga2T, T, 1) - Ijb(mZ2T, T, 1)) / deltaM2 +  (UV_term(mga2T, -2.0, 1) - UV_term(mZ2T, -2.0, 1)) / deltaM2);
                }
                return L_term + T_term;
            }

            // W boson contribution: PihWW
            double PihWW(double mW2L, double mW2T, double T, double phi) {
                return -0.25 * pow(g, 4) * square(phi) * (Ijb(mW2L, T, 2) + 2.0 * Ijb(mW2T, T, 2) + UV_term(mW2T, -2.0, 2));
            }
            
            double DPihWW_Dm2(double mW2) {
                return -0.25 * pow(g, 4) * square(v0) * (3.0 * zeroT_term_deriv(mW2, 3) + UV_term(mW2, -2.0, 3));
            }

            // ======================== Higgs four-points vertex self-energy =========================

            // h particle contribution: Pihh
            double Pihh(double mh2, double T) {
                return 1.5 * lam1 * Ijb(mh2, T, 1);
            }

            // A particle contribution: PihA
            double PihA(double mA2, double T) {
                return 0.5 * lamm * Ijb(mA2, T, 1);
            }

            // H particle contribution: PihH
            double PihH(double mH2, double T) {
                return 0.5 * lamp * Ijb(mH2, T, 1);
            }

            // H± particle contribution: PihHpm
            double PihHpm(double mHpm2, double T) {
                return lam3 * Ijb(mHpm2, T, 1);
            }

            // G particle contribution: PihG
            double PihG(double mG2, double T) {
                return 1.5 * lam1 * Ijb(mG2, T, 1);
            }

            // Z boson contribution: PihZ
            double PihZ(double mZ2L, double mZ2T, double T, double swL, double swT) {
                // Use std::min/std::max for C++11/14 compatibility if std::clamp is not available, 
                // otherwise keep std::clamp if C++17 is enabled.
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double L_term = 0.25 * square(g * cwL + gp * swL) * Ijb(mZ2L, T, 1);
                double T_term = 0.25 * square(g * cwT + gp * swT) * (2.0 * Ijb(mZ2T, T, 1) + UV_term(mZ2T, -2.0, 1));
                
                return L_term + T_term;
            }

            // Photon (gamma) contribution: Pihga
            double Pihga(double mga2L, double mga2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double L_term = 0.25 * square(g * swL - gp * cwL) * Ijb(mga2L, T, 1);
                double T_term = 0.25 * square(g * swT - gp * cwT) * (2.0 * Ijb(mga2T, T, 1) + UV_term(mga2T, -2.0, 1));
                
                return L_term + T_term;
            }

            // W boson contribution: PihW
            double PihW(double mW2L, double mW2T, double T) {
                return 0.5 * square(g) * (Ijb(mW2L, T, 1) + 2.0 * Ijb(mW2T, T, 1) + UV_term(mW2T, -2.0, 1));
            }


            // ======================== Momentum Dependent Self-Energies (Boost Version) =========================

            double Pih3Pdependent(double T, double p2, const std::array<double, 11>& MB2, double mt2, double phi, double swL, double swT) {
                using boost::math::quadrature::gauss_kronrod;

                // Define the integrand using a Lambda expression
                auto integrand = [&](double x) -> double {
                    double factor = x * (1.0 - x) * p2;
                    
                    double mt2star     = mt2 - factor;
                    double mh2star     = MB2[0] - factor;
                    double mG2star     = MB2[1] - factor;
                    double mA2star     = MB2[2] - factor;
                    double mH2star     = MB2[3] - factor;
                    double mHpm2star   = MB2[4] - factor;
                    double mW2Lstar    = MB2[5] - factor;
                    double mZ2Lstar    = MB2[6] - factor;
                    double mga2Lstar   = MB2[7] - factor;
                    double mW2Tstar    = MB2[8] - factor;
                    double mZ2Tstar    = MB2[9] - factor;
                    double mga2Tstar   = MB2[10] - factor;

                    double Pisum = 0.0;
                    Pisum += Pihtt(mt2star, T);
                    Pisum += Pihhh(mh2star, T, phi);
                    Pisum += PihAA(mA2star, T, phi);
                    Pisum += PihHH(mH2star, T, phi);
                    Pisum += PihHpHm(mHpm2star, T, phi);
                    Pisum += PihGG(mG2star, T, phi);
                    Pisum += PihZZ(mZ2Lstar, mZ2Tstar, T, swL, swT, phi);
                    Pisum += Pihgaga(mga2Lstar, mga2Tstar, T, swL, swT, phi);
                    Pisum += PihgaZ(mga2Lstar, mga2Tstar, mZ2Lstar, mZ2Tstar, T, swL, swT, phi);
                    Pisum += PihWW(mW2Lstar, mW2Tstar, T, phi);

                    return Pisum;
                };

                // Perform integration over [0, 1]
                // 1e-4 is the relative tolerance
                double result = gauss_kronrod<double, 15>::integrate(integrand, 0.0, 1.0, 20, 1e-4);
                return result;
            }

            double DPih3_Dp2(double p2, const std::array<double, 11>& MB2, double mt2, double sw) {
                using boost::math::quadrature::gauss_kronrod;

                auto integrand = [&](double x) -> double {
                    double factor = x * (1.0 - x) * p2;

                    double mt2star   = mt2 - factor;
                    double mh2star   = MB2[0] - factor;
                    double mG2star   = MB2[1] - factor;
                    double mA2star   = MB2[2] - factor;
                    double mH2star   = MB2[3] - factor;
                    double mHpm2star = MB2[4] - factor;
                    double mW2star   = MB2[5] - factor; 
                    double mZ2star   = MB2[6] - factor; 

                    double Pisum = 0.0;
                    Pisum += DPihtt_Dm2(mt2star);
                    Pisum += DPihhh_Dm2(mh2star);
                    Pisum += DPihAA_Dm2(mA2star);
                    Pisum += DPihHH_Dm2(mH2star);
                    Pisum += DPihHpHm_Dm2(mHpm2star);
                    Pisum += DPihGG_Dm2(mG2star);
                    Pisum += DPihZZ_Dm2(mZ2star, sw);
                    Pisum += DPihWW_Dm2(mW2star);

                    return -x * (1.0 - x) * Pisum;
                };

                double result = gauss_kronrod<double, 15>::integrate(integrand, 0.0, 1.0, 20, 1e-4);
                return result;
            }

            // Vector overloads for external interface compatibility
            double Pih3Pdependent(double T, double p2, const std::vector<double>& MB2_vec, double mt2, double phi, double swL, double swT) {
                if (MB2_vec.size() != 11) throw std::invalid_argument("MB2 size must be 11");
                std::array<double, 11> MB2_arr;
                std::copy(MB2_vec.begin(), MB2_vec.end(), MB2_arr.begin());
                return Pih3Pdependent(T, p2, MB2_arr, mt2, phi, swL, swT);
            }

            double DPih3_Dp2(double p2, const std::vector<double>& MB2_vec, double mt2, double sw) {
                if (MB2_vec.size() != 11) throw std::invalid_argument("MB2 size must be 11");
                std::array<double, 11> MB2_arr;
                std::copy(MB2_vec.begin(), MB2_vec.end(), MB2_arr.begin());
                return DPih3_Dp2(p2, MB2_arr, mt2, sw);
            }

            double Pih3(double T, const std::vector<double>& MB2, double mt2, double phi, double swL, double swT) {
                double mh2     = MB2[0];
                double mG2     = MB2[1];
                double mA2     = MB2[2];
                double mH2     = MB2[3];
                double mHpm2   = MB2[4];
                double mW2L    = MB2[5];
                double mZ2L    = MB2[6];
                double mga2L   = MB2[7];
                double mW2T    = MB2[8];
                double mZ2T    = MB2[9];
                double mga2T   = MB2[10];

                double Pisum = 0.0;   
                Pisum += Pihtt(mt2, T);
                // Pisum += Pihhh(mh2, T, phi); 
                // Pisum += PihGG(mG2, T, phi);
                Pisum += PihAA(mA2, T, phi);
                Pisum += PihHH(mH2, T, phi);
                Pisum += PihHpHm(mHpm2, T, phi);
                Pisum += PihWW(mW2L, mW2T, T, phi);
                Pisum += PihZZ(mZ2L, mZ2T, T, swL, swT, phi);
                Pisum += Pihgaga(mga2L, mga2T, T, swL, swT, phi);
                Pisum += PihgaZ(mga2L, mga2T, mZ2L, mZ2T, T, swL, swT, phi);

                return Pisum;
            }

            double Pih4(double T, const std::vector<double>& MB2, double swL, double swT) {
                double mh2     = MB2[0];
                double mG2     = MB2[1];
                double mA2     = MB2[2];
                double mH2     = MB2[3];
                double mHpm2   = MB2[4];
                double mW2L    = MB2[5];
                double mZ2L    = MB2[6];
                double mga2L   = MB2[7];
                double mW2T    = MB2[8];
                double mZ2T    = MB2[9];
                double mga2T   = MB2[10];

                double Pisum = 0.0;
                Pisum += Pihh(mh2, T);
                Pisum += PihA(mA2, T);
                Pisum += PihH(mH2, T);
                Pisum += PihHpm(mHpm2, T);
                Pisum += PihG(mG2, T);
                Pisum += PihZ(mZ2L, mZ2T, T, swL, swT);
                Pisum += Pihga(mga2L, mga2T, T, swL, swT);
                Pisum += PihW(mW2L, mW2T, T);

                return Pisum;
            }

            // ======================== Goldstone three-points vertex self-energy =========================
            
            double PiGtt(double mt2, double T) {
                return -6.0 * square(yt) * Ijf(mt2, T, 1);
            }

            double PiGhG(double mh2, double mG2, double T, double phi) {
                return -square(lam1) * square(phi) * Imix(mh2, mG2, T);
            }

            double PiGAH(double mA2, double mH2, double T, double phi) {
                return -square(lam5) * square(phi) * Imix(mA2, mH2, T);
            }

            // ======================== Goldstone four-points vertex self-energy =========================

            double PiGh(double mh2, double T) {
                return 0.5 * lam1 * Ijb(mh2, T, 1);
            }

            double PiGG(double mG2, double T) {
                return 2.5 * lam1 * Ijb(mG2, T, 1);
            }

            double PiGA(double mA2, double T) {
                return 0.5 * lamp * Ijb(mA2, T, 1);
            }

            double PiGH(double mH2, double T) {
                return 0.5 * lamm * Ijb(mH2, T, 1);
            }

            double PiGHpm(double mHpm2, double T) {
                return lam3 * Ijb(mHpm2, T, 1);
            }

            double PiGZ(double mZ2L, double mZ2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double gZ_L = g * cwL + gp * swL;
                double gZ_T = g * cwT + gp * swT;
                
                double L_term = 0.25 * square(gZ_L) * Ijb(mZ2L, T, 1);
                double T_term = 0.25 * square(gZ_T) * (2.0 * Ijb(mZ2T, T, 1) + UV_term(mZ2T, -2, 1));
                
                return L_term + T_term;
            }

            double PiGga(double mga2L, double mga2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double gga_L = g * swL - gp * cwL;
                double gga_T = g * swT - gp * cwT;
                
                double L_term = 0.25 * square(gga_L) * Ijb(mga2L, T, 1);
                double T_term = 0.25 * square(gga_T) * (2.0 * Ijb(mga2T, T, 1) + UV_term(mga2T, -2, 1));
                
                return L_term + T_term;
            }

            double PiGW(double mW2L, double mW2T, double T) {
                return 0.5 * square(g) * (Ijb(mW2L, T, 1) + 2.0 * Ijb(mW2T, T, 1) + UV_term(mW2T, -2, 1));
            }
            
            double PiG3(double T, const std::vector<double>& MB2, double mt2, double phi) {
                double mh2 = MB2[0]; 
                double mG2 = MB2[1]; 
                double mA2 = MB2[2]; 
                double mH2 = MB2[3]; 
                
                double Pisum = 0.0;
                Pisum += PiGtt(mt2, T);
                Pisum += PiGhG(mh2, mG2, T, phi);
                Pisum += PiGAH(mA2, mH2, T, phi);
                
                return Pisum;
            }

            double PiG4(double T, const std::vector<double>& MB2, double swL, double swT) {
                double mh2     = MB2[0]; 
                double mG2     = MB2[1]; 
                double mA2     = MB2[2]; 
                double mH2     = MB2[3]; 
                double mHpm2   = MB2[4]; 
                double mW2L    = MB2[5]; 
                double mZ2L    = MB2[6]; 
                double mga2L   = MB2[7]; 
                double mW2T    = MB2[8]; 
                double mZ2T    = MB2[9]; 
                double mga2T   = MB2[10]; 
                
                double Pisum = 0.0;
                Pisum += PiGh(mh2, T);
                Pisum += PiGG(mG2, T);
                Pisum += PiGA(mA2, T);
                Pisum += PiGH(mH2, T);
                Pisum += PiGHpm(mHpm2, T);
                Pisum += PiGZ(mZ2L, mZ2T, T, swL, swT);
                Pisum += PiGga(mga2L, mga2T, T, swL, swT);
                Pisum += PiGW(mW2L, mW2T, T);
                
                return Pisum;
            }

            // ======================== Particle A's three-points vertex self-energy =========================
            
            /**
             * Polarization function PiAAh from A and h contributions
             */
            double PiAAh(double mA2, double mh2, double T, double phi) {
                return -square(lamm) * square(phi) * Imix(mh2, mA2, T);
            }

            /**
             * Polarization function PiAHG from H and G contributions
             */
            double PiAHG(double mH2, double mG2, double T, double phi) {
                return -square(lam5) * square(phi) * Imix(mH2, mG2, T);
            }

            /**
             * Polarization function PiAHpmG from Hpm and G contributions
             */
            double PiAHpmG(double mHpm2, double mG2, double T, double phi) {
                return -0.5 * square(lam4 - lam5) * square(phi) * Imix(mHpm2, mG2, T);
            }

            // ======================== Particle A's four-points vertex self-energy =========================

            /**
             * Polarization function PiAh from h particle contribution
             */
            double PiAh(double mh2, double T) {
                return 0.5 * lamm * Ijb(mh2, T, 1);
            }

            /**
             * Polarization function PiAG from G particle contribution
             */
            double PiAG(double mG2, double T) {
                return (0.5 * lamp + lam3) * Ijb(mG2, T, 1);
            }

            /**
             * Polarization function PiAA from A particle contribution
             */
            double PiAA(double mA2, double T) {
                return 1.5 * lam2 * Ijb(mA2, T, 1);
            }

            /**
             * Polarization function PiAH from H particle contribution
             */
            double PiAH(double mH2, double T) {
                return 0.5 * lam2 * Ijb(mH2, T, 1);
            }

            /**
             * Polarization function PiAHpm from Hpm particle contribution
             */
            double PiAHpm(double mHpm2, double T) {
                return lam2 * Ijb(mHpm2, T, 1);
            }

            /**
             * Polarization function PiAZ from Z boson contribution
             */
            double PiAZ(double mZ2L, double mZ2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double L_term = 0.25 * square(g * cwL + gp * swL) * Ijb(mZ2L, T, 1);
                double T_term = 0.25 * square(g * cwT + gp * swT) * (2.0 * Ijb(mZ2T, T, 1) + UV_term(mZ2T, -2, 1));
                
                return L_term + T_term;
            }

            /**
             * Polarization function PiAga from photon contribution
             */
            double PiAga(double mga2L, double mga2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double L_term = 0.25 * square(g * swL - gp * cwL) * Ijb(mga2L, T, 1);
                double T_term = 0.25 * square(g * swT - gp * cwT) * (2.0 * Ijb(mga2T, T, 1) + UV_term(mga2T, -2, 1));
                
                return L_term + T_term;
            }

            /**
             * Polarization function PiAW from W boson contribution
             */
            double PiAW(double mW2L, double mW2T, double T) {
                return 0.5 * square(g) * (Ijb(mW2L, T, 1) + 2.0 * Ijb(mW2T, T, 1) + UV_term(mW2T, -2, 1));
            }

            /**
             * 3-points sum polarization function PiA3
             */
            double PiA3(double T, const std::vector<double>& MB2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                
                double Pisum = 0.0;
                Pisum += PiAAh(mA2, mh2, T, phi);
                Pisum += PiAHG(mH2, mG2, T, phi);
                Pisum += PiAHpmG(mHpm2, mG2, T, phi);
                
                return Pisum;
            }

            /**
             * 4-points sum polarization function PiA4
             */
            double PiA4(double T, const std::vector<double>& MB2, double swL, double swT) {
                double mh2     = MB2[0];
                double mG2     = MB2[1];
                double mA2     = MB2[2];
                double mH2     = MB2[3];
                double mHpm2   = MB2[4];
                double mW2L    = MB2[5];
                double mZ2L    = MB2[6];
                double mga2L   = MB2[7];
                double mW2T    = MB2[8];
                double mZ2T    = MB2[9];
                double mga2T   = MB2[10];
                
                double Pisum = 0.0;
                Pisum += PiAh(mh2, T);
                Pisum += PiAG(mG2, T);
                Pisum += PiAA(mA2, T);
                Pisum += PiAH(mH2, T);
                Pisum += PiAHpm(mHpm2, T);
                Pisum += PiAZ(mZ2L, mZ2T, T, swL, swT);
                Pisum += PiAga(mga2L, mga2T, T, swL, swT);
                Pisum += PiAW(mW2L, mW2T, T);
                
                return Pisum;
            }
           
            // ======================== Particle H's three-points vertex self-energy =========================
            
            /**
             * Polarization function PiHHh from H and h contributions
             */
            double PiHHh(double mH2, double mh2, double T, double phi) {
                return -square(lamp) * square(phi) * Imix(mH2, mh2, T);
            }

            /**
             * Polarization function PiHAG from A and G contributions
             */
            double PiHAG(double mA2, double mG2, double T, double phi) {
                return -square(lam5) * square(phi) * Imix(mA2, mG2, T);
            }

            /**
             * Polarization function PiHHpmG from Hpm and G contributions
             */
            double PiHHpmG(double mHpm2, double mG2, double T, double phi) {
                return -0.5 * square(lam4 + lam5) * square(phi) * Imix(mHpm2, mG2, T);
            }

            // ======================== Particle H's four-points vertex self-energy =========================

            /**
             * Polarization function PiHh from h particle contribution
             */
            double PiHh(double mh2, double T) {
                return 0.5 * lamp * Ijb(mh2, T, 1);
            }

            /**
             * Polarization function PiHG from G particle contribution
             */
            double PiHG(double mG2, double T) {
                return (0.5 * lamm + lam3) * Ijb(mG2, T, 1);
            }

            /**
             * Polarization function PiHA from A particle contribution
             */
            double PiHA(double mA2, double T) {
                return 0.5 * lam2 * Ijb(mA2, T, 1);
            }

            /**
             * Polarization function PiHH from H particle contribution
             */
            double PiHH(double mH2, double T) {
                return 1.5 * lam2 * Ijb(mH2, T, 1);
            }

            /**
             * Polarization function PiHHpm from Hpm particle contribution
             */
            double PiHHpm(double mHpm2, double T) {
                return lam2 * Ijb(mHpm2, T, 1);
            }

            /**
             * Polarization function PiHZ from Z boson contribution
             */
            double PiHZ(double mZ2L, double mZ2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double L_term = 0.25 * square(g * cwL + gp * swL) * Ijb(mZ2L, T, 1);
                double T_term = 0.25 * square(g * cwT + gp * swT) * (2.0 * Ijb(mZ2T, T, 1) + UV_term(mZ2T, -2, 1));
                
                return L_term + T_term;
            }

            /**
             * Polarization function PiHga from photon contribution
             */
            double PiHga(double mga2L, double mga2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double L_term = 0.25 * square(g * swL - gp * cwL) * Ijb(mga2L, T, 1);
                double T_term = 0.25 * square(g * swT - gp * cwT) * (2.0 * Ijb(mga2T, T, 1) + UV_term(mga2T, -2, 1));
                
                return L_term + T_term;
            }

            /**
             * Polarization function PiHW from W boson contribution
             */
            double PiHW(double mW2L, double mW2T, double T) {
                return 0.5 * square(g) * (Ijb(mW2L, T, 1) + 2.0 * Ijb(mW2T, T, 1) + UV_term(mW2T, -2, 1));
            }

            /**
             * 3-points sum polarization function PiH3
             */
            double PiH3(double T, const std::vector<double>& MB2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                
                double Pisum = 0.0;
                Pisum += PiHHh(mH2, mh2, T, phi);
                Pisum += PiHAG(mA2, mG2, T, phi);
                Pisum += PiHHpmG(mHpm2, mG2, T, phi);
                
                return Pisum;
            }

            /**
             * 4-points sum polarization function PiH4
             */
            double PiH4(double T, const std::vector<double>& MB2, double swL, double swT) {
                double mh2     = MB2[0];
                double mG2     = MB2[1];
                double mA2     = MB2[2];
                double mH2     = MB2[3];
                double mHpm2   = MB2[4];
                double mW2L    = MB2[5];
                double mZ2L    = MB2[6];
                double mga2L   = MB2[7];
                double mW2T    = MB2[8];
                double mZ2T    = MB2[9];
                double mga2T   = MB2[10];
                
                double Pisum = 0.0;
                Pisum += PiHh(mh2, T);
                Pisum += PiHG(mG2, T);
                Pisum += PiHA(mA2, T);
                Pisum += PiHH(mH2, T);
                Pisum += PiHHpm(mHpm2, T);
                Pisum += PiHZ(mZ2L, mZ2T, T, swL, swT);
                Pisum += PiHga(mga2L, mga2T, T, swL, swT);
                Pisum += PiHW(mW2L, mW2T, T);
                
                return Pisum;
            }

            // ======================== Particle Hpm's three-points vertex self-energy =========================
            
            /**
             * Polarization function PiHpmHpmh from Hpm and h contributions
             */
            double PiHpmHpmh(double mHpm2, double mh2, double T, double phi) {
                return -square(lam3) * square(phi) * Imix(mHpm2, mh2, T);
            }

            /**
             * Polarization function PiHpmAG from A and G contributions
             */
            double PiHpmAG(double mA2, double mG2, double T, double phi) {
                return -0.25 * square(lam4 - lam5) * square(phi) * Imix(mA2, mG2, T);
            }

            /**
             * Polarization function PiHpmHG from H and G contributions
             */
            double PiHpmHG(double mH2, double mG2, double T, double phi) {
                return -0.25 * square(lam4 + lam5) * square(phi) * Imix(mH2, mG2, T);
            }

            // ======================== Particle Hpm's four-points vertex self-energy =========================

            /**
             * Polarization function PiHpmh from h particle contribution
             */
            double PiHpmh(double mh2, double T) {
                return 0.5 * lam3 * Ijb(mh2, T, 1);
            }

            /**
             * Polarization function PiHpmG from G particle contribution
             */
            double PiHpmG(double mG2, double T) {
                return (1.5 * lam3 + lam4) * Ijb(mG2, T, 1);
            }

            /**
             * Polarization function PiHpmA from A particle contribution
             */
            double PiHpmA(double mA2, double T) {
                return 0.5 * lam2 * Ijb(mA2, T, 1);
            }

            /**
             * Polarization function PiHpmH from H particle contribution
             */
            double PiHpmH(double mH2, double T) {
                return 0.5 * lam2 * Ijb(mH2, T, 1);
            }

            /**
             * Polarization function PiHpmHpm from Hpm particle contribution
             */
            double PiHpmHpm(double mHpm2, double T) {
                return 2.0 * lam2 * Ijb(mHpm2, T, 1);
            }

            /**
             * Polarization function PiHpmZ from Z boson contribution
             */
            double PiHpmZ(double mZ2L, double mZ2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double L_term = 0.25 * square(g * cwL - gp * swL) * Ijb(mZ2L, T, 1);
                double T_term = 0.25 * square(g * cwT - gp * swT) * (2.0 * Ijb(mZ2T, T, 1) + UV_term(mZ2T, -2, 1));
                
                return L_term + T_term;
            }

            /**
             * Polarization function PiHpmga from photon contribution
             */
            double PiHpmga(double mga2L, double mga2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                
                double L_term = 0.25 * square(g * swL + gp * cwL) * Ijb(mga2L, T, 1);
                double T_term = 0.25 * square(g * swT + gp * cwT) * (2.0 * Ijb(mga2T, T, 1) + UV_term(mga2T, -2, 1));
                
                return L_term + T_term;
            }

            /**
             * Polarization function PiHpmW from W boson contribution
             */
            double PiHpmW(double mW2L, double mW2T, double T) {
                return 0.5 * square(g) * (Ijb(mW2L, T, 1) + 2.0 * Ijb(mW2T, T, 1) + UV_term(mW2T, -2, 1));
            }

            /**
             * 3-points sum polarization function PiHpm3
             */
            double PiHpm3(double T, const std::vector<double>& MB2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                
                double Pisum = 0.0;
                Pisum += PiHpmAG(mA2, mG2, T, phi);
                Pisum += PiHpmHG(mH2, mG2, T, phi);
                Pisum += PiHpmHpmh(mHpm2, mh2, T, phi);
                
                return Pisum;
            }

            /**
             * 4-points sum polarization function PiHpm4
             */
            double PiHpm4(double T, const std::vector<double>& MB2, double swL, double swT) {
                double mh2     = MB2[0];
                double mG2     = MB2[1];
                double mA2     = MB2[2];
                double mH2     = MB2[3];
                double mHpm2   = MB2[4];
                double mW2L    = MB2[5];
                double mZ2L    = MB2[6];
                double mga2L   = MB2[7];
                double mW2T    = MB2[8];
                double mZ2T    = MB2[9];
                double mga2T   = MB2[10];
                
                double Pisum = 0.0;
                Pisum += PiHpmh(mh2, T);
                Pisum += PiHpmG(mG2, T);
                Pisum += PiHpmA(mA2, T);
                Pisum += PiHpmH(mH2, T);
                Pisum += PiHpmHpm(mHpm2, T);
                Pisum += PiHpmZ(mZ2L, mZ2T, T, swL, swT);
                Pisum += PiHpmga(mga2L, mga2T, T, swL, swT);
                Pisum += PiHpmW(mW2L, mW2T, T);
                
                return Pisum;
            }



    };




}

#endif