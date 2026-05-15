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
#include <Eigen/Dense>
#include <string>

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


    struct gapEqResult {
        std::vector<double> x; // vector of mass squared for all particles, in the order of mh2, mG2, mA2, mH2, mHpm2, mW2L, mZ2L, mga2L, mW2T, mZ2T, mga2T,  swL, swT
        std::string message; // message for convergence status
        double loss; // the loss function value at the solution
        bool success;
        
        gapEqResult() : x(13, 0.0) {}
    };



    class SelfEnergy {
        private:
            // constants
            const double v0 = 246.22; // GeV
            const double mh = 125.10; // GeV
            const double mt = 172.76; // GeV
            const double yt = sqrt(2.0)*mt/v0;
            const double g = 0.65175;
            const double gp = 0.35742;
            //const double gstar = 110.75;
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
                double L_term = -0.125 * pow_4(g * cwL + gp * swL) * square(phi) * Ijb(mZ2L, T, 2);
                double T_term = -0.125 * pow_4(g * cwT + gp * swT) * square(phi) * (2.0 * Ijb(mZ2T, T, 2) + UV_term(mZ2T, -2.0, 2));
                return L_term + T_term;
            }

            double DPihZZ_Dm2(double mZ2, double sw) {
                double cw = std::sqrt(1.0 - square(sw));
                return -0.125 * pow_4(g * cw + gp * sw) * square(v0) * (3.0 * zeroT_term_deriv(mZ2, 3) + UV_term(mZ2, -2.0, 3));
            }

            // Photon (gamma) contribution: Pihgaga
            double Pihgaga(double mga2L, double mga2T, double T, double swL, double swT, double phi) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double L_term = -0.125 * pow_4(g * swL - gp * cwL) * square(phi) * Ijb(mga2L, T, 2);
                double T_term = -0.125 * pow_4(g * swT - gp * cwT) * square(phi) * (2.0 * Ijb(mga2T, T, 2) + UV_term(mga2T, -2.0, 2));
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
                return -0.25 * pow_4(g) * square(phi) * (Ijb(mW2L, T, 2) + 2.0 * Ijb(mW2T, T, 2) + UV_term(mW2T, -2.0, 2));
            }
            
            double DPihWW_Dm2(double mW2) {
                return -0.25 * pow_4(g) * square(v0) * (3.0 * zeroT_term_deriv(mW2, 3) + UV_term(mW2, -2.0, 3));
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

            double Pih3Pdependent(double T, double p2, const std::array<double, 13>& MB2, double mt2, double phi) {
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
                    double swL    = MB2[11];
                    double swT     = MB2[12];

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
                double result = gauss_kronrod<double, 15>::integrate(integrand, 0.0, 1.0, 50, 1e-4);
                return result;
            }

            double DPih3_Dp2(double p2, const std::array<double, 13>& MB2, double mt2) {
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
                    double sw        = MB2[11];

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

                double result = gauss_kronrod<double, 15>::integrate(integrand, 0.0, 1.0, 50, 1e-4);
                return result;
            }

            // Vector overloads for external interface compatibility
            double Pih3Pdependent(double T, double p2, const std::vector<double>& MB2_vec, double mt2, double phi) {
                if (MB2_vec.size() != 13) throw std::invalid_argument("MB2 size must be 13");
                std::array<double, 13> MB2_arr;
                std::copy(MB2_vec.begin(), MB2_vec.end(), MB2_arr.begin());
                return Pih3Pdependent(T, p2, MB2_arr, mt2, phi);
            }

            double DPih3_Dp2(double p2, const std::vector<double>& MB2_vec, double mt2) {
                if (MB2_vec.size() != 13) throw std::invalid_argument("MB2 size must be 13");
                std::array<double, 13> MB2_arr;
                std::copy(MB2_vec.begin(), MB2_vec.end(), MB2_arr.begin());
                return DPih3_Dp2(p2, MB2_arr, mt2);
            }

            double Pih3(double T, const std::vector<double>& MB2, double mt2, double phi) {
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
                double swL     = MB2[11];
                double swT     = MB2[12];

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

            double Pih4(double T, const std::vector<double>& MB2) {
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
                double swL     = MB2[11];
                double swT     = MB2[12];

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

            double PiG4(double T, const std::vector<double>& MB2) {
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
                double swL     = MB2[11]; 
                double swT     = MB2[12]; 

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
            double PiA4(double T, const std::vector<double>& MB2) {
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
                double swL     = MB2[11];
                double swT     = MB2[12];

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
            double PiH4(double T, const std::vector<double>& MB2) {
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
                double swL = MB2[11];
                double swT = MB2[12];
                
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
            double PiHpm4(double T, const std::vector<double>& MB2) {
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
                double swL = MB2[11];
                double swT = MB2[12];

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


            // ======================== Gauge bosons three-points vertex self-energy (Transverse) =========================
            
            // Polarization function PiWTff from top quark contribution
            double PiWTff(double mt2, double T) {
                if (std::abs(mt2) < 1e-4) {
                    return square(g) * (-Ijf(mt2/2.0, T, 1) - UV_term(mt2/2.0, 1.0/6.0, 1) + 2.0 * DTIjf(mt2/2.0, T, 1) - Ijf(mt2, T, 1) - UV_term(mt2, -11.0/6.0, 1));
                } else {
                    return square(g) * ((I0f(mt2, T) - I0f(0.0, T) + UV_term(mt2, 1.0/6.0, 0))/mt2 - 2.0 * (DTIjf(mt2, T, 0) - DTIjf(0.0, T, 0))/mt2 - Ijf(mt2, T, 1) - UV_term(mt2, -11.0/6.0, 1));
                }
            }

            // Polarization function PiTSS from a and b contributions
            // Applicable to PiWThG, PiWTHHpm, PiWTAHpm, Pi33ThG, Pi33TAH, Pi44ThG, Pi44TAH, Pi34ThG, Pi34TAH
            double PiTSS(double ma2, double mb2, double T) {
                double deltaM2 = mb2 - ma2;
                double Temp_term, UV_term_val;
                
                if (std::abs(deltaM2) < 1e-4) {
                    double arg = (ma2 + mb2) / 2.0;
                    Temp_term = -Ijb(mb2, T, 1) + arg * Ijb(arg, T, 2) - (-0.5 * Ijb(arg, T, 1) + DTIjb(arg, T, 1));
                    UV_term_val = -UV_term(mb2, 2.0/3.0, 1) + arg * UV_term(arg, 2.0/3.0, 2) + 0.5 * UV_term(arg, 2.0/3.0, 1);
                } else {
                    Temp_term = -Ijb(mb2, T, 1) + ma2 * Imix(ma2, mb2, T) - (0.5 * I0b(mb2, T) - DTIjb(mb2, T, 0) - 0.5 * I0b(ma2, T) + DTIjb(ma2, T, 0))/deltaM2;
                    UV_term_val = -UV_term(mb2, 2.0/3.0, 1) + ma2 * (UV_term(ma2, 2.0/3.0, 1) - UV_term(mb2, 2.0/3.0, 1))/deltaM2 - (0.5 * UV_term(mb2, 2.0/3.0, 0) - 0.5 * UV_term(ma2, 2.0/3.0, 0))/deltaM2;
                }
                
                return Temp_term + UV_term_val;
            }

            // Polarization function PiWThG from h and G contributions
            double PiWThG(double mh2, double mG2, double T) {
                return square(g)/3.0 * PiTSS(mh2, mG2, T);
            }

            // Polarization function PiWTHHpm from H and Hpm contributions
            double PiWTHHpm(double mH2, double mHpm2, double T) {
                return square(g)/3.0 * PiTSS(mH2, mHpm2, T);
            }

            // Polarization function PiWTAHpm from A and Hpm contributions
            double PiWTAHpm(double mA2, double mHpm2, double T) {
                return square(g)/3.0 * PiTSS(mA2, mHpm2, T);
            }

            // Polarization function PiTGG from G and G contributions
            // Applicable to PiWTGG, Pi33TGG, Pi33THpHm, Pi44TGG, Pi44THpHm, Pi34TGG, Pi34THpHm
            double PiTGG(double mG2, double T) {
                return -0.5 * Ijb(mG2, T, 1) + mG2 * Ijb(mG2, T, 2) - DTIjb(mG2, T, 1) - 0.5 * UV_term(mG2, 2.0/3.0, 1) + mG2 * UV_term(mG2, 2.0/3.0, 2);
            }

            // Polarization function PiWTGG from G and G contributions
            double PiWTGG(double mG2, double T) {
                return square(g)/3.0 * PiTGG(mG2, T);
            }

            // Polarization function PiTSV from scalar and vector contributions
            // Applicable to PiWThW, PiWTGga, PiWTGZ
            double PiTSV(double ms2, double mv2L, double mv2T, double T, double gL, double gT) {
                double deltaM2L = ms2 - mv2L;
                double deltaM2T = ms2 - mv2T;
                
                double TransverTerm;
                if (std::abs(deltaM2T) < 1e-4) {
                    double argT = (mv2T + ms2) / 2.0;
                    TransverTerm = 2.0 * Ijb(argT, T, 2) + UV_term(argT, -2.0/3.0, 2);
                } else {
                    TransverTerm = 2.0 * Imix(mv2T, ms2, T) + (UV_term(mv2T, -2.0/3.0, 1) - UV_term(ms2, -2.0/3.0, 1))/deltaM2T;
                }
                
                double LongTerm;
                if (std::abs(ms2) < 1e-3) {
                    if (std::abs(deltaM2L) < 1e-5) {
                        double argL = (mv2L + ms2) / 2.0;
                        LongTerm = 0.5 * Ijb(argL, T, 2) - DTIjb(argL, T, 2) + 0.5 * UV_term(argL, 2.0/3.0, 2);
                    } else {
                        double LongTerm1 = -(0.5 * Ijb(ms2/2.0, T, 1) - DTIjb(ms2/2.0, T, 1))/deltaM2L - 0.5 * UV_term(ms2/2.0, 2.0/3.0, 1)/deltaM2L;
                        double LongTerm2 = (-0.5 * I0b(mv2L, T) + DTIjb(mv2L, T, 0) - 1.5 * I0b(0.0, T))/(mv2L * deltaM2L) + (-0.5 * UV_term(mv2L, 2.0/3.0, 0))/(mv2L * deltaM2L);
                        LongTerm = LongTerm1 + LongTerm2;
                    }
                } else {
                    if (std::abs(deltaM2L) < 1e-5) {
                        double argL = (mv2L + ms2) / 2.0;
                        double LongTerm1 = -0.5 * (Ijb(argL, T, 0) + argL * Ijb(argL, T, 1))/square(argL) - 0.5 * (UV_term(argL, 2.0/3.0, 0) + argL * UV_term(argL, 2.0/3.0, 1))/square(argL);
                        double LongTerm2 = (DTIjb(argL, T, 0) + argL * DTIjb(argL, T, 1))/square(argL) - 1.5 * I0b(0.0, T)/square(argL);
                        LongTerm = LongTerm1 + LongTerm2;
                    } else {
                        if (std::abs(mv2L) < 1e-3) {
                            double LongTerm1 = (0.5 * Ijb(mv2L/2.0, T, 1) - DTIjb(mv2L/2.0, T, 1))/deltaM2L + 0.5 * UV_term(mv2L/2.0, 2.0/3.0, 1)/deltaM2L;
                            double LongTerm2 = -(-0.5 * I0b(ms2, T) + DTIjb(ms2, T, 0) - 1.5 * I0b(0.0, T))/(ms2 * deltaM2L) + 0.5 * UV_term(ms2, 2.0/3.0, 0)/(ms2 * deltaM2L);
                            LongTerm = LongTerm1 + LongTerm2;
                        } else {
                            LongTerm = -(-0.5 * I0b(ms2, T) + DTIjb(ms2, T, 0) - 1.5 * I0b(0.0, T))/(ms2 * deltaM2L) + (-0.5 * I0b(mv2L, T) + DTIjb(mv2L, T, 0) - 1.5 * I0b(0.0, T))/(mv2L * deltaM2L) - (-0.5 * UV_term(ms2, 2.0/3.0, 0))/(ms2 * deltaM2L) + (-0.5 * UV_term(mv2L, 2.0/3.0, 0))/(mv2L * deltaM2L);
                        }
                    }
                }
                
                return gT * TransverTerm + gL * LongTerm;
            }

            // Polarization function PiWThW from h and W contributions
            double PiWThW(double mh2, double mW2L, double mW2T, double T, double phi) {
                return -pow_4(g) * square(phi) / 12.0 * PiTSV(mh2, mW2L, mW2T, T, 1.0, 1.0);
            }

            // Polarization function PiWTGga from G and photon contributions
            double PiWTGga(double mG2, double mga2L, double mga2T, double T, double phi, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL2 = 1.0 - square(swL);
                double cwT2 = 1.0 - square(swT);
                return -square(g) * square(gp) * square(phi) / 12.0 * PiTSV(mG2, mga2L, mga2T, T, cwL2, cwT2);
            }

            // Polarization function PiWTGZ from G and Z contributions
            double PiWTGZ(double mG2, double mZ2L, double mZ2T, double T, double phi, double swL, double swT) {
                return -square(g) * square(gp) * square(phi) / 12.0 * PiTSV(mG2, mZ2L, mZ2T, T, square(swL), square(swT));
            }

            // Polarization function PiWTVV from two vector boson contributions
            double PiWTVV(double m12, double m22, double T, double eps) {
                double deltaM2 = m22 - m12;
                double T_term, UV_term_val;
                
                if (std::abs(deltaM2) < 1e-4) {
                    double arg = (m22 + m12) / 2.0;
                    T_term = -Ijb(m22, T, 1) + m12 * Ijb(arg, T, 2) - (-0.5 * Ijb(arg, T, 1) + DTIjb(arg, T, 1));
                    UV_term_val = -UV_term(m22, eps, 1) + m12 * UV_term(arg, eps, 2) + 0.5 * UV_term(arg, eps, 1);
                } else {
                    T_term = -Ijb(m22, T, 1) + m12 * Imix(m12, m22, T) - (0.5 * I0b(m22, T) - DTIjb(m22, T, 0) - 0.5 * I0b(m12, T) + DTIjb(m12, T, 0))/deltaM2;
                    UV_term_val = -UV_term(m22, eps, 1) + m12 * (UV_term(m12, eps, 1) - UV_term(m22, eps, 1))/deltaM2 - (0.5 * UV_term(m22, eps, 0) - 0.5 * UV_term(m12, eps, 0))/deltaM2;
                }
                
                return T_term + UV_term_val;
            }

            // Polarization function PiWTgaW from photon and W contributions
            double PiWTgaW(double mW2L, double mga2L, double mW2T, double mga2T, double T, double swL, double swT) {
                return 4.0 * square(g) * (2.0/3.0 * square(swT) * PiWTVV(mga2T, mW2T, T, -1.0/3.0) + 1.0/3.0 * square(swL) * PiWTVV(mga2L, mW2L, T, 2.0/3.0));
            }

            // Polarization function PiWTZW from Z and W contributions
            double PiWTZW(double mW2L, double mZ2L, double mW2T, double mZ2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL2 = 1.0 - square(swL);
                double cwT2 = 1.0 - square(swT);
                return 4.0 * square(g) * (2.0/3.0 * cwT2 * PiWTVV(mZ2T, mW2T, T, -1.0/3.0) + 1.0/3.0 * cwL2 * PiWTVV(mZ2L, mW2L, T, 2.0/3.0));
            }

            // Polarization function PiWTcc from ghost contributions
            double PiWTcc(double T) {
                return square(g) * Ijb(0.0, T, 1);
            }

            // Polarization function sum PiWT3
            double PiWT3(double T, const std::vector<double>& MB2, double mt2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double mW2T = MB2[8];
                double mZ2T = MB2[9];
                double mga2T = MB2[10];
                double swL = MB2[11];
                double swT = MB2[12];
                
                double Pisum = 0.0;
                Pisum += PiWTff(mt2, T);
                Pisum += PiWThG(mh2, mG2, T);
                Pisum += PiWTGG(mG2, T);
                Pisum += PiWTHHpm(mH2, mHpm2, T);
                Pisum += PiWTAHpm(mA2, mHpm2, T);
                Pisum += PiWThW(mh2, mW2L, mW2T, T, phi);
                Pisum += PiWTGga(mG2, mga2L, mga2T, T, phi, swL, swT);
                Pisum += PiWTGZ(mG2, mZ2L, mZ2T, T, phi, swL, swT);
                Pisum += PiWTgaW(mW2L, mga2L, mW2T, mga2T, T, swL, swT);
                Pisum += PiWTZW(mW2L, mZ2L, mW2T, mZ2T, T, swL, swT);
                Pisum += PiWTcc(T);
                
                return Pisum;
            }

            // ======================== Gauge bosons four-points vertex self-energy (Transverse) =========================
            
            // Polarization function PiWTh from h contribution
            double PiWTh(double mh2, double T) {
                return 0.25 * square(g) * Ijb(mh2, T, 1);
            }

            // Polarization function PiWTG from G contribution
            double PiWTG(double mG2, double T) {
                return 0.75 * square(g) * Ijb(mG2, T, 1);
            }

            // Polarization function PiWTA from A contribution
            double PiWTA(double mA2, double T) {
                return 0.25 * square(g) * Ijb(mA2, T, 1);
            }

            // Polarization function PiWTH from H contribution
            double PiWTH(double mH2, double T) {
                return 0.25 * square(g) * Ijb(mH2, T, 1);
            }

            // Polarization function PiWTHpm from Hpm contribution
            double PiWTHpm(double mHpm2, double T) {
                return 0.5 * square(g) * Ijb(mHpm2, T, 1);
            }

            // Polarization function PiTV from vector boson contribution
            double PiTV(double mV2L, double mV2T, double T, double gL, double gT) {
                double T_term = 4.0/3.0 * (Ijb(mV2T, T, 1) + UV_term(mV2T, -4.0/3.0, 1));
                double L_term;
                
                if (std::abs(mV2L) < 1e-4) {
                    L_term = 1.0/6.0 * (-Ijb(mV2L/2.0, T, 1) + 2.0 * DTIjb(mV2L/2.0, T, 1) - UV_term(mV2L/2.0, 2.0/3.0, 1)) + Ijb(mV2L, T, 1);
                } else {
                    L_term = 1.0/(6.0 * mV2L) * (I0b(mV2L, T) - 2.0 * DTIjb(mV2L, T, 0) - I0b(0.0, T) + 2.0 * DTIjb(0.0, T, 0) + UV_term(mV2L, 2.0/3.0, 0)) + Ijb(mV2L, T, 1);
                }
                
                return gT * T_term + gL * L_term;
            }

            // Polarization function PiWTZ from Z boson contribution
            double PiWTZ(double mZ2L, double mZ2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL2 = 1.0 - square(swL);
                double cwT2 = 1.0 - square(swT);
                return square(g) * PiTV(mZ2L, mZ2T, T, cwL2, cwT2);
            }

            // Polarization function PiWTW from W boson contribution
            double PiWTW(double mW2L, double mW2T, double T) {
                return square(g) * PiTV(mW2L, mW2T, T, 1.0, 1.0);
            }

            // Polarization function PiWTga from photon contribution
            double PiWTga(double mga2L, double mga2T, double T, double swL, double swT) {
                return square(g) * PiTV(mga2L, mga2T, T, square(swL), square(swT));
            }

            // Polarization function sum PiWT4
            double PiWT4(double T, const std::vector<double>& MB2) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double mW2T = MB2[8];
                double mZ2T = MB2[9];
                double mga2T = MB2[10];
                double swL = MB2[11];
                double swT = MB2[12];
                
                double Pisum = 0.0;
                Pisum += PiWTh(mh2, T);
                Pisum += PiWTG(mG2, T);
                Pisum += PiWTA(mA2, T);
                Pisum += PiWTH(mH2, T);
                Pisum += PiWTHpm(mHpm2, T);
                Pisum += PiWTZ(mZ2L, mZ2T, T, swL, swT);
                Pisum += PiWTW(mW2L, mW2T, T);
                Pisum += PiWTga(mga2L, mga2T, T, swL, swT);
                
                return Pisum;
            }

            // ======================== Gauge bosons (W) three-points vertex self-energy (Longitudinal) =========================
            
            // Polarization function PiWLff from fermion contribution
            double PiWLff(double mt2, double T) {
                double T_term, UV_term_val;
                
                if (std::abs(mt2) < 1e-4) {
                    T_term = 3.0 * square(g) * (Ijf(mt2/2.0, T, 1) - 2.0 * DTIjf(mt2/2.0, T, 1) - Ijf(mt2, T, 1) - 6.0 * DTIjf(0.0, T, 1));
                    UV_term_val = 3.0 * square(g) * (UV_term(mt2/2.0, -1.0/2.0, 1) - UV_term(mt2, -1.0/2.0, 1));
                } else {
                    T_term = 3.0 * square(g) * (-(I0f(mt2, T) - I0f(0.0, T))/mt2 + 2.0 * (DTIjf(mt2, T, 0) - DTIjf(0.0, T, 0))/mt2 - Ijf(mt2, T, 1) - 6.0 * DTIjf(0.0, T, 1));
                    UV_term_val = 3.0 * square(g) * (-UV_term(mt2, -1.0/2.0, 0)/mt2 - UV_term(mt2, -1.0/2.0, 1));
                }
                
                return T_term + UV_term_val;
            }

            // Polarization function PiWLGG from G and G contributions
            double PiWLGG(double mG2, double T) {
                return -0.5 * square(g) * (Ijb(mG2, T, 1) - 2.0 * DTIjb(mG2, T, 1));
            }

            // Polarization function PiLSS from two scalar contributions
            // Applicable to PiWLhG, PiWLHHpm, PiWLAHpm, Pi33LhG, Pi33LAH, Pi44LhG, Pi44LAH, Pi34LhG, Pi34LAH
            double PiLSS(double s1, double s2, double T) {
                double deltaM2 = s2 - s1;
                double result;
                
                if (std::abs(deltaM2) < 1e-4) {
                    double arg = (s2 + s1) / 2.0;
                    result = -Ijb(arg, T, 1) + 2.0 * DTIjb(arg, T, 1);
                } else {
                    result = (I0b(s2, T) - 2.0 * DTIjb(s2, T, 0) - I0b(s1, T) + 2.0 * DTIjb(s1, T, 0)) / deltaM2;
                }
                
                return result;
            }

            // Polarization function PiWLhG from h and G contributions
            double PiWLhG(double mh2, double mG2, double T) {
                return 0.5 * square(g) * PiLSS(mh2, mG2, T);
            }

            // Polarization function PiWLHHpm from H and Hpm contributions
            double PiWLHHpm(double mH2, double mHpm2, double T) {
                return 0.5 * square(g) * PiLSS(mH2, mHpm2, T);
            }

            // Polarization function PiWLAHpm from A and Hpm contributions
            double PiWLAHpm(double mA2, double mHpm2, double T) {
                return 0.5 * square(g) * PiLSS(mA2, mHpm2, T);
            }

            // Polarization function PiLSV from scalar and vector contributions
            // Applicable to PiWLhW, PiWLGga, PiWLGZ, Pi33LhZ, Pi44LGW, Pi44LhZ, Pi34LhZ
            double PiLSV(double ms2, double mv2L, double T) {
                double deltaM2L = ms2 - mv2L;
                double LongTerm;
                
                if (std::abs(ms2) < 1e-3) {
                    if (std::abs(deltaM2L) < 1e-5) {
                        double argL = (mv2L + ms2) / 2.0;
                        LongTerm = Ijb(argL, T, 2) - 0.5 * Ijb(argL, T, 2) + DTIjb(argL, T, 2);
                    } else {
                        double LongTerm1 = (0.5 * Ijb(ms2/2.0, T, 1) - DTIjb(ms2/2.0, T, 1)) / deltaM2L;
                        double LongTerm2 = -(-0.5 * I0b(mv2L, T) + DTIjb(mv2L, T, 0) - 1.5 * I0b(0.0, T)) / (mv2L * deltaM2L);
                        LongTerm = Imix(mv2L, ms2, T) + LongTerm1 + LongTerm2;
                    }
                } else {
                    if (std::abs(deltaM2L) < 1e-5) {
                        double argL = (mv2L + ms2) / 2.0;
                        double LongTerm1 = 0.5 * (Ijb(argL, T, 0) + argL * Ijb(argL, T, 1)) / square(argL);
                        double LongTerm2 = -(DTIjb(argL, T, 0) + argL * DTIjb(argL, T, 1)) / square(argL);
                        LongTerm = Ijb(argL, T, 2) + LongTerm1 + LongTerm2 + 1.5 * I0b(0.0, T) / square(argL);
                    } else {
                        if (std::abs(mv2L) < 1e-3) {
                            double LongTerm1 = -(0.5 * Ijb(mv2L/2.0, T, 1) - DTIjb(mv2L/2.0, T, 1)) / deltaM2L;
                            double LongTerm2 = (-0.5 * I0b(ms2, T) + DTIjb(ms2, T, 0) - 1.5 * I0b(0.0, T)) / (ms2 * deltaM2L);
                            LongTerm = Imix(mv2L, ms2, T) + LongTerm1 + LongTerm2;
                        } else {
                            LongTerm = Imix(mv2L, ms2, T) + (-0.5 * I0b(ms2, T) + DTIjb(ms2, T, 0) - 1.5 * I0b(0.0, T)) / (ms2 * deltaM2L) - (-0.5 * I0b(mv2L, T) + DTIjb(mv2L, T, 0) - 1.5 * I0b(0.0, T)) / (mv2L * deltaM2L);
                        }
                    }
                }
                
                return LongTerm;
            }

            // Polarization function PiWLhW from h and W contributions
            double PiWLhW(double mh2, double mW2L, double T, double phi) {
                return -0.25 * pow_4(g) * square(phi) * PiLSV(mh2, mW2L, T);
            }

            // Polarization function PiWLGga from G and photon contributions
            double PiWLGga(double mG2, double mga2L, double T, double phi, double swL) {
                swL = std::clamp(swL, 0.0, 1.0);
                double cwL2 = 1.0 - square(swL);
                return -0.25 * square(g) * square(gp) * cwL2 * square(phi) * PiLSV(mG2, mga2L, T);
            }

            // Polarization function PiWLGZ from G and Z contributions
            double PiWLGZ(double mG2, double mZ2L, double T, double phi, double swL) {
                return -0.25 * square(g) * square(gp) * square(swL) * square(phi) * PiLSV(mG2, mZ2L, T);
            }

            // Polarization function PiWLVV from two vector boson contributions
            double PiWLVV(double m12, double m22, double T, double eps) {
                double deltaM2 = m22 - m12;
                double result;
                
                if (std::abs(deltaM2) < 1e-4) {
                    double arg = (m22 + m12) / 2.0;
                    result = -Ijb(arg, T, 1) + 2.0 * DTIjb(arg, T, 1) - UV_term(arg, eps, 1);
                } else {
                    result = (Ijb(m22, T, 0) - Ijb(m12, T, 0) - 2.0 * DTIjb(m22, T, 0) + 2.0 * DTIjb(m12, T, 0) + UV_term(m22, eps, 0) - UV_term(m12, eps, 0)) / deltaM2;
                }
                
                return result;
            }

            // Polarization function PiWLgaW from photon and W contributions
            double PiWLgaW(double mW2L, double mga2L, double mW2T, double mga2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                return 2.0 * square(g) * (square(swL) * PiWLVV(mga2L, mW2L, T, 0.0) + 2.0 * square(swT) * PiWLVV(mga2T, mW2T, T, -1.0));
            }

            // Polarization function PiWLZW from Z and W contributions
            double PiWLZW(double mW2L, double mZ2L, double mW2T, double mZ2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL2 = 1.0 - square(swL);
                double cwT2 = 1.0 - square(swT);
                return 2.0 * square(g) * (cwL2 * PiWLVV(mZ2L, mW2L, T, 0.0) + 2.0 * cwT2 * PiWLVV(mZ2T, mW2T, T, -1.0));
            }

            // Polarization function PiWLcc from ghost contributions
            double PiWLcc(double T) {
                return -square(g) * Ijb(0.0, T, 1);
            }

            // Polarization function sum PiWL3
            double PiWL3(double T, const std::vector<double>& MB2, double mt2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double mW2T = MB2[8];
                double mZ2T = MB2[9];
                double mga2T = MB2[10];
                double swL = MB2[11];
                double swT = MB2[12];
                
                double Pisum = 0.0;
                Pisum += PiWLff(mt2, T);
                Pisum += PiWLhG(mh2, mG2, T);
                Pisum += PiWLGG(mG2, T);
                Pisum += PiWLHHpm(mH2, mHpm2, T);
                Pisum += PiWLAHpm(mA2, mHpm2, T);
                Pisum += PiWLhW(mh2, mW2L, T, phi);
                Pisum += PiWLGga(mG2, mga2L, T, phi, swL);
                Pisum += PiWLGZ(mG2, mZ2L, T, phi, swL);
                Pisum += PiWLgaW(mW2L, mga2L, mW2T, mga2T, T, swL, swT);
                Pisum += PiWLZW(mW2L, mZ2L, mW2T, mZ2T, T, swL, swT);
                Pisum += PiWLcc(T);
                
                return Pisum;
            }

            // ======================== Gauge bosons (W) four-points vertex self-energy (Longitudinal) =========================
            
            // Polarization function PiWLh from h contribution
            double PiWLh(double mh2, double T) {
                return 0.25 * square(g) * Ijb(mh2, T, 1);
            }

            // Polarization function PiWLG from G contribution
            double PiWLG(double mG2, double T) {
                return 0.75 * square(g) * Ijb(mG2, T, 1);
            }

            // Polarization function PiWLA from A contribution
            double PiWLA(double mA2, double T) {
                return 0.25 * square(g) * Ijb(mA2, T, 1);
            }

            // Polarization function PiWLH from H contribution
            double PiWLH(double mH2, double T) {
                return 0.25 * square(g) * Ijb(mH2, T, 1);
            }

            // Polarization function PiWLHpm from Hpm contribution
            double PiWLHpm(double mHpm2, double T) {
                return 0.5 * square(g) * Ijb(mHpm2, T, 1);
            }

            // Polarization function PiWLV from vector boson contribution
            double PiWLV(double mV2L, double mV2T, double T, double gL, double gT) {
                double T_term = 2.0 * Ijb(mV2T, T, 1) + UV_term(mV2T, -2.0, 1);
                double L_term;
                
                if (std::abs(mV2L) < 1e-4) {
                    L_term = -0.5 * (-Ijb(mV2L/2.0, T, 1) + 2.0 * DTIjb(mV2L/2.0, T, 1));
                } else {
                    L_term = -0.5 * (I0b(mV2L, T) - 2.0 * DTIjb(mV2L, T, 0) - I0b(0.0, T) + 2.0 * DTIjb(0.0, T, 0)) / mV2L;
                }
                
                return gT * T_term + gL * L_term;
            }

            // Polarization function PiWLZ from Z boson contribution
            double PiWLZ(double mZ2L, double mZ2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL2 = 1.0 - square(swL);
                double cwT2 = 1.0 - square(swT);
                return square(g) * PiWLV(mZ2L, mZ2T, T, cwL2, cwT2);
            }

            // Polarization function PiWLW from W boson contribution
            double PiWLW(double mW2L, double mW2T, double T) {
                return square(g) * PiWLV(mW2L, mW2T, T, 1.0, 1.0);
            }

            // Polarization function PiWLga from photon contribution
            double PiWLga(double mga2L, double mga2T, double T, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                return square(g) * PiWLV(mga2L, mga2T, T, square(swL), square(swT));
            }

            // Polarization function sum PiWL4
            double PiWL4(double T, const std::vector<double>& MB2) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double mW2T = MB2[8];
                double mZ2T = MB2[9];
                double mga2T = MB2[10];
                double swL = MB2[11];
                double swT = MB2[12];
                
                double Pisum = 0.0;
                Pisum += PiWLh(mh2, T);
                Pisum += PiWLG(mG2, T);
                Pisum += PiWLA(mA2, T);
                Pisum += PiWLH(mH2, T);
                Pisum += PiWLHpm(mHpm2, T);
                Pisum += PiWLZ(mZ2L, mZ2T, T, swL, swT);
                Pisum += PiWLW(mW2L, mW2T, T);
                Pisum += PiWLga(mga2L, mga2T, T, swL, swT);
                
                return Pisum;
            }

    
            // ======================== Gauge bosons (33) three-points vertex self-energy (Transverse) =========================
            
            // Polarization function Pi33Tff from fermion contribution
            double Pi33Tff(double mt2, double T) {
                double T_term = -(Ijf(mt2, T, 1) - 2.0 * DTIjf(mt2, T, 1) + 7.0 * Ijf(0.0, T, 1) - 14.0 * DTIjf(0.0, T, 1)) + (-Ijf(mt2, T, 1) + mt2 * Ijf(mt2, T, 2) - 7.0 * Ijf(0.0, T, 1));
                double UV_term_val = -UV_term(mt2, 1.0/6.0, 1) - UV_term(mt2, -11.0/6.0, 1) + mt2 * UV_term(mt2, -11.0/6.0, 2);
                return 0.5 * square(g) * (T_term + UV_term_val);
            }

            // Polarization function Pi33ThG from h and G contributions
            double Pi33ThG(double mh2, double mG2, double T) {
                return square(g)/3.0 * PiTSS(mh2, mG2, T);
            }

            // Polarization function Pi33TAH from A and H contributions
            double Pi33TAH(double mA2, double mH2, double T) {
                return square(g)/3.0 * PiTSS(mA2, mH2, T);
            }

            // Polarization function Pi33THpHm from Hpm contribution
            double Pi33THpHm(double mHpm2, double T) {
                return square(g)/3.0 * PiTGG(mHpm2, T);
            }

            // Polarization function Pi33TGG from G and G contributions
            double Pi33TGG(double mG2, double T) {
                return square(g)/3.0 * PiTGG(mG2, T);
            }

            // Polarization function Pi33TWW from W and W contributions
            double Pi33TWW(double mW2L, double mW2T, double T) {
                double T_term = -0.5 * Ijb(mW2T, T, 1) + mW2T * Ijb(mW2T, T, 2) - DTIjb(mW2T, T, 1) - 0.5 * UV_term(mW2T, -1.0/3.0, 1) + mW2T * UV_term(mW2T, -1.0/3.0, 2);
                double L_term = -0.5 * Ijb(mW2L, T, 1) + mW2L * Ijb(mW2L, T, 2) - DTIjb(mW2L, T, 1) - 0.5 * UV_term(mW2L, 2.0/3.0, 1) + mW2L * UV_term(mW2L, 2.0/3.0, 2);
                return 4.0 * square(g) * (2.0/3.0 * T_term + 1.0/3.0 * L_term);
            }

            // Polarization function Pi33ThZ from h and Z contributions
            double Pi33ThZ(double mh2, double mZ2L, double mZ2T, double T, double phi, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double gL = square(square(g) * cwL + g * gp * swL);
                double gT = square(square(g) * cwT + g * gp * swT);
                return -square(phi) / 12.0 * PiTSV(mh2, mZ2L, mZ2T, T, gL, gT);
            }

            // Polarization function Pi33Thga from h and photon contributions
            double Pi33Thga(double mh2, double mga2L, double mga2T, double T, double phi, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double gL = square(square(g) * swL - g * gp * cwL);
                double gT = square(square(g) * swT - g * gp * cwT);
                return -square(phi) / 12.0 * PiTSV(mh2, mga2L, mga2T, T, gL, gT);
            }

            // Polarization function Pi33Tcc from ghost contribution
            double Pi33Tcc(double T) {
                return square(g) * Ijb(0.0, T, 1);
            }

            // Polarization function sum Pi33T3
            double Pi33T3(double T, const std::vector<double>& MB2, double mt2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double mW2T = MB2[8];
                double mZ2T = MB2[9];
                double mga2T = MB2[10];
                double swL = MB2[11];
                double swT = MB2[12];
                
                double Pisum = 0.0;
                Pisum += Pi33Tff(mt2, T);
                Pisum += Pi33ThG(mh2, mG2, T);
                Pisum += Pi33TGG(mG2, T);
                Pisum += Pi33THpHm(mHpm2, T);
                Pisum += Pi33TAH(mA2, mH2, T);
                Pisum += Pi33TWW(mW2L, mW2T, T);
                Pisum += Pi33ThZ(mh2, mZ2L, mZ2T, T, phi, swL, swT);
                Pisum += Pi33Thga(mh2, mga2L, mga2T, T, phi, swL, swT);
                Pisum += Pi33Tcc(T);
                
                return Pisum;
            }            

            // ======================== Gauge bosons (33) four-points vertex self-energy (Transverse) =========================
            
            // Polarization function Pi33Th from h contribution
            double Pi33Th(double mh2, double T) {
                return 0.25 * square(g) * Ijb(mh2, T, 1);
            }

            // Polarization function Pi33TG from G contribution
            double Pi33TG(double mG2, double T) {
                return 0.75 * square(g) * Ijb(mG2, T, 1);
            }

            // Polarization function Pi33TA from A contribution
            double Pi33TA(double mA2, double T) {
                return 0.25 * square(g) * Ijb(mA2, T, 1);
            }

            // Polarization function Pi33TH from H contribution
            double Pi33TH(double mH2, double T) {
                return 0.25 * square(g) * Ijb(mH2, T, 1);
            }

            // Polarization function Pi33THpm from Hpm contribution
            double Pi33THpm(double mHpm2, double T) {
                return 0.5 * square(g) * Ijb(mHpm2, T, 1);
            }

            // Polarization function Pi33TW from W contribution
            double Pi33TW(double mW2L, double mW2T, double T) {
                return 2.0 * square(g) * PiTV(mW2L, mW2T, T, 1.0, 1.0);
            }

            // Polarization function sum Pi33T4
            double Pi33T4(double T, const std::vector<double>& MB2) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mW2T = MB2[8];
                
                double Pisum = 0.0;
                Pisum += Pi33Th(mh2, T);
                Pisum += Pi33TG(mG2, T);
                Pisum += Pi33TA(mA2, T);
                Pisum += Pi33TH(mH2, T);
                Pisum += Pi33THpm(mHpm2, T);
                Pisum += Pi33TW(mW2L, mW2T, T);
                
                return Pisum;
            }

            // ======================== Gauge bosons (33) three-points vertex self-energy (Longitudinal) =========================
            
            // Polarization function Pi33Lff from fermion contribution
            double Pi33Lff(double mt2, double T) {
                return 0.75 * square(g) * (-4.0 * DTIjf(mt2, T, 1) - 28.0 * DTIjf(0.0, T, 1) + 2.0 * mt2 * Ijf(mt2, T, 2) + mt2 * UV_term(mt2, -1.0, 2));
            }

            // Polarization function Pi33LhG from h and G contributions
            double Pi33LhG(double mh2, double mG2, double T) {
                return PiWLhG(mh2, mG2, T);
            }

            // Polarization function Pi33LGG from G and G contributions
            double Pi33LGG(double mG2, double T) {
                return PiWLGG(mG2, T);
            }

            // Polarization function Pi33LHpHm from Hp and Hm contributions
            double Pi33LHpHm(double mHpm2, double T) {
                return PiWLGG(mHpm2, T);
            }

            // Polarization function Pi33LAH from A and H contributions
            double Pi33LAH(double mA2, double mH2, double T) {
                return 0.5 * square(g) * PiLSS(mA2, mH2, T);
            }

            // Polarization function Pi33LWW from W and W contributions
            double Pi33LWW(double mW2L, double mW2T, double T) {
                return 2.0 * square(g) * (-Ijb(mW2L, T, 1) + 2.0 * DTIjb(mW2L, T, 1) - 2.0 * Ijb(mW2T, T, 1) + 4.0 * DTIjb(mW2T, T, 1) - UV_term(mW2T, -2.0, 1));
            }

            // Polarization function Pi33LhZ from h and Z contributions
            double Pi33LhZ(double mh2, double mZ2L, double T, double phi, double swL) {
                swL = std::clamp(swL, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                return -0.25 * square(square(g) * cwL + g * gp * swL) * square(phi) * PiLSV(mh2, mZ2L, T);
            }

            // Polarization function Pi33Lhga from h and photon contributions
            double Pi33Lhga(double mh2, double mga2L, double T, double phi, double swL) {
                swL = std::clamp(swL, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                return -0.25 * square(square(g) * swL - g * gp * cwL) * square(phi) * PiLSV(mh2, mga2L, T);
            }

            // Polarization function Pi33Lcc from ghost contribution
            double Pi33Lcc(double T) {
                return -square(g) * Ijb(0.0, T, 1);
            }

            // Polarization function sum Pi33L3
            double Pi33L3(double T, const std::vector<double>& MB2, double mt2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double mW2T = MB2[8];
                double swL = MB2[11];
                
                double Pisum = 0.0;
                Pisum += Pi33Lff(mt2, T);
                Pisum += Pi33LhG(mh2, mG2, T);
                Pisum += Pi33LGG(mG2, T);
                Pisum += Pi33LHpHm(mHpm2, T);
                Pisum += Pi33LAH(mA2, mH2, T);
                Pisum += Pi33LWW(mW2L, mW2T, T);
                Pisum += Pi33LhZ(mh2, mZ2L, T, phi, swL);
                Pisum += Pi33Lhga(mh2, mga2L, T, phi, swL);
                Pisum += Pi33Lcc(T);
                
                return Pisum;
            }

            // ======================== Gauge bosons (33) four-points vertex self-energy (Longitudinal) =========================
            
            // Polarization function Pi33Lh from h contribution
            double Pi33Lh(double mh2, double T) {
                return 0.25 * square(g) * Ijb(mh2, T, 1);
            }

            // Polarization function Pi33LG from G contribution
            double Pi33LG(double mG2, double T) {
                return 0.75 * square(g) * Ijb(mG2, T, 1);
            }

            // Polarization function Pi33LA from A contribution
            double Pi33LA(double mA2, double T) {
                return 0.25 * square(g) * Ijb(mA2, T, 1);
            }

            // Polarization function Pi33LH from H contribution
            double Pi33LH(double mH2, double T) {
                return 0.25 * square(g) * Ijb(mH2, T, 1);
            }

            // Polarization function Pi33LHpm from Hpm contribution
            double Pi33LHpm(double mHpm2, double T) {
                return 0.5 * square(g) * Ijb(mHpm2, T, 1);
            }

            // Polarization function Pi33LW from W contribution
            double Pi33LW(double mW2L, double mW2T, double T) {
                return 2.0 * PiWLW(mW2L, mW2T, T);
            }

            // Polarization function sum Pi33L4
            double Pi33L4(double T, const std::vector<double>& MB2) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mW2T = MB2[8];
                
                double Pisum = 0.0;
                Pisum += Pi33Lh(mh2, T);
                Pisum += Pi33LG(mG2, T);
                Pisum += Pi33LA(mA2, T);
                Pisum += Pi33LH(mH2, T);
                Pisum += Pi33LHpm(mHpm2, T);
                Pisum += Pi33LW(mW2L, mW2T, T);
                
                return Pisum;
            }

            // ======================== Gauge bosons (44) three-points vertex self-energy (Transverse) =========================
            
            // Polarization function Pi44Tff from fermion contribution
            double Pi44Tff(double mt2, double T) {
                double Temp_term = -34.0/3.0 * (Ijf(mt2, T, 1) - 2.0 * DTIjf(mt2, T, 1)) + 34.0/3.0 * (-Ijf(mt2, T, 1) + mt2 * Ijf(mt2, T, 2)) - 16.0 * mt2 * Ijf(mt2, T, 2);
                double UV_term_val = -34.0/3.0 * UV_term(mt2, 1.0/6.0, 1) + 34.0/3.0 * (-UV_term(mt2, -11.0/6.0, 1) + mt2 * UV_term(mt2, -11.0/6.0, 2)) - 4.0 * mt2 * UV_term(mt2, -2.0, 2);
                return square(gp)/12.0 * (Temp_term + UV_term_val);
            }

            // Polarization function Pi44ThG from h and G contributions
            double Pi44ThG(double mh2, double mG2, double T) {
                return square(gp)/3.0 * PiTSS(mh2, mG2, T);
            }

            // Polarization function Pi44TAH from A and H contributions
            double Pi44TAH(double mA2, double mH2, double T) {
                return square(gp)/3.0 * PiTSS(mA2, mH2, T);
            }

            // Polarization function Pi44THpHm from Hp and Hm contributions
            double Pi44THpHm(double mHpm2, double T) {
                return square(gp)/3.0 * PiTGG(mHpm2, T);
            }

            // Polarization function Pi44TGG from G and G contributions
            double Pi44TGG(double mG2, double T) {
                return square(gp)/3.0 * PiTGG(mG2, T);
            }

            // Polarization function Pi44TGW from G and W contributions
            double Pi44TGW(double mG2, double mW2L, double mW2T, double T, double phi) {
                return -square(g) * square(gp) / 6.0 * square(phi) * PiTSV(mG2, mW2L, mW2T, T, 1.0, 1.0);
            }

            // Polarization function Pi44ThZ from h and Z contributions
            double Pi44ThZ(double mh2, double mZ2L, double mZ2T, double T, double phi, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double gL = square(square(gp) * swL + g * gp * cwL);
                double gT = square(square(gp) * swT + g * gp * cwT);
                return -square(phi) / 12.0 * PiTSV(mh2, mZ2L, mZ2T, T, gL, gT);
            }

            // Polarization function Pi44Thga from h and photon contributions
            double Pi44Thga(double mh2, double mga2L, double mga2T, double T, double phi, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double gL = square(square(gp) * cwL - g * gp * swL);
                double gT = square(square(gp) * cwT - g * gp * swT);
                return -square(phi) / 12.0 * PiTSV(mh2, mga2L, mga2T, T, gL, gT);
            }

            // Polarization function sum Pi44T3
            double Pi44T3(double T, const std::vector<double>& MB2, double mt2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double mW2T = MB2[8];
                double mZ2T = MB2[9];
                double mga2T = MB2[10];
                double swL = MB2[11];
                double swT = MB2[12];
                
                double Pisum = 0.0;
                Pisum += Pi44Tff(mt2, T);
                Pisum += Pi44ThG(mh2, mG2, T);
                Pisum += Pi44TGG(mG2, T);
                Pisum += Pi44TAH(mA2, mH2, T);
                Pisum += Pi44THpHm(mHpm2, T);
                Pisum += Pi44TGW(mG2, mW2L, mW2T, T, phi);
                Pisum += Pi44ThZ(mh2, mZ2L, mZ2T, T, phi, swL, swT);
                Pisum += Pi44Thga(mh2, mga2L, mga2T, T, phi, swL, swT);
                
                return Pisum;
            }

            // ======================== Gauge bosons (44) four-points vertex self-energy (Transverse) =========================
            
            // Polarization function Pi44Th from h contribution
            double Pi44Th(double mh2, double T) {
                return 0.25 * square(gp) * Ijb(mh2, T, 1);
            }

            // Polarization function Pi44TG from G contribution
            double Pi44TG(double mG2, double T) {
                return 0.75 * square(gp) * Ijb(mG2, T, 1);
            }

            // Polarization function Pi44TA from A contribution
            double Pi44TA(double mA2, double T) {
                return 0.25 * square(gp) * Ijb(mA2, T, 1);
            }

            // Polarization function Pi44TH from H contribution
            double Pi44TH(double mH2, double T) {
                return 0.25 * square(gp) * Ijb(mH2, T, 1);
            }

            // Polarization function Pi44THpm from Hpm contribution
            double Pi44THpm(double mHpm2, double T) {
                return 0.5 * square(gp) * Ijb(mHpm2, T, 1);
            }

            // Polarization function sum Pi44T4
            double Pi44T4(double T, const std::vector<double>& MB2) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                
                double Pisum = 0.0;
                Pisum += Pi44Th(mh2, T);
                Pisum += Pi44TG(mG2, T);
                Pisum += Pi44TA(mA2, T);
                Pisum += Pi44TH(mH2, T);
                Pisum += Pi44THpm(mHpm2, T);
                
                return Pisum;
            }            

            // ======================== Gauge bosons (44) three-points vertex self-energy (Longitudinal) =========================
            
            // Polarization function Pi44Lff from fermion contribution
            double Pi44Lff(double mt2, double T) {
                return square(gp) / 12.0 * (-68.0 * DTIjf(mt2, T, 1) + 18.0 * mt2 * Ijf(mt2, T, 2) - 412.0 * Ijf(0.0, T, 1) + 9.0 * mt2 * UV_term(mt2, -1.0, 2));
            }

            // Polarization function Pi44LhG from h and G contributions
            double Pi44LhG(double mh2, double mG2, double T) {
                return 0.5 * square(gp) * PiLSS(mh2, mG2, T);
            }

            // Polarization function Pi44LAH from A and H contributions
            double Pi44LAH(double mA2, double mH2, double T) {
                return 0.5 * square(gp) * PiLSS(mA2, mH2, T);
            }

            // Polarization function Pi44LGG from G and G contributions
            double Pi44LGG(double mG2, double T) {
                return -0.5 * square(gp) * (Ijb(mG2, T, 1) - 2.0 * DTIjb(mG2, T, 1));
            }

            // Polarization function Pi44LHpHm from Hp and Hm contributions
            double Pi44LHpHm(double mHpm2, double T) {
                return -0.5 * square(gp) * (Ijb(mHpm2, T, 1) - 2.0 * DTIjb(mHpm2, T, 1));
            }

            // Polarization function Pi44LGW from G and W contributions
            double Pi44LGW(double mG2, double mW2L, double T, double phi) {
                return -0.5 * square(g) * square(gp) * square(phi) * PiLSV(mG2, mW2L, T);
            }

            // Polarization function Pi44LhZ from h and Z contributions
            double Pi44LhZ(double mh2, double mZ2L, double T, double phi, double swL) {
                swL = std::clamp(swL, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                return -0.25 * square(square(gp) * swL + g * gp * cwL) * square(phi) * PiLSV(mh2, mZ2L, T);
            }

            // Polarization function Pi44Lhga from h and photon contributions
            double Pi44Lhga(double mh2, double mga2L, double T, double phi, double swL) {
                swL = std::clamp(swL, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                return -0.25 * square(square(gp) * cwL - g * gp * swL) * square(phi) * PiLSV(mh2, mga2L, T);
            }

            // Polarization function sum Pi44L3
            double Pi44L3(double T, const std::vector<double>& MB2, double mt2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mW2L = MB2[5];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double swL = MB2[11];
                
                double Pisum = 0.0;
                Pisum += Pi44Lff(mt2, T);
                Pisum += Pi44LhG(mh2, mG2, T);
                Pisum += Pi44LGG(mG2, T);
                Pisum += Pi44LAH(mA2, mH2, T);
                Pisum += Pi44LHpHm(mHpm2, T);
                Pisum += Pi44LGW(mG2, mW2L, T, phi);
                Pisum += Pi44LhZ(mh2, mZ2L, T, phi, swL);
                Pisum += Pi44Lhga(mh2, mga2L, T, phi, swL);
                
                return Pisum;
            }

            // ======================== Gauge bosons (44) four-points vertex self-energy (Longitudinal) =========================
            
            // Polarization function Pi44Lh from h contribution
            double Pi44Lh(double mh2, double T) {
                return 0.25 * square(gp) * Ijb(mh2, T, 1);
            }

            // Polarization function Pi44LG from G contribution
            double Pi44LG(double mG2, double T) {
                return 0.75 * square(gp) * Ijb(mG2, T, 1);
            }

            // Polarization function Pi44LA from A contribution
            double Pi44LA(double mA2, double T) {
                return 0.25 * square(gp) * Ijb(mA2, T, 1);
            }

            // Polarization function Pi44LH from H contribution
            double Pi44LH(double mH2, double T) {
                return 0.25 * square(gp) * Ijb(mH2, T, 1);
            }

            // Polarization function Pi44LHpm from Hpm contribution
            double Pi44LHpm(double mHpm2, double T) {
                return 0.5 * square(gp) * Ijb(mHpm2, T, 1);
            }

            // Polarization function sum Pi44L4
            double Pi44L4(double T, const std::vector<double>& MB2) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                
                double Pisum = 0.0;
                Pisum += Pi44Lh(mh2, T);
                Pisum += Pi44LG(mG2, T);
                Pisum += Pi44LA(mA2, T);
                Pisum += Pi44LH(mH2, T);
                Pisum += Pi44LHpm(mHpm2, T);
                
                return Pisum;
            }

            // ======================== Gauge bosons (34) three-points vertex self-energy (Transverse) =========================
            
            // Polarization function Pi34Tff from fermion contribution
            double Pi34Tff(double mt2, double T) {
                double Temp_term = (2.0 * DTIjf(mt2, T, 1) - Ijf(mt2, T, 1)) - 12.0 * mt2 * Ijf(mt2, T, 2) - (Ijf(mt2, T, 1) - mt2 * Ijf(mt2, T, 2));
                double UV_term_val = -UV_term(mt2, 1.0/6.0, 1) - 3.0 * mt2 * UV_term(mt2, -2.0, 2) - (UV_term(mt2, -11.0/6.0, 1) - mt2 * UV_term(mt2, -11.0/6.0, 2));
                return g * gp / 6.0 * (Temp_term + UV_term_val);
            }

            // Polarization function Pi34ThG from h and G contributions
            double Pi34ThG(double mh2, double mG2, double T) {
                return -g * gp / 3.0 * PiTSS(mh2, mG2, T);
            }

            // Polarization function Pi34TAH from A and H contributions
            double Pi34TAH(double mA2, double mH2, double T) {
                return -g * gp / 3.0 * PiTSS(mA2, mH2, T);
            }

            // Polarization function Pi34THpHm from Hp and Hm contributions
            double Pi34THpHm(double mHpm2, double T) {
                return g * gp / 3.0 * PiTGG(mHpm2, T);
            }

            // Polarization function Pi34TGG from G and G contributions
            double Pi34TGG(double mG2, double T) {
                return g * gp / 3.0 * PiTGG(mG2, T);
            }

            // Polarization function Pi34ThZ from h and Z contributions
            double Pi34ThZ(double mh2, double mZ2L, double mZ2T, double T, double phi, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double gL = (square(gp) * swL + g * gp * cwL) * (square(g) * cwL + g * gp * swL);
                double gT = (square(gp) * swT + g * gp * cwT) * (square(g) * cwT + g * gp * swT);
                return square(phi) / 12.0 * PiTSV(mh2, mZ2L, mZ2T, T, gL, gT);
            }

            // Polarization function Pi34Thga from h and photon contributions
            double Pi34Thga(double mh2, double mga2L, double mga2T, double T, double phi, double swL, double swT) {
                swL = std::clamp(swL, 0.0, 1.0);
                swT = std::clamp(swT, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                double cwT = std::sqrt(1.0 - square(swT));
                double gL = (square(gp) * cwL - g * gp * swL) * (square(g) * swL - g * gp * cwL);
                double gT = (square(gp) * cwT - g * gp * swT) * (square(g) * swT - g * gp * cwT);
                return -square(phi) / 12.0 * PiTSV(mh2, mga2L, mga2T, T, gL, gT);
            }

            // Polarization function sum Pi34T3
            double Pi34T3(double T, const std::vector<double>& MB2, double mt2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double mZ2T = MB2[9];
                double mga2T = MB2[10];
                double swL = MB2[11];
                double swT = MB2[12];

                double Pisum = 0.0;
                Pisum += Pi34Tff(mt2, T);
                Pisum += Pi34ThG(mh2, mG2, T);
                Pisum += Pi34TGG(mG2, T);
                Pisum += Pi34TAH(mA2, mH2, T);
                Pisum += Pi34THpHm(mHpm2, T);
                Pisum += Pi34ThZ(mh2, mZ2L, mZ2T, T, phi, swL, swT);
                Pisum += Pi34Thga(mh2, mga2L, mga2T, T, phi, swL, swT);
                
                return Pisum;
            }

            // ======================== Gauge bosons (34) four-points vertex self-energy (Transverse) =========================
            
            // Polarization function Pi34Th from h contribution
            double Pi34Th(double mh2, double T) {
                return -0.25 * gp * g * Ijb(mh2, T, 1);
            }

            // Polarization function Pi34TG from G contribution
            double Pi34TG(double mG2, double T) {
                return 0.25 * gp * g * Ijb(mG2, T, 1);
            }

            // Polarization function Pi34TA from A contribution
            double Pi34TA(double mA2, double T) {
                return -0.25 * gp * g * Ijb(mA2, T, 1);
            }

            // Polarization function Pi34TH from H contribution
            double Pi34TH(double mH2, double T) {
                return -0.25 * gp * g * Ijb(mH2, T, 1);
            }

            // Polarization function Pi34THpm from Hpm contribution
            double Pi34THpm(double mHpm2, double T) {
                return 0.5 * gp * g * Ijb(mHpm2, T, 1);
            }

            // Polarization function sum Pi34T4
            double Pi34T4(double T, const std::vector<double>& MB2) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                
                double Pisum = 0.0;
                Pisum += Pi34Th(mh2, T);
                Pisum += Pi34TG(mG2, T);
                Pisum += Pi34TA(mA2, T);
                Pisum += Pi34TH(mH2, T);
                Pisum += Pi34THpm(mHpm2, T);
                
                return Pisum;
            }

            // ======================== Gauge bosons (34) three-points vertex self-energy (Longitudinal) =========================
            
            // Polarization function Pi34Lff from fermion contribution
            double Pi34Lff(double mt2, double T) {
                return 0.25 * g * gp * (-4.0 * DTIjf(mt2, T, 1) + 4.0 * Ijf(0.0, T, 1) - 6.0 * mt2 * Ijf(mt2, T, 2) - 3.0 * mt2 * UV_term(mt2, -1.0, 2));
            }

            // Polarization function Pi34LhG from h and G contributions
            double Pi34LhG(double mh2, double mG2, double T) {
                return -0.5 * g * gp * PiLSS(mh2, mG2, T);
            }

            // Polarization function Pi34LGG from G and G contributions
            double Pi34LGG(double mG2, double T) {
                return -0.5 * g * gp * (Ijb(mG2, T, 1) - 2.0 * DTIjb(mG2, T, 1));
            }

            // Polarization function Pi34LAH from A and H contributions
            double Pi34LAH(double mA2, double mH2, double T) {
                return -0.5 * g * gp * PiLSS(mA2, mH2, T);
            }

            // Polarization function Pi34LHpHm from Hp and Hm contributions
            double Pi34LHpHm(double mHpm2, double T) {
                return -0.5 * g * gp * (Ijb(mHpm2, T, 1) - 2.0 * DTIjb(mHpm2, T, 1));
            }

            // Polarization function Pi34LhZ from h and Z contributions
            double Pi34LhZ(double mh2, double mZ2L, double T, double phi, double swL) {
                swL = std::clamp(swL, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                return 0.25 * (square(gp) * swL + g * gp * cwL) * (square(g) * cwL + g * gp * swL) * square(phi) * PiLSV(mh2, mZ2L, T);
            }

            // Polarization function Pi34Lhga from h and photon contributions
            double Pi34Lhga(double mh2, double mga2L, double T, double phi, double swL) {
                swL = std::clamp(swL, 0.0, 1.0);
                double cwL = std::sqrt(1.0 - square(swL));
                return -0.25 * (square(gp) * cwL - g * gp * swL) * (square(g) * swL - g * gp * cwL) * square(phi) * PiLSV(mh2, mga2L, T);
            }

            // Polarization function sum Pi34L3
            double Pi34L3(double T, const std::vector<double>& MB2, double mt2, double phi) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                double mZ2L = MB2[6];
                double mga2L = MB2[7];
                double swL = MB2[11];
                
                double Pisum = 0.0;
                Pisum += Pi34Lff(mt2, T);
                Pisum += Pi34LhG(mh2, mG2, T);
                Pisum += Pi34LGG(mG2, T);
                Pisum += Pi34LAH(mA2, mH2, T);
                Pisum += Pi34LHpHm(mHpm2, T);
                Pisum += Pi34LhZ(mh2, mZ2L, T, phi, swL);
                Pisum += Pi34Lhga(mh2, mga2L, T, phi, swL);
                
                return Pisum;
            }

            // ======================== Gauge bosons (34) four-points vertex self-energy (Longitudinal) =========================
            
            // Polarization function Pi34Lh from h contribution
            double Pi34Lh(double mh2, double T) {
                return -0.25 * gp * g * Ijb(mh2, T, 1);
            }

            // Polarization function Pi34LG from G contribution
            double Pi34LG(double mG2, double T) {
                return 0.25 * gp * g * Ijb(mG2, T, 1);
            }

            // Polarization function Pi34LA from A contribution
            double Pi34LA(double mA2, double T) {
                return -0.25 * gp * g * Ijb(mA2, T, 1);
            }

            // Polarization function Pi34LH from H contribution
            double Pi34LH(double mH2, double T) {
                return -0.25 * gp * g * Ijb(mH2, T, 1);
            }

            // Polarization function Pi34LHpm from Hpm contribution
            double Pi34LHpm(double mHpm2, double T) {
                return 0.5 * gp * g * Ijb(mHpm2, T, 1);
            }

            // Polarization function sum Pi34L4
            double Pi34L4(double T, const std::vector<double>& MB2) {
                double mh2 = MB2[0];
                double mG2 = MB2[1];
                double mA2 = MB2[2];
                double mH2 = MB2[3];
                double mHpm2 = MB2[4];
                
                double Pisum = 0.0;
                Pisum += Pi34Lh(mh2, T);
                Pisum += Pi34LG(mG2, T);
                Pisum += Pi34LA(mA2, T);
                Pisum += Pi34LH(mH2, T);
                Pisum += Pi34LHpm(mHpm2, T);
                
                return Pisum;
            }


            /**
             * Calculate the loss value for the mass spectrum at a given background field value and temperature.
             * @param bosons Pair of vectors containing the mixing angles (size 2) and mass squared values of the bosons (size 11)
             * @param MB2 Array of mass squared values (size 13: 11 masses + swL + swT)
             * @param phi Background field value
             * @param T Temperature
             * @return The loss value (norm of the difference between input and calculated masses)
             * and the new mass spectrum (vector of size 13: 11 masses + swL + swT)
             */
            std::pair<double, std::vector<double>> calc_loss(const std::pair<std::vector<double>, std::vector<double>>& bosons, const std::vector<double>& x, double phi, double T) {
                // Tree-level boson mass spectrum and conterterms
                const auto& [mass_squared_values, mixing_angles] = bosons;
                const double mh2_init = mass_squared_values[0];
                const double mG2_init = mass_squared_values[1];
                const double mA2_init = mass_squared_values[2];
                const double mH2_init = mass_squared_values[3];
                const double mHpm2_init = mass_squared_values[4];
                const double mW2_init = 0.25 * square(g) * square(phi);
                const double m332_init = 0.25 * square(g) * square(phi);
                const double m442_init = 0.25 * square(gp) * square(phi);
                const double m342_init = -0.25 * g * gp * square(phi);
                const double mt2_init = 0.5 * square(yt) * square(phi);

                const double swL_init = mixing_angles[0];
                const double swT_init = mixing_angles[1];


                // h mass
                double Mh2 = mh2_init + Pih4(T, x) + Pih3(T, x, mt2_init, phi);
                
                // G mass
                double MG2 = mG2_init + PiG4(T, x) + PiG3(T, x, mt2_init, phi);
                
                // A mass
                double MA2 = mA2_init + PiA4(T, x) + PiA3(T, x, phi);
                
                // H mass
                double MH2 = mH2_init + PiH4(T, x) + PiH3(T, x, phi);
                
                // Hpm mass
                double MHpm2 = mHpm2_init + PiHpm4(T, x) + PiHpm3(T, x, phi);
                
                // Gauge boson mass (Longitudinal): mga2L, mZ2L, swL
                double M332L = m332_init + Pi33L4(T, x) + Pi33L3(T, x, mt2_init, phi);
                double M442L = m442_init + Pi44L4(T, x) + Pi44L3(T, x, mt2_init, phi);
                double M342L = m342_init + Pi34L4(T, x) + Pi34L3(T, x, mt2_init, phi);
                
                // Solve for Z and photon masses (Longitudinal)
                Eigen::Matrix2d M2Lmatrix;
                M2Lmatrix << M332L, M342L,
                             M342L, M442L;
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solverL(M2Lmatrix);
                double Mga2L = solverL.eigenvalues()[0];
                double MZ2L = solverL.eigenvalues()[1];
                double sL = solverL.eigenvectors()(0,0);

                // Gauge boson mass (Transverse): mga2T, mZ2T, swT
                double M332T = m332_init + Pi33T4(T, x) + Pi33T3(T, x, mt2_init, phi);
                double M442T = m442_init + Pi44T4(T, x) + Pi44T3(T, x, mt2_init, phi);
                double M342T = m342_init + Pi34T4(T, x) + Pi34T3(T, x, mt2_init, phi);
                
                // Solve for Z and photon masses (Transverse)
                Eigen::Matrix2d M2Tmatrix;
                M2Tmatrix << M332T, M342T,
                             M342T, M442T;
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solverT(M2Tmatrix);
                double Mga2T = solverT.eigenvalues()[0];
                double MZ2T = solverT.eigenvalues()[1];
                double sT = solverT.eigenvectors()(0,0);

                // Gauge boson mass (W): mW2L, mW2T
                double MW2L = mW2_init + PiWL4(T, x) + PiWL3(T, x, mt2_init, phi);
                double MW2T = mW2_init + PiWT4(T, x) + PiWT3(T, x, mt2_init, phi);
                
                // Construct the calculated mass squared array
                std::vector<double> x_new = {Mh2, MG2, MA2, MH2, MHpm2, MW2L, MZ2L, Mga2L, MW2T, MZ2T, Mga2T, sL, sT};
                
                // Calculate the difference
                Eigen::VectorXd deltaMB2(13);
                for (int i = 0; i < 13; ++i) {
                    if (x[i] <= 1e-3) {
                        deltaMB2[i] = std::abs(x[i] - x_new[i]);
                    } else {
                        deltaMB2[i] = std::abs((x[i] - x_new[i]) / x[i]);
                    }
                }
                
                return make_pair(deltaMB2.sum(), x_new);
            }


            gapEqResult solve_gap_equations(double phi, double T, double tol, const std::pair<std::vector<double>, std::vector<double>>& bosons_bare, const std::pair<std::vector<double>, std::vector<double>>& bosons_init, int max_iter = 300) {

                gapEqResult result;
                double loss = 1e10; // Initialize with a large loss value
                std::vector<double> prev_prev(13); // previous previous mass spectrum
                std::vector<double> prev(13); // previous mass spectrum
                for (int i = 0; i < 11; ++i) {
                    prev_prev[i] = bosons_init.first[i];
                    prev[i] = bosons_init.first[i];
                }
                prev_prev[11] = bosons_init.second[0];
                prev_prev[12] = bosons_init.second[1];
                prev[11] = bosons_init.second[0];
                prev[12] = bosons_init.second[1];

                for (int iter = 0; iter < max_iter; ++iter) {
                    auto [loss_new, x_new] = calc_loss(bosons_bare, prev, phi, T);

                    if (loss_new < tol) {
                        result.x = x_new;
                        result.loss = loss_new;
                        result.success = true;
                        result.message = "Converged to specified precision after " + std::to_string(iter + 1) + " iterations";
                        return result;
                    }
                    else {
                        // checking the loss value if decreased 
                        if (loss_new > loss) { // improve the iteration
                            if (iter < 50) loss = loss_new;

                            for (int i = 0; i < 13; ++i) {
                                prev[i] = prev_prev[i] + 0.5 * (prev[i] - prev_prev[i]); // simple extrapolation
                            } 
                        } else {
                            prev_prev = prev;
                            prev = x_new;
                            loss = loss_new;
                        }
                    }
                }
                result.x = prev_prev; // Return the last mass spectrum before the final iteration
                result.loss = loss;
                result.success = false;
                result.message = "Failed to converge after maximum iterations";
                return result;
            }




    };




}

#endif