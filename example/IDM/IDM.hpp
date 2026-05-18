#ifndef POTENTIAL_IDM_HPP_INCLUDED
#define POTENTIAL_IDM_HPP_INCLUDED

/*
   IDM
*/

#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <future>
#include <atomic>
#include <mutex>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <Eigen/Dense>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_interp2d.h>

#include "potential.hpp"
#include "SelfEnergy.hpp"

namespace EffectivePotential {

    enum class ResummationScheme {
        None,
        Parwani,
        ArnoldEspinosa,
        PartialDressing,
        DolanJackiw
    };

    enum class ThermalMassScheme {
        Tree,
        HighT,
        Exact,
        CTterm
    };

    struct IDMParameters {
        // constants
        double v0;
        double mh;
        double mt;
        double yt;
        double g ;
        double gp ;
        double gstar ;
        // input parameters
        double lam2 ;
        double lamL ;
        double mA;
        double mH ;
        double mHpm ;
        //others
        double lam1 ;
        double mu1sq ;
        double mu2sq ;
        double lamm ;
        double lamp ;
        double lam3 ;
        double lam4 ;
        double lam5 ;
    };

    struct MassSplines {
        alglib::spline2dinterpolant Mh2;
        alglib::spline2dinterpolant MG2;
        alglib::spline2dinterpolant MA2;
        alglib::spline2dinterpolant MH2;
        alglib::spline2dinterpolant MHpm2;
        alglib::spline2dinterpolant MW2L;
        alglib::spline2dinterpolant MZ2L;
        alglib::spline2dinterpolant Mga2L;
        alglib::spline2dinterpolant MW2T;
        alglib::spline2dinterpolant MZ2T;
        alglib::spline2dinterpolant Mga2T;
        alglib::spline2dinterpolant swL;
        alglib::spline2dinterpolant swT;
    };
  
    struct ConterTermParameters {
        double delta_mu1sq;
        double delta_lam1;
        double delta_mu2sq;
        double delta_lamm;
        double delta_lamp;
        double delta_lam3;
    };

    class IDM : public Potential {
    public:
        // Initialize the parameters of the model. The input parameters are lam2, lamL, mA, mH, mHpm, and a string paramNumber that is used to label the parameter point in the output files. The other parameters are derived from these input parameters.
        void init_params(double lam2_, double lamL_, double mA_, double mH_, double mHpm_, std::string paramNumber_) {
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

            paramNumber = paramNumber_;

        }

        IDMParameters get_params() const {
            IDMParameters params;
            params.v0 = v0;
            params.mh = mh;
            params.mt = mt;
            params.yt = yt;
            params.g = g;
            params.gp = gp;
            params.gstar = gstar;
            params.lam2 = lam2;
            params.lamL = lamL;
            params.mA = mA;
            params.mH = mH;
            params.mHpm = mHpm;
            params.lam1 = lam1;
            params.mu1sq = mu1sq;
            params.mu2sq = mu2sq;
            params.lamm = lamm;
            params.lamp = lamp;
            params.lam3 = lam3;
            params.lam4 = lam4;
            params.lam5 = lam5;
            return params;
        }
        
        ConterTermParameters get_counterterm_params() const {
            ConterTermParameters ct_params;
            ct_params.delta_mu1sq = delta_mu1sq;
            ct_params.delta_lam1 = delta_lam1;
            ct_params.delta_mu2sq = delta_mu2sq;
            ct_params.delta_lamm = delta_lamm;
            ct_params.delta_lamp = delta_lamp;
            ct_params.delta_lam3 = delta_lam3;
            return ct_params;
        }

        MassSplines get_mass_splines() const {
            return mass_splines_;
        }

        size_t get_n_scalars() const override { return 1; }
        
        bool forbidden(Eigen::VectorXd X) const override { return X[0] < -0.1; } 
        
        std::vector<Eigen::VectorXd> get_low_t_phases() const override { 
            Eigen::VectorXd phase(1);
            phase[0] = v0;
            return {phase};
        }
        
        double V0(Eigen::VectorXd X) const {
            double phi = X[0];
            return 0.5 * mu1sq * square(phi) + 0.125 * lam1 * pow_4(phi);
        }
        
        double V1CT(Eigen::VectorXd X) const {
            double phi = X[0];
            return 0.5 * delta_mu1sq * square(phi) + 0.125 * delta_lam1 * pow_4(phi);
        }

        std::pair<std::vector<double>, std::vector<double>> boson_massSq(Eigen::VectorXd X, double T, ThermalMassScheme ms) const {
            double phi = X[0];

            // tree-level boson mass spectrum
            double mh2 = mu1sq + 1.5 * lam1 * square(phi);
            double mG2 = mu1sq + 0.5 * lam1 * square(phi);
            double mA2 = mu2sq + 0.5 * lamm * square(phi);
            double mH2 = mu2sq + 0.5 * lamp * square(phi);
            double mHpm2 = mu2sq + 0.5 * lam3 * square(phi);
            double mW2 = 0.25 * square(g) * square(phi);
            double mB2 = 0.25 * square(gp) * square(phi);
            double mW3B2 = -0.25 * g * gp * square(phi);

            switch (ms) {
                case ThermalMassScheme::Tree: {
                    Eigen::Matrix2d M2matrix;
                    M2matrix << mW2, mW3B2,
                                mW3B2, mB2;
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(M2matrix);
                    double mga2 = solver.eigenvalues()[0];
                    double mZ2 = solver.eigenvalues()[1];
                    double sw = solver.eigenvectors()(0,0);
                    std::vector<double> mSquares = {mh2, mG2, mA2, mH2, mHpm2, mW2, mZ2, mga2, mW2, mZ2, mga2};
                    std::vector<double> mixing_angles = {sw, sw} ;
                    return std::make_pair(mSquares, mixing_angles);
                }
                case ThermalMassScheme::CTterm: { 
                    mh2 += delta_mu1sq + 1.5 * delta_lam1 * square(phi);
                    mG2 += delta_mu1sq + 0.5 * delta_lam1 * square(phi);
                    mA2 += delta_mu2sq + 0.5 * delta_lamm * square(phi);
                    mH2 += delta_mu2sq + 0.5 * delta_lamp * square(phi);
                    mHpm2 += delta_mu2sq + 0.5 * delta_lam3 * square(phi);
                    Eigen::Matrix2d M2matrix;
                    M2matrix << mW2, mW3B2,
                                mW3B2, mB2;
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(M2matrix);
                    double mga2 = solver.eigenvalues()[0];
                    double mZ2 = solver.eigenvalues()[1];
                    double sw = solver.eigenvectors()(0,0);
                    std::vector<double> mSquares = {mh2, mG2, mA2, mH2, mHpm2, mW2, mZ2, mga2, mW2, mZ2, mga2};
                    std::vector<double> mixing_angles = {sw, sw} ;
                    return std::make_pair(mSquares, mixing_angles);
                }
                case ThermalMassScheme::HighT: {
                    double Pih = square(T) * (9.0/4.0 * square(g) + 3.0/4.0 * square(gp) + 3 * lam1 + 2 * lam3 + lam4 + 3 * square(yt))/12.0;
                    double PiG = Pih;
                    double PiA = square(T) * (9.0/4.0 * square(g) + 3.0/4.0 * square(gp) + 3 * lam2 + 2 * lam3 + lam4 )/12.0;
                    double PiH = PiA;
                    double PiHpm = PiA;
                    double PiLWpm = 2 * square(g) * square(T);
                    double PiTWpm = 0;
                    double PiLW3 = 2 * square(g) * square(T);
                    double PiTW3 = 0;
                    double PiLB = 2 * square(gp) * square(T);
                    double PiTB = 0;
                    double PiLW3B = 0;
                    double PiTW3B = 0;
                    double Mh2 = mh2 + Pih;
                    double MG2 = mG2 + PiG;
                    double MA2 = mA2 + PiA;
                    double MH2 = mH2 + PiH;
                    double MHpm2 = mHpm2 + PiHpm;
                    double MW2L = mW2 + PiLWpm;
                    double MW2T = mW2 + PiTWpm;
                    double MW32L = mW2 + PiLW3;
                    double MW32T = mW2 + PiTW3;
                    double MB2L = mB2 + PiLB;
                    double MB2T = mB2 + PiTB;
                    double MW3B2L = mW3B2 + PiLW3B;
                    double MW3B2T = mW3B2 + PiTW3B;
                    Eigen::Matrix2d M2Lmatrix;
                    M2Lmatrix << MW32L, MW3B2L,
                                MW3B2L, MB2L;
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solverL(M2Lmatrix);
                    double Mga2L = solverL.eigenvalues()[0];
                    double MZ2L = solverL.eigenvalues()[1];
                    double swL = solverL.eigenvectors()(0,0);
                    Eigen::Matrix2d M2Tmatrix;
                    M2Tmatrix << MW32T, MW3B2T,
                                MW3B2T, MB2T;
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solverT(M2Tmatrix);
                    double Mga2T = solverT.eigenvalues()[0];
                    double MZ2T = solverT.eigenvalues()[1];
                    double swT = solverT.eigenvectors()(0,0);
                    std::vector<double> MSquares = {Mh2, MG2, MA2, MH2, MHpm2, MW2L, MZ2L, Mga2L, MW2T, MZ2T, Mga2T};
                    std::vector<double> mixing_angles = {swL, swT} ;
                    return std::make_pair(MSquares, mixing_angles);
                }
                case ThermalMassScheme::Exact: { 
                    // exact thermal masses, as it requires solving gap equations self-consistently. 
                    double Mh2 = alglib::spline2dcalc(mass_splines_.Mh2, phi, T);
                    double MG2 = alglib::spline2dcalc(mass_splines_.MG2, phi, T);
                    double MA2 = alglib::spline2dcalc(mass_splines_.MA2, phi, T);
                    double MH2 = alglib::spline2dcalc(mass_splines_.MH2, phi, T);
                    double MHpm2 = alglib::spline2dcalc(mass_splines_.MHpm2, phi, T);
                    double MW2L = alglib::spline2dcalc(mass_splines_.MW2L, phi, T);
                    double MZ2L = alglib::spline2dcalc(mass_splines_.MZ2L, phi, T);
                    double Mga2L = alglib::spline2dcalc(mass_splines_.Mga2L, phi, T);
                    double MW2T = alglib::spline2dcalc(mass_splines_.MW2T, phi, T);
                    double MZ2T = alglib::spline2dcalc(mass_splines_.MZ2T, phi, T);
                    double Mga2T = alglib::spline2dcalc(mass_splines_.Mga2T, phi, T);
                    double swL = alglib::spline2dcalc(mass_splines_.swL, phi, T);
                    double swT = alglib::spline2dcalc(mass_splines_.swT, phi, T);
                    std::vector<double> MSquares = {Mh2, MG2, MA2, MH2, MHpm2, MW2L, MZ2L, Mga2L, MW2T, MZ2T, Mga2T};
                    std::vector<double> mixing_angles = {swL, swT};
                    return std::make_pair(MSquares, mixing_angles);
                }
                default:
                    throw std::invalid_argument("unknown ThermalMassScheme");
            }
        }

        //Calculate the fermion particle spectrum.
        std::vector<double> boson_dofs() const {
            std::vector<double> dofs = {1, 3, 1, 1, 2, 2, 1, 1, 4, 2, 2};
            return dofs;
        }

        std::vector<double> fermion_massSq(Eigen::VectorXd X) const {
            double phi = X[0];
            // Top quark
            double mt2 = 0.5 * square(yt) * square(phi);
            std::vector<double> mSquare = {mt2};

            return mSquare;
        }

        std::vector<double> fermion_dofs() const {
            std::vector<double> dofs = {12}; // 3 colors * 4 components for Dirac fermion
            return dofs;
        }

        void set_resummation_scheme(ResummationScheme scheme) {
            ResumScheme = scheme;
        }

         // One-loop zero-temperature correction to potential. 
        double V1(std::pair<std::vector<double>, std::vector<double>>& bosons, std::vector<double>& fermions) const {
            double result = 0;
            std::vector<double> nf = fermion_dofs();
            std::vector<double> mf2 = fermions;
            for (int i = 0; i < nf.size(); i++) {
                result += 0.5 * nf[i] * zeroT_term_deriv(mf2[i], 0);
            }

            std::vector<double> nb = boson_dofs();
            std::vector<double> mb2 = bosons.first;
            for (int i = 0; i < nb.size(); i++) {
                result -= 0.5 * nb[i] * zeroT_term_deriv(mb2[i], 0);
            }
            result -= 0.5 * UV_term(mb2[8], -4, 0);
            result -= 0.5 * UV_term(mb2[9], -2, 0);
            result -= 0.5 * UV_term(mb2[10], -2, 0);
            return result;
        }

        // One-loop finite-temperature correction to potential.
        double V1T(std::pair<std::vector<double>, std::vector<double>>& bosons, std::vector<double>& fermions, double T) const {
            double correction = 0;

            if (T <= std::numeric_limits<double>::min())
                return 0.;

            const auto nb = boson_dofs();
            const auto mb2 = bosons.first;
            const auto nf = fermion_dofs();
            const auto mf2 = fermions;

            if (nb.size() != mb2.size()) {
                throw std::runtime_error("Boson dofs and masses do not match");
            }

            if (nf.size() != mf2.size()) {
                throw std::runtime_error("Fermion dofs and masses do not match");
            }

            // boson correction
            for (size_t i = 0; i < mb2.size(); ++i) {
                correction += nb[i] * J_B(mb2[i] / square(T), 0);
            }

            // fermion correction
            for (size_t i = 0; i < mf2.size(); ++i) {
                correction += nf[i] * J_F(mf2[i] / square(T), 0);
            }

            return correction * pow_4(T) / (2. * square(M_PI));
        }

        //Calculates the daisy resummation (AE resummation scheme).
        double Vdaisy(Eigen::VectorXd X, double T) const { 
            double phi = X[0];
            auto [Mb2_HighT, Sw_HighT] = boson_massSq(X, T, ThermalMassScheme::HighT);
            auto [mb2_tree, sw_tree] = boson_massSq(X, T, ThermalMassScheme::Tree);
            const auto nb = boson_dofs();
            double correction = 0;
            for (size_t i = 0; i < nb.size(); ++i) {
                correction -= nb[i] * (std::pow(std::max(Mb2_HighT[i], 0.0), 1.5) - std::pow(std::max(mb2_tree[i], 0.0), 1.5) );
            }
            return T * correction / (12.0 * M_PI);
        }

        /**
         * Calculates the derivative of the one-loop potential
         * with respect to the field variables.
         */
        double dV1_dX(Eigen::VectorXd X, double T, ThermalMassScheme ms) const {
            double phi = X[0];
            double y = 0.0;

            // Fermion contribution (top quark)
            std::vector<double> mf2 = fermion_massSq(X);  
            std::vector<double> nf = fermion_dofs();  
            std::vector<double> ct = {square(yt) * phi}; 
            for (size_t i = 0; i < nf.size(); ++i) {
                y -= 0.5 * nf[i] * ct[i] * Ijf(mf2[i], T, 1);
            }    

            // Boson contribution
            auto bosons = boson_massSq(X, T, ms);
            auto nb = boson_dofs();
            const auto& mb2 = bosons.first;

            std::vector<double> cb(11);
            cb[0] = 3.0 * lam1 * phi;           // d(mh2)/d(phi)
            cb[1] = lam1 * phi;                  // d(mG2)/d(phi)
            cb[2] = lamm * phi;                  // d(mA2)/d(phi)
            cb[3] = lamp * phi;                  // d(mH2)/d(phi)
            cb[4] = lam3 * phi;                  // d(mHpm2)/d(phi)
            cb[5] = 0.5 * square(g) * phi;      // d(mW2L)/d(phi)
            cb[6] = 0.5 * (square(g) + square(gp)) * phi;  // d(mZ2L)/d(phi)
            cb[7] = 0.0;                         // d(mga2L)/d(phi) = 0
            cb[8] = 0.5 * square(g) * phi;      // d(mW2T)/d(phi)
            cb[9] = 0.5 * (square(g) + square(gp)) * phi;  // d(mZ2T)/d(phi)
            cb[10] = 0.0;                        // d(mga2T)/d(phi) = 0

            // Boson contribution: sum(0.5 * nb * cb * Ijb(M2, T, 1))
            for (size_t i = 0; i < mb2.size(); ++i) {
                y += 0.5 * nb[i] * cb[i] * Ijb(mb2[i], T, 1);
            }

            // UV terms for transverse modes (indices 8, 9, 10 correspond to mW2T, mZ2T, mga2T)
            y += cb[8] * UV_term(mb2[8], -2.0, 1);
            y += 0.5 * cb[9] * UV_term(mb2[9], -2.0, 1);
            y += 0.5 * cb[10] * UV_term(mb2[10], -2.0, 1);

            return y;
        }

        Eigen::VectorXd dV_dx(Eigen::VectorXd X, double T) const override {
            switch (ResumScheme) {
                case ResummationScheme::None:
                case ResummationScheme::Parwani:
                case ResummationScheme::ArnoldEspinosa:
                case ResummationScheme::DolanJackiw:
                    return Potential::dV_dx(X, T);
                case ResummationScheme::PartialDressing: {
                    double phi = X[0];
                    Eigen::VectorXd dVdX(1);
                    double dV0dX = mu1sq * phi + 0.5 * lam1 * cube(phi);
                    double dVCTdX = delta_mu1sq * phi + 0.5 * delta_lam1 * cube(phi);
                    double dV1dX = dV1_dX(X, T, ThermalMassScheme::Exact);
                    dVdX(0) = dV0dX + dVCTdX + dV1dX;
                    return dVdX;
                }
                default: 
                    throw std::invalid_argument("unknown ResummationScheme");
                
            }
           
        }

        Eigen::VectorXd d2V_dxdt(Eigen::VectorXd X, double T) const override {
            switch (ResumScheme) {
                case ResummationScheme::None:
                case ResummationScheme::Parwani:
                case ResummationScheme::ArnoldEspinosa:
                case ResummationScheme::DolanJackiw:
                    return Potential::d2V_dxdt(X, T);
                case ResummationScheme::PartialDressing: {
                    // Numerical derivative of the analytical first derivative with respect to temperature
                    // Mimics the implementation of Potential::dV_dx but uses dV1dX instead of V
                    
                    Eigen::VectorXd result = Eigen::VectorXd::Zero(X.size());
                    
                    for (int ii = 0; ii < X.size(); ++ii) {
                        for (int jj = 0; jj < n_h_xy.size(); ++jj) {
                            const double T_shifted = T + n_h_xy[jj] * h ;
                            result(ii) += dV1_dX(X, T_shifted, ThermalMassScheme::Exact) * coeff_xy[jj] / h;
                        }
                    }
                    
                    return result;
                }
                default: 
                    throw std::invalid_argument("unknown ResummationScheme");
            }
        }


        Eigen::MatrixXd d2V_dx2(Eigen::VectorXd X, double T) const override { 
            switch (ResumScheme) {
                case ResummationScheme::None:
                case ResummationScheme::Parwani:
                case ResummationScheme::ArnoldEspinosa:
                case ResummationScheme::DolanJackiw:
                    return Potential::d2V_dx2(X, T);
                case ResummationScheme::PartialDressing: {
                    // Numerical second derivative using analytical first derivative
                    // Mimics the implementation of Potential::d2V_dx2 but uses dV1_dX instead of V
                    
                    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(X.size(), X.size());
                    
                    for (int ii = 0; ii < X.size(); ++ii) {
                        Eigen::VectorXd X_shifted = X;
                        for (int jj = 0; jj < n_h_xy.size(); ++jj) {
                            X_shifted(ii) = X(ii) + n_h_xy[jj] * h;
                            hessian(ii, ii) += dV1_dX(X_shifted, T, ThermalMassScheme::Exact) * coeff_xy[jj] / h;
                        }
                    }
                    hessian(0, 0) +=  mu1sq + 1.5 * lam1 * square(X[0]);
                    hessian(0, 0) += delta_mu1sq + 1.5 * delta_lam1 * square(X[0]);
                    
                    return hessian;
                }
                default: 
                    throw std::invalid_argument("unknown ResummationScheme");
            }
        }


        double V(Eigen::VectorXd X, double T) const override {
            auto bosons_tree = boson_massSq(X, T, ThermalMassScheme::Tree);
            auto fermions = fermion_massSq(X);
            switch (ResumScheme) { 
                case ResummationScheme::None: {
                    return V0(X) + V1(bosons_tree, fermions) + V1CT(X) + V1T(bosons_tree, fermions, T);
                }
                case ResummationScheme::Parwani: {
                    auto bosons_highT = boson_massSq(X, T, ThermalMassScheme::HighT);
                    return V0(X) + V1(bosons_highT, fermions) + V1CT(X) + V1T(bosons_highT, fermions, T) ;
                }
                case ResummationScheme::ArnoldEspinosa: {
                    return V0(X) + V1(bosons_tree, fermions) + V1CT(X) + V1T(bosons_tree, fermions, T) + Vdaisy(X, T);
                }
                case ResummationScheme::DolanJackiw: {
                    auto bosons_highT = boson_massSq(X, T, ThermalMassScheme::HighT);
                    return V0(X) + V1(bosons_tree, fermions) + V1CT(X) + V1T(bosons_highT, fermions, T);
                }
                case ResummationScheme::PartialDressing: {
                    using boost::math::quadrature::gauss_kronrod;
            
                    double phi = X[0];
                    
                    // If phi equals reference point phi= 0, return 0
                    if (std::abs(phi) < std::numeric_limits<double>::min()) {
                        return 0.0;
                    }
                    
                    // Define the integrand as a lambda function
                    auto integrand = [&](double x) -> double {
                        Eigen::VectorXd X_point(1);
                        X_point[0] = x;
                        return dV1_dX(X_point, T, ThermalMassScheme::Exact);
                    };
                    
                    // Perform integration using Gauss-Kronrod quadrature
                    // 15-point rule, relative tolerance 1e-4, max iterations 50
                    double V1_integrated = gauss_kronrod<double, 15>::integrate(integrand, 0.0, phi, 50, 1e-4);
                    return V0(X) + V1CT(X) + V1_integrated;
                }
                default: 
                    throw std::invalid_argument("unknown ResummationScheme");
            }
        }


        void calc_conterterms() {
            SelfEnergy selfenergy(lam2, lamL, mA, mH, mHpm); 

            Eigen::VectorXd vev(1);
            vev[0] = v0;
            std::vector<double> MB2_vec(13);
            auto [Mb2, sw] = boson_massSq(vev, 0.0, ThermalMassScheme::Tree);
            for (int i = 0; i < 11; ++i) {
                MB2_vec[i] = Mb2[i];
            }
            MB2_vec[11] = sw[0];
            MB2_vec[12] = sw[1];
            double eq1 = - dV1_dX(vev, 0.0, ThermalMassScheme::Tree) / v0;
            double deltaZ = selfenergy.DPih3_Dp2(square(mh), MB2_vec, square(mt)) ;
            double se_term = selfenergy.Pih3Pdependent(.0, square(mh), MB2_vec, square(mt), v0) + selfenergy.Pih4(.0, MB2_vec);
            double eq2 = deltaZ * square(mh) - se_term;
            delta_lam1 =  (eq2 - eq1)  / square(v0);
            delta_mu1sq = (3.0*eq1 - eq2) / 2.0;

            delta_lamp = - Gamma_lamp();
            delta_mu2sq = - (selfenergy.PiH3(.0, MB2_vec, v0) + selfenergy.PiH4(.0, MB2_vec) + 0.5 * delta_lamp * square(v0));
            delta_lam3 = - 2.0 * (selfenergy.PiHpm3(.0, MB2_vec, v0) + selfenergy.PiHpm4(.0, MB2_vec) + delta_mu2sq) / square(v0);
            delta_lamm = - 2.0 * (selfenergy.PiA3(.0, MB2_vec, v0) + selfenergy.PiA4(.0, MB2_vec) + delta_mu2sq) / square(v0);
        }

        /**
         * Initialize mass splines by solving gap equations on a grid asynchronously.
         * Saves results to disk and loads them into spline interpolators.
         */
        void init_mass_splines () {
            double phimin = 0.0, phimax = 400.0;
            double Tmin = 0.0, Tmax = 300.0;
            int n_phi = 400, n_T = 300;
            std::string spline_data_path = "/home/dayun/data"; // Adjust path as needed
            std::filesystem::create_directories(spline_data_path);
            
            std::string M2_dat_path = spline_data_path + "/M2_" + paramNumber + ".txt";
            std::string bad_points_path = spline_data_path + "/bad_points_" + paramNumber + ".txt";

            const int n_mass = 13; // mh2, mG2, mA2, mH2, mHpm2, mW2L, mZ2L, mga2L, mW2T, mZ2T, mga2T, swL, swT
            const size_t total_points = static_cast<size_t>(n_phi) * n_T;
            
            // Flat storage: index = i_phi * n_T + i_T
            std::vector<double> yM_flat(total_points * n_mass, 0.0);
            //std::vector<bool> valid_points(total_points, false);
            std::vector<std::tuple<double, double, int, int>> bad_points_list;
            
            std::mutex bad_points_mutex;
            std::atomic<int> completed_points(0);
            auto start_time = std::chrono::high_resolution_clock::now();

            // Generate grid coordinates
            std::vector<double> xphi(n_phi);
            std::vector<double> xT(n_T);
            for (int i = 0; i < n_phi; ++i) xphi[i] = phimin + i * (phimax - phimin) / (n_phi - 1);
            for (int j = 0; j < n_T; ++j) xT[j] = Tmin + j * (Tmax - Tmin) / (n_T - 1);

            // Check if data exists
            if (std::filesystem::exists(M2_dat_path)) {
                load_mass_data(M2_dat_path, xphi, xT, yM_flat, n_phi, n_T, n_mass);
                std::cout << "Loaded mass data from " << M2_dat_path << std::endl;
            } else {
                std::cout << "Generating mass data on grid (" << n_phi << "x" << n_T << ")..." << std::endl;

                // Define the worker lambda to handle a range of grid points
                // Modified to accept start/end indices and references to grid coordinates
                auto worker = [this, n_mass, &yM_flat, &bad_points_list, &bad_points_mutex, &completed_points, total_points, start_time](
                    int start_idx, int end_idx, 
                    const std::vector<double>& xphi, const std::vector<double>& xT,
                    int n_T) {
                    
                    for (size_t idx = static_cast<size_t>(start_idx); idx < static_cast<size_t>(end_idx); ++idx) {
                        // Convert flat index back to 2D grid coordinates
                        int i = static_cast<int>(idx / n_T);
                        int j = static_cast<int>(idx % n_T);
                        
                        double phi = xphi[i];
                        double T = xT[j];

                        try {
                            // Solve gap equations
                            auto bosons_init = boson_massSq(Eigen::VectorXd::Constant(1, phi), T, ThermalMassScheme::Tree); // Initial guess
                            auto bosons_bare = boson_massSq(Eigen::VectorXd::Constant(1, phi), T, ThermalMassScheme::CTterm);
                            
                            // Create a local SelfEnergy instance to call the solver
                            SelfEnergy selfenergy(this->lam2, this->lamL, this->mA, this->mH, this->mHpm);
                            gapEqResult result = selfenergy.solve_gap_equations(phi, T, 1e-4, bosons_bare, bosons_init, 300);

                            size_t base_idx = idx * n_mass;
                            if (result.success) {
                                for (int k = 0; k < n_mass; ++k) {
                                    yM_flat[base_idx + k] = result.x[k];
                                }
                                //valid_points[idx] = true;
                            } else {
                                // Mark as NaN
                                for (int k = 0; k < n_mass; ++k) {
                                    yM_flat[base_idx + k] = std::nan("");
                                }
                                std::lock_guard<std::mutex> lock(bad_points_mutex);
                                bad_points_list.emplace_back(phi, T, i, j);
                                std::cerr << "\nWarning: phi=" << phi << ", T=" << T << ": " << result.message << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::lock_guard<std::mutex> lock(bad_points_mutex);
                            bad_points_list.emplace_back(phi, T, i, j);
                            std::cerr << "\nException at phi=" << phi << ", T=" << T << ": " << e.what() << std::endl;
                            size_t base_idx = idx * n_mass;
                            for (int k = 0; k < n_mass; ++k) yM_flat[base_idx + k] = std::nan("");
                        }

                        int completed = ++completed_points;
                        if (completed % 500 == 0 || completed == total_points) {
                            auto now = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double> elapsed = now - start_time;
                            double progress = (static_cast<double>(completed) / total_points) * 100.0;
                            double avg_time = elapsed.count() / completed;
                            double remaining_seconds = avg_time * (total_points - completed);
                            
                            // Helper lambda to format seconds into HH:MM:SS
                            auto format_time = [](double seconds) -> std::string {
                                int total_secs = static_cast<int>(seconds);
                                int h = total_secs / 3600;
                                int m = (total_secs % 3600) / 60;
                                int s = total_secs % 60;
                                std::ostringstream oss;
                                oss << std::setfill('0') << std::setw(2) << h << ":"
                                    << std::setfill('0') << std::setw(2) << m << ":"
                                    << std::setfill('0') << std::setw(2) << s;
                                return oss.str();
                            };

                            std::lock_guard<std::mutex> lock(bad_points_mutex); // Protect cout
                            std::cout << "\rProgress: " << completed << "/" << total_points 
                                      << " (" << std::fixed << std::setprecision(1) << progress << "%) | "
                                      << "Elapsed: " << format_time(elapsed.count()) << " | "
                                      << "Est. Remaining: " << format_time(remaining_seconds) << "   " << std::flush;
                        }
                    }
                };

                // Launch async tasks with static partitioning
                std::vector<std::future<void>> futures;
                // Determine the number of concurrent threads
                unsigned int max_threads = std::thread::hardware_concurrency();
                if (max_threads == 0) {
                    max_threads = 8; // Fallback default if hardware concurrency is not available
                }
                // Ensure at least 1 thread and not more than total points
                if (max_threads < 1) max_threads = 1;
                if (max_threads > total_points) max_threads = static_cast<unsigned int>(total_points);
                
                std::cout << "Using " << max_threads << " concurrent threads." << std::endl;

                // Calculate points per thread
                size_t points_per_thread = total_points / max_threads;
                size_t remainder = total_points % max_threads;

                futures.reserve(max_threads);

                size_t start_idx = 0;
                for (unsigned int t = 0; t < max_threads; ++t) {
                    // Distribute remainder points one by one to the first 'remainder' threads
                    size_t current_thread_points = points_per_thread + (t < remainder ? 1 : 0);
                    size_t end_idx = start_idx + current_thread_points;

                    // Launch async task for this chunk
                    // Capture xphi and xT by reference as they are valid until futures are joined
                    futures.push_back(std::async(std::launch::async, [&worker, start_idx, end_idx, &xphi, &xT, n_T]() {
                        worker(start_idx, end_idx, xphi, xT, n_T);
                    }));
                    
                    start_idx = end_idx;
                }
                
                // Wait for all threads to complete
                for (auto& f : futures) {
                    f.get();
                }
                std::cout << "\nGeneration complete." << std::endl;

                // Save data
                save_mass_data(M2_dat_path, xphi, xT, yM_flat, n_phi, n_T, n_mass);
                // Save bad points if any were found during generation or if we want to ensure the file reflects current state
                // Note: If loading from disk, bad_points_list is empty. We rely on the existence of the file on disk.
                if (!bad_points_list.empty()) {
                    std::ofstream bad_file(bad_points_path);
                    bad_file << "phi T i j" << std::endl;
                    for (const auto& bp : bad_points_list) {
                        bad_file << std::fixed << std::setprecision(6) << std::get<0>(bp) << " " << std::get<1>(bp) << " " << std::get<2>(bp) << " " << std::get<3>(bp) << std::endl;
                    }
                    bad_file.close();
                    std::cout << "Saved " << bad_points_list.size() << " bad points to " << bad_points_path << std::endl;
                }
            }

            create_mass_splines(xphi, xT, yM_flat, n_phi, n_T, n_mass);

        }

    private:
        MassSplines mass_splines_;

        void load_mass_data(const std::string& path, 
                            std::vector<double>& xphi, std::vector<double>& xT,
                            std::vector<double>& yM_flat,
                            int n_phi, int n_T, int n_mass) {
            std::ifstream file(path);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + path);
            }
            
            yM_flat.resize(static_cast<size_t>(n_phi) * n_T * n_mass);
            
            double phi, T;
            int count = 0;
            
            while (file >> phi) {
                file >> T;
                if (count >= n_phi * n_T) break;
                
                for (int k = 0; k < n_mass; ++k) {     
                    file >> yM_flat[count * n_mass + k];
                }
                count++;
            }
            
            if (count != n_phi * n_T) {
                std::cerr << "Warning: Loaded " << count << " points, expected " << n_phi * n_T << std::endl;
            }
        }

        void save_mass_data(const std::string& path,
                            const std::vector<double>& xphi, const std::vector<double>& xT,
                            const std::vector<double>& yM_flat,
                            int n_phi, int n_T, int n_mass) {
            std::ofstream file(path);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for writing: " + path);
            }

            for (int i = 0; i < n_phi; ++i) {
                for (int j = 0; j < n_T; ++j) {
                    size_t idx = static_cast<size_t>(i) * n_T + j;
                    file << std::scientific << std::setprecision(6)
                         << xphi[i] << " " << xT[j];
                    
                    for (int k = 0; k < n_mass; ++k) {
                        file << " " << yM_flat[idx * n_mass + k];
                    }
                    file << "\n";
                }
            }
            file.close();
        }

        void create_mass_splines(const std::vector<double>& xphi, const std::vector<double>& xT, 
                            const std::vector<double>& yM_flat, 
                            int n_phi, int n_T, int n_mass) {
            MassSplines result;
            // Build bad point mask
            std::vector<std::vector<bool>> is_bad(n_phi, std::vector<bool>(n_T, false));
            std::string bad_points_path = "/home/dayun/data/bad_points_" + paramNumber + ".txt";

            if (std::filesystem::exists(bad_points_path)) {
                std::ifstream bad_file(bad_points_path);
                std::string line;
                std::getline(bad_file, line); // Skip header
                while (std::getline(bad_file, line)) {
                    if (line.empty()) continue;
                    std::istringstream iss(line);
                    double phi, T;
                    int i, j;
                    // Try to read four values: phi T i j
                    if (!(iss >> phi >> T >> i >> j)) {
                        std::cerr << "Warning: malformed line in bad points file: " << line << std::endl;
                        continue;
                    }
                    // Boundary check
                    if (i >= 0 && i < n_phi && j >= 0 && j < n_T) {
                        is_bad[i][j] = true;
                    } else {
                        std::cerr << "Warning: bad point index (" << i << "," << j 
                                << ") out of range [0," << n_phi-1 << "] x [0," << n_T-1 << "]" << std::endl;
                    }
                }
                // Count bad points
                int bad_count = 0;
                for (const auto& row : is_bad)
                    bad_count += std::count(row.begin(), row.end(), true);
                std::cout << "Loaded " << bad_count << " bad grid points from " << bad_points_path << std::endl;
            } else {
                std::cout << "No bad points file found. Assuming all grid points are valid." << std::endl;
            }

            alglib::real_1d_array xphi_alglib, xT_alglib;
            xphi_alglib.setcontent(n_phi, xphi.data());
            xT_alglib.setcontent(n_T, xT.data());

            // 2. Local cubic spline interpolation (use at most 10 neighboring points, call parent class make_cubic_spline)
            auto local_cubic_spline = [this](const std::vector<double>& xs, const std::vector<double>& ys, double x0) -> double {
                const size_t n = xs.size();
                if (n == 0) return std::nan("");
                if (n == 1) return ys[0];
                // Use at most 10 points (already truncated to neighbors when passed in)
                alglib::real_1d_array x_arr, y_arr;
                x_arr.setcontent(n, xs.data());
                y_arr.setcontent(n, ys.data());
                alglib::spline1dinterpolant spline = make_cubic_spline(x_arr, y_arr);
                return alglib::spline1dcalc(spline, x0);
            };

            // 3. Process each mass
            for (int k = 0; k < n_mass; ++k) {
                // Extract 2D grid, set bad points to NaN
                std::vector<double> f_flat(n_phi * n_T);
                for (int i = 0; i < n_phi; ++i) {
                    for (int j = 0; j < n_T; ++j) {
                        // ALGLIB expects index: f_flat[j * n_phi + i] = value
                        if (!is_bad[i][j]) {
                            f_flat[j * n_phi + i] = yM_flat[(i * n_T + j) * n_mass + k];
                        } else { // Fill bad points
                            // Along T direction (fixed i): take up to 10 points before and after j (max 20 valid)
                            std::vector<double> T_vals, mass_vals;
                            int start_j = std::max(0, j - 10);
                            int end_j = std::min(n_T, j + 11);
                            for (int jj = start_j; jj < end_j; ++jj) {
                                if (!is_bad[i][jj]) {
                                    T_vals.push_back(xT[jj]);
                                    mass_vals.push_back(yM_flat[(i * n_T + jj) * n_mass + k]);
                                }
                            }
                            if (T_vals.size() >= 2) {
                                f_flat[j * n_phi + i] = local_cubic_spline(T_vals, mass_vals, xT[j]);
                                continue;
                            }

                            // Along phi direction (fixed j): take up to 10 points before and after i (max 20 valid)
                            std::vector<double> phi_vals, mass_vals2;
                            int start_i = std::max(0, i - 10);
                            int end_i = std::min(n_phi, i + 11);
                            for (int ii = start_i; ii < end_i; ++ii) {
                                if (!is_bad[ii][j]) {
                                    phi_vals.push_back(xphi[ii]);
                                    mass_vals2.push_back(yM_flat[(ii * n_T + j) * n_mass + k]);
                                }
                            }
                            if (phi_vals.size() >= 2) {
                                f_flat[j * n_phi + i] = local_cubic_spline(phi_vals, mass_vals2, xphi[i]);
                                continue;
                            }

                            // Fallback: nearest neighbor (3x3 neighborhood)
                            bool found = false;
                            for (int di = -1; di <= 1 && !found; ++di) {
                                for (int dj = -1; dj <= 1 && !found; ++dj) {
                                    int ni = i + di, nj = j + dj;
                                    if (ni >= 0 && ni < n_phi && nj >= 0 && nj < n_T && !is_bad[ni][nj]) {
                                        f_flat[j * n_phi + i] = f_flat[nj * n_phi + ni];
                                        found = true;
                                    }
                                }
                            }
                            if (!found) f_flat[j * n_phi + i] = 0.0; // Extreme case (should not happen)
                        }
                    }
                }

                 // 3.4 Convert f_flat to alglib::real_1d_array
                alglib::real_1d_array f_alglib;
                f_alglib.setcontent(n_phi * n_T, f_flat.data());

                // 3.5 Build bicubic spline (vector version, D=1)
                alglib::spline2dinterpolant spline2d;
                alglib::spline2dbuildbicubicv(xphi_alglib, n_phi, xT_alglib, n_T, f_alglib, 1, spline2d);

                // Assign
                switch (k) {
                    case 0:  result.Mh2   = spline2d; break;
                    case 1:  result.MG2   = spline2d; break;
                    case 2:  result.MA2   = spline2d; break;
                    case 3:  result.MH2   = spline2d; break;
                    case 4:  result.MHpm2 = spline2d; break;
                    case 5:  result.MW2L  = spline2d; break;
                    case 6:  result.MZ2L  = spline2d; break;
                    case 7:  result.Mga2L = spline2d; break;
                    case 8:  result.MW2T  = spline2d; break;
                    case 9:  result.MZ2T  = spline2d; break;
                    case 10: result.Mga2T = spline2d; break;
                    case 11: result.swL   = spline2d; break;
                    case 12: result.swT   = spline2d; break;
                    default: std::cerr << "Error: unexpected mass index " << k << std::endl;
                }
            }
        }

        double Gamma_lamp() const {
            double result = 0.0;
            result += - 0.5 * lamp * lam1 * Ijb(square(mh), 0.0, 2) + 2.0 * lamp * square(lam1*v0) * Ijb(square(mh), 0.0, 3);
            result += - 0.5 * lam2 * lamm * Ijb(square(mA), 0.0, 2) + 2.0 * lam2 * square(lamm*v0) * Ijb(square(mA), 0.0, 3);
            result += - 1.5 * lam2 * lamp * Ijb(square(mH), 0.0, 2) + 6.0 * lam2 * square(lamp*v0) * Ijb(square(mH), 0.0, 3);
            result += - lam2 * lam3 * Ijb(square(mHpm), 0.0, 2) + 2.0 * lam2 * square(lam3*v0) * Ijb(square(mHpm), 0.0, 3);
            double mZ2 = 0.25 * (square(g) + square(gp)) * square(v0);
            result -= 0.125 * square(square(g)+ square(gp)) * (3.0*Ijb(mZ2, 0.0, 2) + UV_term(mZ2, -2.0, 2) );
            double mW2 = 0.25 * square(g) * square(v0);
            result -= 0.25 * pow_4(g) * (3.0*Ijb(mW2, 0.0, 2) + UV_term(mW2, -2.0, 2) );

            return result;
        }

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
        std::string paramNumber = "000";
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

        ResummationScheme ResumScheme = ResummationScheme::None;











      };     





    



   




}  // namespace EffectivePotential

#endif