#ifndef POTENTIAL_IDM_HPP_INCLUDED
#define POTENTIAL_IDM_HPP_INCLUDED

/*
   IDM
*/

#include <complex>
#include <fstream>
#include <future>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <filesystem>
#include <string>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <interpolation.h>
#include "effectivepotential/potential.hpp"
#include "SelfEnergy.hpp"
#include "thermal_function.hpp"
#include "DRIDM_NNLO.hpp"

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

        const alglib::spline2dinterpolant& get(int idx) const {
            switch (idx) {
                case 0:  return Mh2;
                case 1:  return MG2;
                case 2:  return MA2;
                case 3:  return MH2;
                case 4:  return MHpm2;
                case 5:  return MW2L;
                case 6:  return MZ2L;
                case 7:  return Mga2L;
                case 8:  return MW2T;
                case 9:  return MZ2T;
                case 10: return Mga2T;
                case 11: return swL;
                case 12: return swT;
                default: throw std::out_of_range("Invalid mass index");
            }
        }
        // 非 const 版本，允许修改
        alglib::spline2dinterpolant& get(int idx) {
            return const_cast<alglib::spline2dinterpolant&>(const_cast<const MassSplines*>(this)->get(idx));
        }
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

        bool check_perturbativity() const {
            double max_coupling = std::max({std::abs(lam1), std::abs(lam2), std::abs(lam3), std::abs(lam4), std::abs(lam5)});
            return max_coupling < 4 * M_PI; // Perturbativity condition
        }

        bool check_vacuum_stability() const {
            // Vacuum stability conditions for IDM
            bool condition1 = lam1 > 0;
            bool condition2 = lam2 > 0;
            bool condition3 = lam3 + std::min(0.0, lam4 - std::abs(lam5)) > -std::sqrt(lam1 * lam2);
            return condition1 && condition2 && condition3;
        }

        size_t get_n_scalars() const override { return 1; }
        
        bool forbidden(Eigen::VectorXd X) const override { return X[0] < -0.1; } 
        
        std::vector<Eigen::VectorXd> apply_symmetry(Eigen::VectorXd X) const override {
            return {-X};
        }
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
                    double sw = std::abs(solver.eigenvectors()(0,0));
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
                    double sw = std::abs(solver.eigenvectors()(0,0));
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
                    double swL = std::abs(solverL.eigenvectors()(0,0));
                    Eigen::Matrix2d M2Tmatrix;
                    M2Tmatrix << MW32T, MW3B2T,
                                MW3B2T, MB2T;
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solverT(M2Tmatrix);
                    double Mga2T = solverT.eigenvalues()[0];
                    double MZ2T = solverT.eigenvalues()[1];
                    double swT = std::abs(solverT.eigenvectors()(0,0));
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
                            gapEqResult result = selfenergy.solve_gap_equations(phi, T, 1e-3, bosons_bare, bosons_init, 500); // 1% tolerance

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
                                // std::cerr << "\nWarning: phi=" << phi << ", T=" << T << ": " << result.message << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::lock_guard<std::mutex> lock(bad_points_mutex);
                            bad_points_list.emplace_back(phi, T, i, j);
                            std::cerr << "\nException at phi=" << phi << ", T=" << T << ": " << e.what() << std::endl;
                            size_t base_idx = idx * n_mass;
                            for (int k = 0; k < n_mass; ++k) yM_flat[base_idx + k] = std::nan("");
                        }

                        int completed = ++completed_points;
                        if (completed % 1000 == 0 || completed == total_points) {
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
            //std::cout << "Mass splines has created." << std::endl;
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
            
            std::string line;
            int count = 0;
            int line_number = 0;
            
            while (std::getline(file, line)) {
                line_number++;
                if (line.empty()) continue;

                std::istringstream iss(line);
                double phi, T;
                
                // Read phi and T
                if (!(iss >> phi >> T)) {
                    std::cerr << "Warning: Failed to parse phi/T at line " << line_number << std::endl;
                    continue; // Skip malformed lines instead of breaking immediately, or break depending on strictness
                }
                
                if (count >= n_phi * n_T) break;
                
                bool row_valid = true;
                for (int k = 0; k < n_mass; ++k) {     
                    std::string token;
                    if (!(iss >> token)) {
                        std::cerr << "Error: Missing data for mass component " << k << " at line " << line_number << std::endl;
                        row_valid = false;
                        break;
                    }
                    
                    // Handle NaN and Inf explicitly
                    if (token == "nan" || token == "NaN" || token == "-nan" || token == "-NaN") {
                        yM_flat[count * n_mass + k] = std::nan("");
                    } else if (token == "inf" || token == "Inf" || token == "+inf" || token == "+Inf") {
                        yM_flat[count * n_mass + k] = std::numeric_limits<double>::infinity();
                    } else if (token == "-inf" || token == "-Inf") {
                        yM_flat[count * n_mass + k] = -std::numeric_limits<double>::infinity();
                    } else {
                        try {
                            yM_flat[count * n_mass + k] = std::stod(token);
                        } catch (const std::exception& e) {
                            std::cerr << "Error: Invalid number format '" << token << "' at line " << line_number << ", col " << k << std::endl;
                            row_valid = false;
                            break;
                        }
                    }
                }
                
                if (row_valid) {
                    count++;
                } else {
                    throw std::runtime_error("Failed to parse data at line " + std::to_string(line_number));
                }
            }
            file.close();
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
                            int start_j = std::max(0, j - 50);
                            int end_j = std::min(n_T, j + 51);
                            //std::cout << "deal with bad point (i,j) = (" << i << ", " << j << ")" << std::endl;
                            for (int jj = start_j; jj < end_j; ++jj) {
                                if (!is_bad[i][jj]) {
                                    T_vals.push_back(xT[jj]);
                                    mass_vals.push_back(yM_flat[(i * n_T + jj) * n_mass + k]);
                                }
                            }
                            double interpolated_val = std::nan("");
                            if (T_vals.size() >= 2) {
                                try {
                                    interpolated_val = local_cubic_spline(T_vals, mass_vals, xT[j]);
                                } catch (...) {
                                    interpolated_val = std::nan("");
                                }
                            }
                            
                            if (std::isfinite(interpolated_val)) {
                                f_flat[j * n_phi + i] = interpolated_val;
                                //std::cout << "Interpolated bad point (i,j) = (" << i << ", " << j << ") to " << interpolated_val << std::endl;
                                continue;
                            }

                            // Along phi direction (fixed j): take up to 10 points before and after i (max 20 valid)
                            std::vector<double> phi_vals, mass_vals2;
                            int start_i = std::max(0, i - 50);
                            int end_i = std::min(n_phi, i + 51);
                            for (int ii = start_i; ii < end_i; ++ii) {
                                if (!is_bad[ii][j]) {
                                    phi_vals.push_back(xphi[ii]);
                                    mass_vals2.push_back(yM_flat[(ii * n_T + j) * n_mass + k]);
                                }
                            }
                            if (phi_vals.size() >= 2) {
                                try {
                                    interpolated_val = local_cubic_spline(phi_vals, mass_vals2, xphi[i]);
                                } catch (...) {
                                    interpolated_val = std::nan("");
                                }
                            }
                            
                            if (std::isfinite(interpolated_val)) {
                                f_flat[j * n_phi + i] = interpolated_val;
                                continue;
                            }

                            // Fallback: nearest neighbor (3x3 neighborhood)
                            bool found = false;
                            for (int di = -1; di <= 1 && !found; ++di) {
                                for (int dj = -1; dj <= 1 && !found; ++dj) {
                                    int ni = i + di, nj = j + dj;
                                    if (ni >= 0 && ni < n_phi && nj >= 0 && nj < n_T && !is_bad[ni][nj]) {
                                        f_flat[j * n_phi + i] = yM_flat[(ni * n_T + nj) * n_mass + k];
                                        found = true;
                                    }
                                }
                            }
                            if (!found) f_flat[j * n_phi + i] = 0.0; // Extreme case (should not happen)
                        }
                    }
                }

                 // 3.4 Convert f_flat to alglib::real_1d_array
                // Sanitize data before passing to ALGLIB to prevent ap_error from NaN/Inf
                for (auto& val : f_flat) {
                    if (!std::isfinite(val)) {
                        val = 0.0; // Replace NaN/Inf with 0.0 as a safe fallback for spline construction
                    }
                }

                alglib::real_1d_array f_alglib;
                f_alglib.setcontent(n_phi * n_T, f_flat.data());

                // 3.5 Build bicubic spline (vector version, D=1)
                alglib::spline2dinterpolant spline2d;
                try {
                    alglib::spline2dbuildbicubicv(xphi_alglib, n_phi, xT_alglib, n_T, f_alglib, 1, spline2d);
                } catch (const alglib::ap_error& e) {
                    std::cerr << "ALGLIB Error building spline for mass index " << k << ": " << e.msg << std::endl;
                    // Initialize with a dummy spline or rethrow depending on desired behavior
                    // Here we initialize with zeros to allow program continuation, but this indicates bad data
                    alglib::real_1d_array zero_f;
                    zero_f.setlength(n_phi * n_T);
                    alglib::spline2dbuildbicubicv(xphi_alglib, n_phi, xT_alglib, n_T, zero_f, 1, spline2d);
                }
                mass_splines_.get(k) = spline2d;
                // Assign
                // switch (k) {
                //     case 0:  mass_splines_.Mh2   = spline2d; break;
                //     case 1:  mass_splines_.MG2   = spline2d; break;
                //     case 2:  mass_splines_.MA2   = spline2d; break;
                //     case 3:  mass_splines_.MH2   = spline2d; break;
                //     case 4:  mass_splines_.MHpm2 = spline2d; break;
                //     case 5:  mass_splines_.MW2L  = spline2d; break;
                //     case 6:  mass_splines_.MZ2L  = spline2d; break;
                //     case 7:  mass_splines_.Mga2L = spline2d; break;
                //     case 8:  mass_splines_.MW2T  = spline2d; break;
                //     case 9:  mass_splines_.MZ2T  = spline2d; break;
                //     case 10: mass_splines_.Mga2T = spline2d; break;
                //     case 11: mass_splines_.swL   = spline2d; break;
                //     case 12: mass_splines_.swT   = spline2d; break;
                //     default: std::cerr << "Error: unexpected mass index " << k << std::endl;
                // }
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



////////////////////////
//        DR   ////////
////////////////////////
    class IDM_DR : public Potential {

        private:
            PROPERTY(int, potential_flag, 1);
            PROPERTY(int, matching_flag, 2);
            PROPERTY(bool, running_flag, true);

            // constants
            const double v0 = 246.22; // GeV
            const double mh = 125.10; // GeV
            const double mt = 172.76; // GeV
            const double g = 0.65175;
            const double gp = 0.35742;

            double gpsq_input = gp*gp;
            double gsq_input = g*g;
            double gssq_input = 1.2172 * 1.2172; //SU3
            double yt_input = sqrt(2.0)*mt/v0;
            //const double gstar = 110.75;

            // input parameters
            double lam2_input = 0;
            double lamL_input = 0;
            double mA_input = 0;
            double mH_input = 0;
            double mHpm_input = 0;

            //others
            double lam1_input = 0;
            double mu1sq_input = 0;
            double mu2sq_input = 0;
            double lamm_input = 0;
            double lamp_input = 0;
            double lam3_input = 0;
            double lam4_input = 0;
            double lam5_input = 0;

        public:
            IDM_DR(double lam2_input_, double lamL_input_, double mA_input_, double mH_input_, double mHpm_input_) {
                lam2_input = lam2_input_;
                lamL_input = lamL_input_;
                mA_input = mA_input_;
                mH_input = mH_input_;
                mHpm_input = mHpm_input_;
                lam1_input = square(mh/v0);
                mu1sq_input = - 0.5 * square(mh);

                lam4_input = (square(mA_input) + square(mH_input) - 2 * square(mHpm_input))/ square(v0);
                lam5_input = (square(mH_input) - square(mA_input))/ square(v0);
                lam3_input = 2 * lamL_input - lam4_input - lam5_input;
                mu2sq_input = 0.5* (2 * square(mHpm_input) - lam3_input * square(v0));
                lamp_input = lam3_input + lam4_input + lam5_input;
                lamm_input = lam3_input + lam4_input - lam5_input;


                std::vector<double> x0 = {gpsq_input, gsq_input, gssq_input, lam1_input, lam2_input, lam3_input, lam4_input, lam5_input, yt_input, mu1sq_input, mu2sq_input};

                solveBetas(x0, RG_scale, 1., 5000., 0.5);
            }

            size_t get_n_scalars() const override { return 1;}

            IDMParameters get_params() const {
                IDMParameters params;
                params.v0 = v0;
                params.mh = mh;
                params.mt = mt;
                params.yt = yt_input;
                params.g = g;
                params.gp = gp;
                params.gstar = 110.75;
                params.lam2 = lam2_input;
                params.lamL = lamL_input;
                params.mA = mA_input;
                params.mH = mH_input;
                params.mHpm = mHpm_input;
                params.lam1 = lam1_input;
                params.mu1sq = mu1sq_input;
                params.mu2sq = mu2sq_input;
                params.lamm = lamm_input;
                params.lamp = lamp_input;
                params.lam3 = lam3_input;
                params.lam4 = lam4_input;
                params.lam5 = lam5_input;
                return params;
            }

            std::vector<Eigen::VectorXd> apply_symmetry(Eigen::VectorXd X) const override {
                return {-X};
            }

            bool forbidden(Eigen::VectorXd X) const override { return X[0] < -0.1; } 

            bool check_perturbativity() const {
                double max_coupling = std::max({std::abs(lam1_input), std::abs(lam2_input), std::abs(lam3_input), std::abs(lam4_input), std::abs(lam5_input)});
                return max_coupling < 4 * M_PI; // Perturbativity condition
            }

            bool check_vacuum_stability() const {
                // Vacuum stability conditions for IDM
                bool condition1 = lam1_input > 0;
                bool condition2 = lam2_input > 0;
                bool condition3 = lam3_input + std::min(0.0, lam4_input - std::abs(lam5_input)) > -std::sqrt(lam1_input * lam2_input);
                return condition1 && condition2 && condition3;
            }

            void Betas( const std::vector<double>& x, std::vector<double>& dxdt, const double t) override {
                double gpsq = x[0];
                double gsq = x[1];
                double gssq = x[2];
                double lam1 = x[3];
                double lam2 = x[4];
                double lam3 = x[5];
                double lam4 = x[6];
                double lam5 = x[7];
                double yt = x[8];
                double mu1sq = x[9];
                double mu2sq = x[10];
                dxdt[0] = 1./t * 7. /(8*M_PI*M_PI) * square(gpsq);
                dxdt[1] = - 1./t * 3. /(8*M_PI*M_PI) * square(gsq);
                dxdt[2] = - 1./t * 7. /(8*M_PI*M_PI) * square(gssq);
                dxdt[3] = 1./t * 1. / (64. * M_PI * M_PI) * (
                    9. * square(gsq) + 6. * gsq * (gpsq - 6. * lam1) + 
                    3. * (square(gpsq) - 4. * gpsq * lam1 + 16. * lam1 * (square(yt) + lam1)) + 
                    8. * (-6. * pow_4(yt) + 2. * square(lam3) + 2. * lam3 * lam4 + square(lam4) + square(lam5))
                );
                dxdt[4] = 1./t * 1. / (64. * M_PI * M_PI) * (
                    9. * square(gsq) + 6. * gsq * (gpsq - 6. * lam2) + 
                    3. * (square(gpsq) - 4. * gpsq * lam2 ) + 
                    8. * (6. * square(lam2) + 2. * square(lam3) + 2. * lam3 * lam4 + square(lam4) + square(lam5))
                );
                dxdt[5] = 1./t * 1. / (64. * M_PI * M_PI) * (
                    9. * square(gsq) + 3. * square(gpsq) - 12. * gpsq * lam3 - 
                    6. * gsq * (gpsq + 6. * lam3) + 
                    8. * (lam3 * (3. * (square(yt) + lam1 + lam2) + 2. * lam3) + (lam1 + lam2) * lam4 + square(lam4) + square(lam5))
                );
                dxdt[6] = 1./t * 1. / (16. * M_PI * M_PI) * (
                    3. * gsq * (gpsq - 3. * lam4) - 3. * gpsq * lam4 + 
                    2. * lam4 * (3. * square(yt) + lam1 + lam2 + 4. * lam3 + 2. * lam4) + 8. * square(lam5)
                );
                dxdt[7] = 1./t * 1. / (16. * M_PI * M_PI) * (
                    (-9. * gsq - 3. * gpsq + 2. * (3. * square(yt) + lam1 + lam2 + 4. * lam3 + 6. * lam4)) * lam5
                );
                dxdt[8] = 1./t * 1. / (192. * M_PI * M_PI) * (
                    yt * (-27. * gsq - 17. * gpsq - 96. * gssq + 54. * square(yt))
                );
                dxdt[9] = 1./t * 1. / (32. * M_PI * M_PI) * (
                    -3. * (3. * gsq + gpsq - 4. * (square(yt) + lam1)) * mu1sq + 
                    4. * (2. * lam3 + lam4) * mu2sq
                );
                dxdt[10] = 1./t * 1. / (32. * M_PI * M_PI) * (
                    8. * lam3 * mu1sq + 4. * lam4 * mu1sq - 
                    3. * (3. * gsq + gpsq - 4. * lam2) * mu2sq
                );
            }

            std::vector<double> get_3d_parameters(double T) const {
                //-----------------------------------------------
                // 4D parameters
                //-----------------------------------------------
                double RG_scale_4D = M_PI * T;
                double Lb = 2.*log(M_PI) + 2. * EulerGamma - 2. * log(4 * M_PI);
                double Lf = Lb + 4. * log(2.);
                double gpsq, gsq, gssq, lam1, lam2, lam3, lam4, lam5, yt, mu1sq, mu2sq;

                if ( running_flag ) {
                    gpsq = alglib::spline1dcalc(RGEs[0], RG_scale_4D);
                    gsq = alglib::spline1dcalc(RGEs[1], RG_scale_4D);
                    gssq = alglib::spline1dcalc(RGEs[2], RG_scale_4D);
                    lam1 = alglib::spline1dcalc(RGEs[3], RG_scale_4D);
                    lam2 = alglib::spline1dcalc(RGEs[4], RG_scale_4D);
                    lam3 = alglib::spline1dcalc(RGEs[5], RG_scale_4D);
                    lam4 = alglib::spline1dcalc(RGEs[6], RG_scale_4D);
                    lam5 = alglib::spline1dcalc(RGEs[7], RG_scale_4D);
                    yt = alglib::spline1dcalc(RGEs[8], RG_scale_4D);
                    mu1sq = alglib::spline1dcalc(RGEs[9], RG_scale_4D);
                    mu2sq = alglib::spline1dcalc(RGEs[10], RG_scale_4D);
                } else {
                    gpsq = gpsq_input;
                    gsq = gsq_input;
                    gssq = gssq_input;
                    lam1 = lam1_input;
                    lam2 = lam2_input;
                    lam3 = lam3_input;
                    lam4 = lam4_input;
                    lam5 = lam5_input;
                    yt = yt_input;
                    mu1sq = mu1sq_input;
                    mu2sq = mu2sq_input;
                }
                //-----------------------------------------------
                // 3D couplings
                //-----------------------------------------------
                double gsq3d = gsq * T + (square(gsq) * (2. + 21. * Lb - 12. * Lf) * T) / (48. * square(M_PI));
                double gpsq3d = gpsq * T - (square(gpsq) * (Lb + 20. * Lf) * T) / (48. * square(M_PI));
                double gssq3d = gssq * T + (square(gssq) * (1. + 11. * Lb - 4. * Lf) * T) / (16. * square(M_PI));
                double lam13d = -1. / (128. * square(M_PI)) * T * (
                    square(gpsq) * (-2. + 3. * Lb) + square(gsq) * (-6. + 9. * Lb) - 12. * gpsq * Lb * lam1 +
                    gsq * (gpsq * (-4. + 6. * Lb) - 36. * Lb * lam1) +
                    8. * (-16. * square(M_PI) * lam1 + 6. * Lf * square(yt) * (-square(yt) + lam1) +
                        Lb * (6. * square(lam1) + 2. * square(lam3) + 2. * lam3 * lam4 + square(lam4) + square(lam5)))
                );
                double lam23d = -1. / (128. * square(M_PI)) * T * (
                    square(gpsq) * (-2. + 3. * Lb) + square(gsq) * (-6. + 9. * Lb) - 12. * gpsq * Lb * lam2 -
                    128. * square(M_PI) * lam2 +
                    gsq * (gpsq * (-4. + 6. * Lb) - 36. * Lb * lam2) +
                    8. * Lb * (6. * square(lam2) + 2. * square(lam3) + 2. * lam3 * lam4 + square(lam4) + square(lam5))
                );
                double lam33d = -1. / (128. * square(M_PI)) * T * (
                    square(gpsq) * (-2. + 3. * Lb) + square(gsq) * (-6. + 9. * Lb) - 12. * gpsq * Lb * lam3 +
                    gsq * (gpsq * (4. - 6. * Lb) - 36. * Lb * lam3) +
                    8. * (-16. * square(M_PI) * lam3 + 3. * Lf * square(yt) * lam3 + Lb * (lam3 * (3. * (lam1 + lam2) + 2. * lam3) + (lam1 + lam2) * lam4 + square(lam4) + square(lam5)))
                );
                double lam43d = -1. / (32. * square(M_PI)) * T * (
                    -3. * gpsq * Lb * lam4 - 32. * square(M_PI) * lam4 +
                    6. * Lf * square(yt) * lam4 + 2. * Lb * lam1 * lam4 +
                    2. * Lb * lam2 * lam4 + 8. * Lb * lam3 * lam4 +
                    4. * Lb * square(lam4) + gsq * (gpsq * (-2. + 3. * Lb) - 9. * Lb * lam4) +
                    8. * Lb * square(lam5)
                );
                double lam53d = (T * (9. * gsq * Lb + 3. * gpsq * Lb + 32. * square(M_PI) - 6. * Lf * square(yt) - 2. * Lb * (lam1 + lam2 + 4. * lam3 + 6. * lam4)) * lam5) / (32. * square(M_PI));
                //-----------------------------------------------
                // Mixed temporal-scalar couplings
                //-----------------------------------------------
                double lambdaVLL_1 = (square(gsq) * T) / (4. * square(M_PI));
                double lambdaVLL_2 = -(gsq * gpsq * T) / (4. * square(M_PI));
                double lambdaVLL_3 = -(181. * square(gpsq) * T) / (36. * square(M_PI));
                double lambdaVLL_4 = -(3. * gsq * gssq * T) / (4. * square(M_PI));  
                double lambdaVLL_5 = -(11. * gpsq * gssq * T) / (12. * square(M_PI));
                double lambdaVLL_6 = -(sqrt(gssq* gpsq)  * gssq * T) / (4. * square(M_PI));
                double lambdaVLL_7 = (square(gssq) * T) / (4. * square(M_PI));
                double lambdaVL_1 = -((gssq * square(yt) * T) / (4. * square(M_PI)));
                double lambdaVL_2 = (sqrt(gsq) * sqrt(gpsq) * T) / (192. * square(M_PI)) * (
                    gsq * (5. + 21. * Lb - 12. * Lf) - 
                    gpsq * (-21. + Lb + 20. * Lf) + 
                    12. * (8. * square(M_PI) + square(yt) + lam1 + lam4)
                );
                double lambdaVL_3 = (sqrt(gsq) * sqrt(gpsq) * T) / (192. * square(M_PI)) * (
                    gsq * (5. + 21. * Lb - 12. * Lf) - 
                    gpsq * (-21. + Lb + 20. * Lf) + 
                    12. * (8. * square(M_PI) + lam2 + lam4)
                );  
                double lambdaVL_4 = -1. / (192. * square(M_PI)) * gpsq * T * (
                    -9. * gsq + gpsq * (-39. + 2. * Lb + 40. * Lf) - 
                    4. * (24. * square(M_PI) - 17. * square(yt) + 9. * lam1 + 6. * lam3 + 3. * lam4)
                );
                double lambdaVL_5 = -1. / (192. * square(M_PI)) * gpsq * T * (
                    -9. * gsq + gpsq * (-39. + 2. * Lb + 40. * Lf) - 
                    12. * (8. * square(M_PI) + 3. * lam2 + 2. * lam3 + lam4)
                );
                double lambdaVL_6 = (gsq * T) / (192. * square(M_PI)) * (
                    gsq * (73. + 42. * Lb - 24. * Lf) + 
                    3. * (gpsq + 4. * (8. * square(M_PI) - 3. * square(yt) + 3. * lam1 + 2. * lam3 + lam4))
                );
                double lambdaVL_7 = (gsq * T) / (192. * square(M_PI)) * (
                    gsq * (73. + 42. * Lb - 24. * Lf) + 
                    3. * (gpsq + 4. * (8. * square(M_PI) + 3. * lam2 + 2. * lam3 + lam4))
                );
                //-----------------------------------------------
                // 3D mass
                //-----------------------------------------------
                double RG_scale_3D = sqrt(gsq) * T; // 3D RG scale
                double mu1sq3d_LO = 1./48. * (
                    9. * gsq * T * T + 3. * gpsq * T * T + 
                    4. * (3. * T * T * square(yt) + 3. * T * T * lam1 + 
                        2. * T * T * lam3 + T * T * lam4 + 12. * mu1sq)
                );
                double mu2sq3d_LO = 1./48. * (
                    9. * gsq * T * T + 3. * gpsq * T * T + 
                    12. * T * T * lam2 + 8. * T * T * lam3 + 
                    4. * T * T * lam4 + 48. * mu2sq
                );
                double musqSU2_LO = 2. * gsq * T * T;
                double musqSU3_LO = 2. * gssq * T * T;
                double musqU1_LO = 2. * gpsq * T * T;
                double mu1sq3d_NLO = -1. / (4608. * square(M_PI)) * (
                    -51. * square(gpsq) * T * T + 
                    81. * EulerGamma * square(gpsq) * T * T + 
                    150. * square(gpsq) * Lb * T * T - 
                    60. * square(gpsq) * Lf * T * T + 
                    66. * gpsq * T * T * square(yt) + 
                    576. * gssq * T * T * square(yt) - 
                    47. * gpsq * Lb * T * T * square(yt) + 
                    192. * gssq * Lb * T * T * square(yt) - 
                    55. * gpsq * Lf * T * T * square(yt) - 
                    768. * gssq * Lf * T * T * square(yt) - 
                    108. * Lb * T * T * pow_4(yt) - 
                    36. * gpsq * T * T * lam1 - 
                    216. * EulerGamma * gpsq * T * T * lam1 + 
                    108. * gpsq * Lb * T * T * lam1 + 
                    324. * Lb * T * T * square(yt) * lam1 + 
                    108. * Lf * T * T * square(yt) * lam1 + 
                    432. * EulerGamma * T * T * square(lam1) - 
                    24. * gpsq * T * T * lam3 - 
                    144. * EulerGamma * gpsq * T * T * lam3 + 
                    72. * gpsq * Lb * T * T * lam3 + 
                    144. * Lf * T * T * square(yt) * lam3 + 
                    144. * Lb * T * T * lam1 * lam3 + 
                    144. * Lb * T * T * lam2 * lam3 + 
                    288. * EulerGamma * T * T * square(lam3) - 
                    48. * Lb * T * T * square(lam3) - 
                    12. * gpsq * T * T * lam4 - 
                    72. * EulerGamma * gpsq * T * T * lam4 + 
                    36. * gpsq * Lb * T * T * lam4 + 
                    72. * Lf * T * T * square(yt) * lam4 + 
                    72. * Lb * T * T * lam1 * lam4 + 
                    72. * Lb * T * T * lam2 * lam4 + 
                    288. * EulerGamma * T * T * lam3 * lam4 - 
                    48. * Lb * T * T * lam3 * lam4 + 
                    288. * EulerGamma * T * T * square(lam4) - 
                    120. * Lb * T * T * square(lam4) + 
                    432. * EulerGamma * T * T * square(lam5) - 
                    216. * Lb * T * T * square(lam5) - 
                    216. * gpsq * Lb * mu1sq + 
                    864. * Lf * square(yt) * mu1sq + 
                    864. * Lb * lam1 * mu1sq + 
                    576. * Lb * lam3 * mu2sq + 
                    288. * Lb * lam4 * mu2sq + 
                    9. * gsq * (
                        -72. * Lb * mu1sq + 
                        T * T * (
                            3. * (2. - 7. * Lb + Lf) * square(yt) - 
                            4. * (3. * lam1 + 2. * lam3 + lam4) * (1. + 6. * EulerGamma - 3. * Lb - 72. * log(Glaisher))
                        ) + 
                        6. * gpsq * T * T * (1. + 5. * EulerGamma - 4. * Lb - 60. * log(Glaisher))
                    ) - 
                    9. * square(gsq) * T * T * (67. + 75. * EulerGamma - 84. * Lb + 12. * Lf - 900. * log(Glaisher)) - 
                    972. * square(gpsq) * T * T * log(Glaisher) + 
                    2592. * gpsq * T * T * lam1 * log(Glaisher) - 
                    5184. * T * T * square(lam1) * log(Glaisher) + 
                    1728. * gpsq * T * T * lam3 * log(Glaisher) - 
                    3456. * T * T * square(lam3) * log(Glaisher) + 
                    864. * gpsq * T * T * lam4 * log(Glaisher) - 
                    3456. * T * T * lam3 * lam4 * log(Glaisher) - 
                    3456. * T * T * square(lam4) * log(Glaisher) - 
                    5184. * T * T * square(lam5) * log(Glaisher) + 
                    18. * log(RG_scale_3D / RG_scale_4D) * (
                        33. * square(gsq3d) - 7. * square(gpsq3d) + 
                        8. * gsq3d * (3. * lam13d + 2. * lam33d + lam43d) - 
                        8. * (
                            6. * square(lam13d) + 4. * square(lam33d) + 
                            4. * lam33d * lam43d + 4. * square(lam43d) + 
                            6. * square(lam53d) - 48. * gssq3d * lambdaVL_1 + 
                            8. * square(lambdaVL_1) + 
                            6. * square(lambdaVL_2) + square(lambdaVL_4) + 
                            3. * square(lambdaVL_6)
                        ) + 
                        6. * gsq3d * (
                            -3. * gpsq3d + 
                            4. * (3. * lam13d + 2. * lam33d + lam43d + 4. * lambdaVL_6)
                        )
                    )
                );
                double mu2sq3d_NLO = 1. / (3072. * square(M_PI)) * (
                    -96. * EulerGamma * T * T * (3. * square(lam2) + 2. * square(lam3) + 2. * lam3 * lam4 + 2. * square(lam4) + 3. * square(lam5)) - 
                    8. * Lb * (
                        T * T * (9. * square(yt) * (2. * lam3 + lam4) + 
                                2. * (6. * lam1 * lam3 + 6. * lam2 * lam3 - 2. * square(lam3) + 3. * lam1 * lam4 + 
                                    3. * lam2 * lam4 - 2. * lam3 * lam4 - 5. * square(lam4) - 9. * square(lam5))
                        ) + 
                        24. * (2. * lam3 * mu1sq + lam4 * mu1sq + 3. * lam2 * mu2sq)
                    ) + 
                    4. * gpsq * (
                        36. * Lb * mu2sq + 
                        2. * T * T * (3. * lam2 + 2. * lam3 + lam4) * (1. + 6. * EulerGamma - 3. * Lb - 72. * log(Glaisher))
                    ) + 
                    6. * square(gsq) * T * T * (67. + 75. * EulerGamma - 84. * Lb + 12. * Lf - 900. * log(Glaisher)) - 
                    2. * square(gpsq) * T * T * (-17. + 27. * EulerGamma + 50. * Lb - 20. * Lf - 324. * log(Glaisher)) + 
                    24. * T * T * (Lf * square(yt) * (2. * lam3 + lam4) + 
                                    48. * (3. * square(lam2) + 2. * square(lam3) + 2. * lam3 * lam4 + 2. * square(lam4) + 3. * square(lam5)) * log(Glaisher)
                    ) + 
                    6. * gsq * (
                        72. * Lb * mu2sq + 
                        4. * T * T * (3. * lam2 + 2. * lam3 + lam4) * (1. + 6. * EulerGamma - 3. * Lb - 72. * log(Glaisher)) + 
                        6. * gpsq * T * T * (-1. - 5. * EulerGamma + 4. * Lb + 60. * log(Glaisher))
                    ) - 
                    12. * log(RG_scale_3D / RG_scale_4D) * (
                        33. * square(gsq3d) - 7. * square(gpsq3d) + 
                        8. * gsq3d * (3. * lam23d + 2. * lam33d + lam43d) - 
                        8. * (
                            6. * square(lam23d) + 4. * square(lam33d) + 
                            4. * lam33d * lam43d + 4. * square(lam43d) + 
                            6. * square(lam53d) + 6. * square(lambdaVL_3) + square(lambdaVL_5) + 
                            3. * square(lambdaVL_7)
                        ) + 
                        6. * gsq3d * (
                            -3. * gpsq3d + 
                            4. * (3. * lam23d + 2. * lam33d + lam43d + 4. * lambdaVL_7)
                        )
                    )
                );
                double musqSU2_NLO = (gsq / (576. * square(M_PI))) * (
                    72. * (mu1sq + mu2sq) + 
                    3. * T * T * (
                        -3. * gpsq - 72. * gssq - 3. * square(yt) + 
                        6. * lam1 + 6. * lam2 + 8. * lam3 + 4. * lam4 + 
                        gsq * (115. + 144. * EulerGamma - 672. * log(2.) - 144. * log(M_PI))
                    ) + 
                    8. * gsq * T * T * (-25. * log(T) + 25. * log(RG_scale_4D) + 29. * log(RG_scale_4D / T))
                );
                double musqSU3_NLO = (gssq * T * T / (192. * square(M_PI))) * (
                    -27. * gsq - 11. * gpsq - 12. * square(yt) + 
                    24. * gssq * (5. + 14. * EulerGamma - 22. * log(2.)) + 
                    24. * gssq * (-3. * log(M_PI * T) + 3. * log(RG_scale_4D) + 11. * log(RG_scale_4D / (4. * M_PI * T)))
                );
                double musqU1_NLO = (gpsq / (576. * square(M_PI))) * (
                    72. * (mu1sq + mu2sq) + 
                    T * T * (
                        -27. * gsq + 
                        3. * (-88. * gssq - 11. * square(yt) + 6. * lam1 + 6. * lam2 + 8. * lam3 + 4. * lam4) + 
                        gpsq * (251. - 1008. * EulerGamma + 48. *(21.* log( M_PI)) + log(4.))
                    ) + 
                    8. * gpsq * T * T * (101. * (log(T) - log(RG_scale_4D)) - 25. * log(RG_scale_4D / T))
                );
                double mu1sq3d = mu1sq3d_LO + mu1sq3d_NLO;
                double mu2sq3d = mu2sq3d_LO + mu2sq3d_NLO;
                double musqU1 = musqU1_LO + musqU1_NLO;
                double musqSU2 = musqSU2_LO + musqSU2_NLO;
                double musqSU3 = musqSU3_LO + musqSU3_NLO;

                if ( matching_flag == 0 ) {
                    return {gpsq*T, gsq*T, gssq*T, lam1*T, lam2*T, lam33d*T, lam43d*T, lam53d*T, mu1sq3d_LO, mu2sq3d_LO};
                } else if ( matching_flag == 1 ) {
                    return {gpsq3d, gsq3d, gssq3d, lam13d, lam23d, lam33d, lam43d, lam53d, mu1sq3d, mu2sq3d};
                }
                //---------------------------------
                // integrate out temporal scalar
                //---------------------------------
                // couplings
                double RG_scale_3DUS = gsq * T;

                // integrate out the second doublet
                // double lam13dUS = lam13d - (1. / (16. * M_PI)) * (
                //     (2. * (2. * square(lam33d) + 2. * lam33d * lam43d + square(lam43d) + square(lam53d))) / sqrt(mu2sq3d) + 
                //     (8. * square(lambdaVL_1)) / sqrt(musqSU3) + 
                //     (4. * square(lambdaVL_2)) / (sqrt(musqSU2) + sqrt(musqU1)) + 
                //     square(lambdaVL_4) / sqrt(musqU1) + 
                //     (3. * square(lambdaVL_6)) / sqrt(musqSU2)
                // );
                // std::cout << "lam33d: " << lam33d << std::endl;
                // std::cout << "lam43d: " << lam43d << std::endl;
                // std::cout << "lam53d: " << lam53d << std::endl;
                // std::cout << "mu2sq3d: " << mu2sq3d << std::endl;

                // double gsq3dUS = gsq3d - (square(gsq3d) * (1. / sqrt(mu2sq3d) + 2. / sqrt(musqSU2))) / (48. * M_PI);
                // double gpsq3dUS = gpsq3d - square(gpsq3d) / (48. * M_PI * sqrt(mu2sq3d));
                // double gssq3dUS = gssq3d - square(gssq3d) / (16. * M_PI * sqrt(musqSU3));
                // // masses
                // double mu1sq3dUS_LO = mu1sq3d - (1. / (8. * M_PI)) * (
                //     4. * lam33d * sqrt(mu2sq3d) + 
                //     2. * lam43d * sqrt(mu2sq3d) + 
                //     8. * sqrt(musqSU3) * lambdaVL_1 + 
                //     sqrt(musqU1) * lambdaVL_4 + 
                //     3. * sqrt(musqSU2) * lambdaVL_6
                // );
                // double mu1sq3dUS_NLO = 1. / (128. * square(M_PI)) * (
                //     6. * gsq3d * lam33d + 2. * gpsq3d * lam33d + 
                //     24. * lam23d * lam33d - 8. * square(lam33d) + 
                //     3. * gsq3d * lam43d + gpsq3d * lam43d + 
                //     12. * lam23d * lam43d - 8. * lam33d * lam43d - 
                //     8. * square(lam43d) - 12. * square(lam53d) - 
                //     (3. * square(gsq3d) + square(gpsq3d) - 
                //     12. * gsq3d * (2. * lam33d + lam43d) - 
                //     4. * gpsq3d * (2. * lam33d + lam43d) + 
                //     8. * (2. * square(lam33d) + 2. * lam33d * lam43d + 
                //         2. * square(lam43d) + 3. * square(lam53d))) * log(RG_scale_3D / (2. * sqrt(mu2sq3d))) + 
                //     48. * gsq3d * lambdaVL_1 + 
                //     192. * gsq3d * log(RG_scale_3D / (2. * sqrt(musqSU3))) * lambdaVL_1 - 
                //     16. * square(lambdaVL_1) - 
                //     32. * log(RG_scale_3D / (2. * sqrt(musqSU3))) * square(lambdaVL_1) - 
                //     12. * square(lambdaVL_2) - 
                //     24. * log(RG_scale_3D / (sqrt(musqSU2) + sqrt(musqU1))) * square(lambdaVL_2) - 
                //     2. * square(lambdaVL_4) - 
                //     4. * log(RG_scale_3D / (2. * sqrt(musqU1))) * square(lambdaVL_4) + 
                //     12. * gsq3d * lambdaVL_6 - 6. * square(lambdaVL_6) - 
                //     6. * log(RG_scale_3D / (2. * sqrt(musqSU2))) * (square(gsq3d) - 8. * gsq3d * lambdaVL_6 + 
                //     2. * square(lambdaVL_6)) + 
                //     15. * lambdaVL_6 * lambdaVLL_1 + 
                //     (3. * sqrt(musqSU2) * lambdaVL_4 * lambdaVLL_2) / sqrt(musqU1) + 
                //     (3. * sqrt(musqU1) * lambdaVL_6 * lambdaVLL_2) / sqrt(musqSU2) + 
                //     lambdaVL_4 * lambdaVLL_3 + 
                //     (24. * sqrt(musqSU2) * lambdaVL_1 * lambdaVLL_4) / sqrt(musqSU3) + 
                //     (24. * sqrt(musqSU3) * lambdaVL_6 * lambdaVLL_4) / sqrt(musqSU2) + 
                //     (8. * sqrt(musqU1) * lambdaVL_1 * lambdaVLL_5) / sqrt(musqSU3) + 
                //     (8. * sqrt(musqSU3) * lambdaVL_4 * lambdaVLL_5) / sqrt(musqU1) + 
                //     80. * lambdaVL_1 * lambdaVLL_7
                // );
                // double mu1sq3dUS_beta = 1. / (256. * square(M_PI)) * (
                //     -51. * square(gsq3dUS) + 
                //     5. * square(gpsq3dUS) + 
                //     18. * gsq3dUS * (gpsq3dUS - 4. * lam13dUS) - 
                //     24. * gpsq3dUS * lam13dUS + 
                //     48. * square(lam13dUS)
                // );
                // double mu1Sq3dUS = mu1sq3dUS_LO + mu1sq3dUS_NLO + mu1sq3dUS_beta * log(RG_scale_3DUS / RG_scale_3D);

                double lam13dUS = lam13d - (1. / (16. * M_PI)) * (
                    (8. * lambdaVL_1 * lambdaVL_1) / sqrt(musqSU3) + 
                    (4. * lambdaVL_2 * lambdaVL_2) / (sqrt(musqSU2) + sqrt(musqU1)) + 
                    (lambdaVL_4 * lambdaVL_4) / sqrt(musqU1) + 
                    (3. * lambdaVL_6 * lambdaVL_6) / sqrt(musqSU2)
                );
                double lam23dUS = lam23d - (1. / (16. * M_PI)) * (
                    (4. * lambdaVL_3 * lambdaVL_3) / (sqrt(musqSU2) + sqrt(musqU1)) + 
                    (lambdaVL_5 * lambdaVL_5) / sqrt(musqU1) + 
                    (3. * lambdaVL_7 * lambdaVL_7) / sqrt(musqSU2)
                );
                double lam33dUS = lam33d - (1. / (16. * M_PI)) * (
                    - (4. * lambdaVL_2 * lambdaVL_3) / (sqrt(musqSU2) + sqrt(musqU1)) + 
                    (lambdaVL_4 * lambdaVL_5) / sqrt(musqU1) + 
                    (3. * lambdaVL_6 * lambdaVL_7) / sqrt(musqSU2)
                );
                double lam43dUS = lam43d - (lambdaVL_2 * lambdaVL_3) / (2. * M_PI * (sqrt(musqSU2) + sqrt(musqU1)));
                double lam53dUS = lam53d;
                double gsq3dUS = gsq3d - (gsq3d * gsq3d * gsq3d) / (24. * M_PI * sqrt(musqSU2));
                double gpsq3dUS = gpsq3d;
                double gssq3dUS = gssq3d - (gssq3d * gssq3d * gssq3d) / (16. * M_PI * sqrt(musqSU3));

                double mu1Sq3dUS_LO = mu1sq3d - (8. * sqrt(musqSU3) * lambdaVL_1 + sqrt(musqU1) * lambdaVL_4 + 3. * sqrt(musqSU2) * lambdaVL_6) / (8. * M_PI);
                double mu1Sq3dUS_NLO = (1. / (128. * square(M_PI))) * (
                    48. * gssq3d * lambdaVL_1 + 
                    32. * log(RG_scale_3D / (2. * sqrt(musqSU3))) * (6. * gssq3d - lambdaVL_1) * lambdaVL_1 - 
                    16. * lambdaVL_1 * lambdaVL_1 - 
                    12. * lambdaVL_2 * lambdaVL_2 - 
                    24. * log(RG_scale_3D / (sqrt(musqSU2) + sqrt(musqU1))) * lambdaVL_2 * lambdaVL_2 - 
                    2. * lambdaVL_4 * lambdaVL_4 - 
                    4. * log(RG_scale_3D / (2. * sqrt(musqU1))) * lambdaVL_4 * lambdaVL_4 + 
                    12. * gsq3d * lambdaVL_6 - 
                    6. * lambdaVL_6 * lambdaVL_6 - 
                    6. * log(RG_scale_3D / (2. * sqrt(musqSU2))) * (gsq3d * gsq3d - 8. * gsq3d * lambdaVL_6 + 2. * lambdaVL_6 * lambdaVL_6) + 
                    15. * lambdaVL_6 * lambdaVLL_1 + 
                    (3. * sqrt(musqSU2) * lambdaVL_4 * lambdaVLL_2) / sqrt(musqU1) + 
                    (3. * sqrt(musqU1) * lambdaVL_6 * lambdaVLL_2) / sqrt(musqSU2) + 
                    lambdaVL_4 * lambdaVLL_3 + 
                    (24. * sqrt(musqSU2) * lambdaVL_1 * lambdaVLL_4) / sqrt(musqSU3) + 
                    (24. * sqrt(musqSU3) * lambdaVL_6 * lambdaVLL_4) / sqrt(musqSU2) + 
                    (8. * sqrt(musqU1) * lambdaVL_1 * lambdaVLL_5) / sqrt(musqSU3) + 
                    (8. * sqrt(musqSU3) * lambdaVL_4 * lambdaVLL_5) / sqrt(musqU1) + 
                    80. * lambdaVL_1 * lambdaVLL_7
                );
                double mu1Sq3dUS_beta = (1. / (256. * square(M_PI))) * (
                    -45. * gsq3dUS * gsq3dUS + 
                    7. * gpsq3dUS * gpsq3dUS + 
                    48. * lam23dUS * lam23dUS - 
                    8. * gpsq3dUS * (3. * lam23dUS + 2. * lam33dUS + lam43dUS) + 
                    32. * (lam33dUS * lam33dUS + lam33dUS * lam43dUS + lam43dUS * lam43dUS) + 
                    6. * gsq3dUS * (3. * gpsq3dUS - 4. * (3. * lam23dUS + 2. * lam33dUS + lam43dUS)) + 
                    48. * lam53dUS * lam53dUS
                );
                double mu1Sq3dUS = mu1Sq3dUS_LO + mu1Sq3dUS_NLO + mu1Sq3dUS_beta * log(RG_scale_3DUS / RG_scale_3D);
                double mu2sq3dUS_LO = mu2sq3d - (sqrt(musqU1) * lambdaVL_5 + 3. * sqrt(musqSU2) * lambdaVL_7) / (8. * M_PI);
                double mu2sq3dUS_NLO = (1. / (128. * square(M_PI))) * (
                    -12. * lambdaVL_3 * lambdaVL_3 - 
                    24. * log(RG_scale_3D / (sqrt(musqSU2) + sqrt(musqU1))) * lambdaVL_3 * lambdaVL_3 - 
                    2. * lambdaVL_5 * lambdaVL_5 - 
                    4. * log(RG_scale_3D / (2. * sqrt(musqU1))) * lambdaVL_5 * lambdaVL_5 + 
                    12. * gsq3d * lambdaVL_7 - 
                    6. * lambdaVL_7 * lambdaVL_7 - 
                    6. * log(RG_scale_3D / (2. * sqrt(musqSU2))) * (gsq3d * gsq3d - 8. * gsq3d * lambdaVL_7 + 2. * lambdaVL_7 * lambdaVL_7) + 
                    15. * lambdaVL_7 * lambdaVLL_1 + 
                    (3. * sqrt(musqSU2) * lambdaVL_5 * lambdaVLL_2) / sqrt(musqU1) + 
                    (3. * sqrt(musqU1) * lambdaVL_7 * lambdaVLL_2) / sqrt(musqSU2) + 
                    lambdaVL_5 * lambdaVLL_3 + 
                    (24. * sqrt(musqSU3) * lambdaVL_7 * lambdaVLL_4) / sqrt(musqSU2) + 
                    (8. * sqrt(musqSU3) * lambdaVL_5 * lambdaVLL_5) / sqrt(musqU1)
                );
                double mu2sq3dUS_beta = (1. / (256. * square(M_PI))) * (
                    -45. * gsq3dUS * gsq3dUS + 
                    7. * gpsq3dUS * gpsq3dUS + 
                    48. * lam13dUS * lam13dUS - 
                    8. * gpsq3dUS * (3. * lam13dUS + 2. * lam33dUS + lam43dUS) + 
                    32. * (lam33dUS * lam33dUS + lam33dUS * lam43dUS + lam43dUS * lam43dUS) + 
                    6. * gsq3dUS * (3. * gpsq3dUS - 4. * (3. * lam13dUS + 2. * lam33dUS + lam43dUS)) + 
                    48. * lam53dUS * lam53dUS
                );
                double mu2sq3dUS = mu2sq3dUS_LO + mu2sq3dUS_NLO + mu2sq3dUS_beta * log(RG_scale_3DUS / RG_scale_3D);

                if ( matching_flag == 2 ) {
                    return {gpsq3dUS, gsq3dUS, gssq3dUS, lam13dUS, lam23dUS, lam33dUS, lam43dUS, lam53dUS, mu1Sq3dUS, mu2sq3dUS};
                }
                return {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
            }

            double V(Eigen::VectorXd X, double T) const override {
    
                const std::vector<double> par = get_3d_parameters(T);
                std::complex<double> gpsq(par[0], 0);
                std::complex<double> gsq(par[1], 0);
                std::complex<double> gssq(par[2], 0);
                std::complex<double> lam1(par[3], 0);
                std::complex<double> lam2(par[4], 0);
                std::complex<double> lam3(par[5], 0);
                std::complex<double> lam4(par[6], 0);
                std::complex<double> lam5(par[7], 0);
                std::complex<double> mu1sq(par[8], 0);
                std::complex<double> mu2sq(par[9], 0);

                std::complex<double> phi(X[0]/sqrt(T + 1e-15),0);

                std::complex<double> veffLO = 0.5* pow(phi,2.)*mu1sq + 0.125*pow(phi,4.)*lam1 ;

                if ( potential_flag == 0 ) {
                    return T * veffLO.real();
                }

                std::complex<double> veffNLO = 
                    - pow(mu1sq + 0.5 * lam1 * phi * phi, 1.5) / (4. * M_PI) - 
                    pow(mu1sq + 1.5 * lam1 * phi * phi, 1.5) / (12. * M_PI) - 
                    pow(mu2sq + 0.5 * lam3 * phi * phi, 1.5) / (6. * M_PI) - 
                    pow(mu2sq + 0.5 * (lam3 + lam4 - lam5) * phi * phi, 1.5) / (12. * M_PI) - 
                    pow(mu2sq + 0.5 * (lam3 + lam4 + lam5) * phi * phi, 1.5) / (12. * M_PI) + 
                    2. * (- pow(gsq * phi * phi, 1.5) / (48. * M_PI) - pow((gsq + gpsq) * phi * phi, 1.5) / (96. * M_PI));

                if ( potential_flag == 1 ) {
                    return T * veffLO.real() + T * veffNLO.real();
                }

                if ( potential_flag == 2 ) {
                    DRIDM_NNLO VNNLO;
                    std::complex<double> veffNNLO = VNNLO.V2(X, T, par);
                    return T * veffLO.real() + T * veffNLO.real() + T * veffNNLO.real();
                }

                return 0.;
            }


    };



    



   




}  // namespace EffectivePotential

#endif