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
#include <map>
#include <memory>

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

    // Forward declaration or include for Spline library if available
    // For this example, we define a simple interface. 
    // In production, replace this with GSL gsl_spline2d or Eigen Splines.
    class BivariateSpline {
    public:
        virtual ~BivariateSpline() = default;
        virtual double evaluate(double phi, double T) const = 0;
    };

    // Placeholder implementation. You should replace this with a real library call.
    class LinearBivariateSpline : public BivariateSpline {
    private:
        std::vector<double> x_vals; // phi
        std::vector<double> y_vals; // T
        std::vector<double> z_vals; // flat data [phi_idx * n_T + T_idx]
        int n_x, n_y;
    public:
        LinearBivariateSpline(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z)
            : x_vals(x), y_vals(y), z_vals(z), n_x(x.size()), n_y(y.size()) {}

        double evaluate(double phi, double T) const override {
            // Simple bilinear interpolation or nearest neighbor for placeholder
            // Real implementation requires cubic spline logic
            if (phi < x_vals.front() || phi > x_vals.back() || T < y_vals.front() || T > y_vals.back()) {
                return std::nan("");
            }
            
            // Find indices
            auto it_x = std::lower_bound(x_vals.begin(), x_vals.end(), phi);
            auto it_y = std::lower_bound(y_vals.begin(), y_vals.end(), T);
            
            int idx_x = std::distance(x_vals.begin(), it_x);
            int idx_y = std::distance(y_vals.begin(), it_y);
            
            if (idx_x >= n_x) idx_x = n_x - 1;
            if (idx_y >= n_y) idx_y = n_y - 1;
            if (idx_x > 0 && phi < x_vals[idx_x]) idx_x--;
            if (idx_y > 0 && T < y_vals[idx_y]) idx_y--;

            // Return value (placeholder: just returning grid value)
            return z_vals[idx_x * n_y + idx_y];
        }
    };

    // Structure to hold splines for all thermal masses and mixing angles
    struct MassSplines {
        std::unique_ptr<BivariateSpline> Mh2;
        std::unique_ptr<BivariateSpline> MG2;
        std::unique_ptr<BivariateSpline> MA2;
        std::unique_ptr<BivariateSpline> MH2;
        std::unique_ptr<BivariateSpline> MHpm2;
        std::unique_ptr<BivariateSpline> MW2L;
        std::unique_ptr<BivariateSpline> MZ2L;
        std::unique_ptr<BivariateSpline> Mga2L;
        std::unique_ptr<BivariateSpline> MW2T;
        std::unique_ptr<BivariateSpline> MZ2T;
        std::unique_ptr<BivariateSpline> Mga2T;
        std::unique_ptr<BivariateSpline> swL;
        std::unique_ptr<BivariateSpline> swT;

        MassSplines() = default;
        
        // Helper to access by index for compatibility with old loop-based logic if needed
        // Or just use direct member access which is safer.
        // We will provide a getter that maps index to member for the init function.
        
        // Non-const version: Allows modification of the unique_ptr (e.g., during initialization)
        std::unique_ptr<BivariateSpline>& get_by_index(int index) {
            switch(index) {
                case 0: return Mh2;
                case 1: return MG2;
                case 2: return MA2;
                case 3: return MH2;
                case 4: return MHpm2;
                case 5: return MW2L;
                case 6: return MZ2L;
                case 7: return Mga2L;
                case 8: return MW2T;
                case 9: return MZ2T;
                case 10: return Mga2T;
                case 11: return swL;
                case 12: return swT;
                default: throw std::out_of_range("Invalid mass spline index");
            }
        }
        
        // Const version: Allows read-only access, required for use within const member functions
        const std::unique_ptr<BivariateSpline>& get_by_index(int index) const {
             switch(index) {
                case 0: return Mh2;
                case 1: return MG2;
                case 2: return MA2;
                case 3: return MH2;
                case 4: return MHpm2;
                case 5: return MW2L;
                case 6: return MZ2L;
                case 7: return Mga2L;
                case 8: return MW2T;
                case 9: return MZ2T;
                case 10: return Mga2T;
                case 11: return swL;
                case 12: return swT;
                default: throw std::out_of_range("Invalid mass spline index");
            }
        }
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
                    // exact thermal masses, as it requires solving gap equations self-consistently. This can be implemented in the future if needed.
                    return std::make_pair(std::vector<double>(11, 0.0), std::vector<double>(2, 0.0)); 
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
            std::vector<bool> valid_points(total_points, false);
            std::vector<std::pair<double, double>> bad_points_list;
            
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

                // Define the worker lambda to handle the computation for a single grid point
                auto worker = [this, n_mass, &yM_flat, &valid_points, &bad_points_list, &bad_points_mutex, &completed_points, total_points, start_time](int i, int j, double phi, double T, size_t idx) {
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
                            valid_points[idx] = true;
                        } else {
                            // Mark as NaN
                            for (int k = 0; k < n_mass; ++k) {
                                yM_flat[base_idx + k] = std::nan("");
                            }
                            std::lock_guard<std::mutex> lock(bad_points_mutex);
                            bad_points_list.emplace_back(phi, T);
                            std::cerr << "\nWarning: phi=" << phi << ", T=" << T << ": " << result.message << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lock(bad_points_mutex);
                        bad_points_list.emplace_back(phi, T);
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
                };

                // Launch async tasks for each grid point       
                std::vector<std::future<void>> futures;
                futures.reserve(total_points);

                for (int i = 0; i < n_phi; ++i) {
                    for (int j = 0; j < n_T; ++j) {
                        double phi = xphi[i];
                        double T = xT[j];
                        size_t idx = static_cast<size_t>(i) * n_T + j;

                        futures.push_back(std::async(std::launch::async, [worker, i, j, phi, T, idx]() {
                            worker(i, j, phi, T, idx);
                        }));
                    }
                }

                // Wait for all tasks to complete
                for (auto& f : futures) {
                    f.get();
                }
                std::cout << "\nGeneration complete." << std::endl;

                // Save data
                save_mass_data(M2_dat_path, xphi, xT, yM_flat, n_phi, n_T, n_mass);
            }

            // Save bad points if any were found during generation or if we want to ensure the file reflects current state
            // Note: If loading from disk, bad_points_list is empty. We rely on the existence of the file on disk.
            if (!bad_points_list.empty()) {
                std::ofstream bad_file(bad_points_path);
                bad_file << "phi T" << std::endl;
                for (const auto& bp : bad_points_list) {
                    bad_file << std::fixed << std::setprecision(6) << bp.first << " " << bp.second << std::endl;
                }
                bad_file.close();
                std::cout << "Saved " << bad_points_list.size() << " bad points to " << bad_points_path << std::endl;
            }

            // Create splines
            create_splines(xphi, xT, yM_flat, n_phi, n_T, n_mass, bad_points_path);
        }

        // Helper to get interpolated mass squared
        double get_interpolated_mass_sq(int mass_index, double phi, double T) const {
            if (mass_index < 0 || mass_index >= 13) {
                throw std::out_of_range("Invalid mass index");
            }
            const auto& spline = mass_splines_.get_by_index(mass_index);
            if (!spline) {
                throw std::runtime_error("Spline not initialized");
            }
            return spline->evaluate(phi, T);
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

        void create_splines(const std::vector<double>& xphi, const std::vector<double>& xT,
                            const std::vector<double>& yM_flat,
                            int n_phi, int n_T, int n_mass, const std::string& bad_points_path) {
            
            // Check for bad points file existence before creating splines
            if (std::filesystem::exists(bad_points_path)) {
                throw std::runtime_error("Bad points file detected: " + bad_points_path + ". Please resolve the bad points before proceeding with interpolation.");
            }

            for (int k = 0; k < n_mass; ++k) {
                // Extract column k from flat array
                std::vector<double> z_data(static_cast<size_t>(n_phi) * n_T);
                for (int i = 0; i < n_phi; ++i) {
                    for (int j = 0; j < n_T; ++j) {
                        size_t idx = static_cast<size_t>(i) * n_T + j;
                        z_data[idx] = yM_flat[idx * n_mass + k];
                    }
                }
                
                // Replace LinearBivariateSpline with your preferred high-quality spline implementation
                // e.g., GSL gsl_spline2d_alloc(gsl_spline2d_bicubic, ...)
                mass_splines_.get_by_index(k) = std::make_unique<LinearBivariateSpline>(xphi, xT, z_data);
            }
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