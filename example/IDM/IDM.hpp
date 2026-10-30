#ifndef POTENTIAL_IDM_HPP_INCLUDED
#define POTENTIAL_IDM_HPP_INCLUDED

/*
   IDM
*/

#include <complex>
#include <filesystem>
#include <future>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <interpolation.h>
#include "effectivepotential/potential.hpp"
#include "SelfEnergy.hpp"
#include "DRIDM_NNLO.hpp"
#include "effectivepotential/thermal_function.hpp"
#include "pow.hpp"
#include <gsl/gsl_roots.h>

namespace EffectivePotential {

    enum class ResummationScheme {
        None,
        Parwani,
        ArnoldEspinosa,
        PartialDressing,
        DolanJackiw,
        DR,
        DYQ
    };

    enum class ThermalMassScheme {
        Tree,
        HighT,
        Exact,
        CTterm,
        dMdT,
        dMdx
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
        // alglib::spline2dinterpolant MW2T;
        // alglib::spline2dinterpolant MZ2T;
        // alglib::spline2dinterpolant Mga2T;
        alglib::spline2dinterpolant swL;
        // alglib::spline2dinterpolant swT;

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
                // case 8:  return MW2T;
                // case 9:  return MZ2T;
                // case 10: return Mga2T;
                case 8: return swL;
                // case 12: return swT;
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

    struct Params_qawc {
        double T;
        double t0;

        double M2;
        double lam1;
        double phi;

        double mh2;
        double mG2;
    };

    // ------------------------------------------------------------

    // 根方程 H(t) = 0
    //
    // H(t) = t^2/(1-t)^2
    //      + [M2 + PiIR(T t/(1-t))]/T^2
    // ------------------------------------------------------------
    inline double root_function(double t, void* params) {
        Params_qawc* p = static_cast<Params_qawc*>(params);

        const double k = p->T * t / (1.0 - t);

        return t * t / ((1.0 - t) * (1.0 - t)) + (p->M2 - 4.5 * square(p->lam1*p->phi)*ReI20(p->mh2, k, p->T) - 1.5 * square(p->lam1*p->phi)*ReI20(p->mG2, k, p->T)) / (p->T * p->T);
    }

    // ------------------------------------------------------------
    // 求 t0
    // ------------------------------------------------------------
    inline double find_t0(Params_qawc& p) {
        gsl_function F;
        F.function = &root_function;
        F.params   = &p;

        // --------------------------------------------------------
        // 先找 bracket
        // --------------------------------------------------------

        double t_lo = 1.0e-12;
        double t_hi = 0.5;

        double f_lo = root_function(t_lo, &p);
        double f_hi = root_function(t_hi, &p);

        while (f_lo * f_hi > 0.0) {
            t_hi = 0.5 * (1.0 + t_hi);

            if (t_hi >= 1.0 - 1.0e-12)
            {
                std::cerr << "Cannot bracket the root.\n";
                //std::cout << "t_lo = " << t_lo << " t_hi = " << t_hi << std::endl;
                std::exit(EXIT_FAILURE);
            }

            f_hi = root_function(t_hi, &p);
        }

        // --------------------------------------------------------
        // Brent
        // --------------------------------------------------------

        gsl_root_fsolver* s =
            gsl_root_fsolver_alloc(gsl_root_fsolver_brent);

        gsl_root_fsolver_set(
            s,
            &F,
            t_lo,
            t_hi
        );

        double t0 = 0.0;

        for (int iter = 0; iter < 100; ++iter)
        {
            gsl_root_fsolver_iterate(s);

            t0 = gsl_root_fsolver_root(s);

            double x_lo =
                gsl_root_fsolver_x_lower(s);

            double x_hi =
                gsl_root_fsolver_x_upper(s);

            int status =
                gsl_root_test_interval(
                    x_lo,
                    x_hi,
                    1.0e-10,
                    1.0e-10
                );

            if (status == GSL_SUCCESS)
                break;
        }

        gsl_root_fsolver_free(s);

        return t0;
    }

    //-------------------------------

    //被积函数 G(t)/(t-t0)
    // ------------------------------------------------------------
    inline double G_qawc(double t, void *params) {
        Params_qawc *p = static_cast<Params_qawc *>(params);

        const double T  = p->T;
        const double t0 = p->t0;
        const double M2 = p->M2;
        const double lam1 = p->lam1;
        const double phi = p->phi;
        const double mh2 = p->mh2;
        const double mG2 = p->mG2;

        const double dt = t - t0;

        // Local regularization
        constexpr double delta_t = 1e-8;

        if (std::abs(dt) < delta_t)
        {
            const double k0 = T*t0/(1.0 - t0);

            const double f0  = M2 - 4.5 * square(lam1*phi)*ReI20(mh2, k0, T) - 1.5 * square(lam1*phi)*ReI20(mG2, k0, T);
            const double gp0 = 2.0*k0 - 4.5 * square(lam1*phi)*dI20dk(mh2, k0, T) - 1.5 * square(lam1*phi)*dI20dk(mG2, k0, T);

            // lim_{t->t0} G(t) = f0/(T*gp0)；正则化只在该点取极限值，
            // 不要引入虚假的 (1-t0)/(1-t) 因子，否则正则区与一般式导数不匹配。
            //if (std::isnan(abs(f0/(T*gp0)))||std::isinf(abs(f0/(T*gp0)))) std::cout << "2*k0*std::sqrt(mh2) = " << 2*k0*std::sqrt(mh2) << " " << "k0 = " << k0 << " " << "mG2 = " << mG2 << std::endl;
            return f0/(T*gp0);
        }

        const double k  = T*t /(1.0 - t);
        const double k0 = T*t0/(1.0 - t0);

        const double numerator =
            dt * (M2 - 4.5 * square(lam1*phi)*ReI20(mh2, k, T) - 1.5 * square(lam1*phi)*ReI20(mG2, k, T));
        const double denominator =
            (1.0 - t)*(1.0 - t)
            * (k*k  - 4.5 * square(lam1*phi)*ReI20(mh2, k, T) - 1.5 * square(lam1*phi)*ReI20(mG2, k, T) 
            - k0*k0 + 4.5 * square(lam1*phi)*ReI20(mh2, k0, T) + 1.5 * square(lam1*phi)*ReI20(mG2, k0, T));
        return numerator / denominator;
    }

    //-------------------------------

    // 计算 CPV  int_0^1 G(t)/(t-t0) dt
    // ------------------------------------------------------------
    inline double calculate_CPV(Params_qawc *p) {
        // 加大 workspace 让 QAWC 在小 mG2 陡变区能多细分
        gsl_integration_workspace *workspace =
            gsl_integration_workspace_alloc(100000);

        gsl_function F;
        F.function = &G_qawc;
        F.params   = p;

        double result = 0.0;
        double error  = 0.0;

        const int status = gsl_integration_qawc(
            &F,
            0.0,
            1.0,
            p->t0,
            1e-6,    // 绝对容差
            1e-6,    // 相对容差
            100000,
            workspace,
            &result,
            &error
        );

        gsl_integration_workspace_free(workspace);

        if (status != GSL_SUCCESS)
        {
            std::cerr << "gsl_integration_qawc failed at T=" << p->T
                    << " phi=" << p->phi << " t0=" << p->t0
                    << " mh2=" << p->mh2 << " mG2=" << p->mG2
                    << ": " << gsl_strerror(status) << '\n';
            return std::nan("");   // 失败时返回 NaN，让上层走 Gauss-Kronrod 兜底
        }

        return result;
    }



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

        // void fitmass(Eigen::VectorXd X, double T,  std::array<alglib::spline1dinterpolant, 13>& splines, ThermalMassScheme ms) const {
        //     double phi = X[0];
        //     SelfEnergy se(lam2, lamL, mA, mH, mHpm);
        //     struct GapPoint {
        //         double x;
        //         std::array<double, 13> values;
        //     };
        //     std::vector<GapPoint> points;
        //     for (int i = 0; i < 401; ++i) {
        //         double phix;
        //         double Tx;
        //         if (ms == ThermalMassScheme::dMdx) {
        //             phix = i ;
        //             Tx = T;
        //             auto bosons_init = boson_massSq(Eigen::VectorXd::Constant(1, phix), Tx, ThermalMassScheme::Tree); 
        //             auto bosons_bare = boson_massSq(Eigen::VectorXd::Constant(1, phix), Tx, ThermalMassScheme::CTterm);
        //             auto result = se.solve_gap_equations(phix, Tx, 1e-3, bosons_bare, bosons_init, 300);
        //             if (result.success) {
        //                 GapPoint pt;
        //                 pt.x = phix;
        //                 std::copy(result.x.begin(), result.x.end(), pt.values.begin());
        //                 points.push_back(pt);
        //             } 
        //         } else if (ms == ThermalMassScheme::dMdT) {
        //             Tx = i ;
        //             phix = phi;
        //             auto bosons_init = boson_massSq(Eigen::VectorXd::Constant(1, phix), Tx, ThermalMassScheme::Tree); 
        //             auto bosons_bare = boson_massSq(Eigen::VectorXd::Constant(1, phix), Tx, ThermalMassScheme::CTterm);
        //             auto result = se.solve_gap_equations(phix, Tx, 1e-3, bosons_bare, bosons_init, 300);
        //             if (result.success) {
        //                 GapPoint pt;
        //                 pt.x = Tx;
        //                 std::copy(result.x.begin(), result.x.end(), pt.values.begin());
        //                 points.push_back(pt);
        //             }
        //         }
        //     }
        //     if (points.empty()) throw std::runtime_error("Failed to calculate gap points in phi = " + std::to_string(phi) + " and T = " + std::to_string(T));
        //     double rho = 0.1; // 光滑系数
        //     for (int k = 0; k < 13; ++k) {
        //         // 从 points 中提取第 k 个分量的数据
        //         std::vector<double> x_vals, y_vals;
        //         x_vals.reserve(points.size());
        //         y_vals.reserve(points.size());
        //         for (const auto& pt : points) {
        //             x_vals.push_back(pt.x);
        //             y_vals.push_back(pt.values[k]);
        //         }
        //         // 转换为 ALGLIB 数组
        //         alglib::real_1d_array x_arr, y_arr;
        //         x_arr.setcontent(x_vals.size(), x_vals.data());
        //         y_arr.setcontent(y_vals.size(), y_vals.data());
        //         if (x_vals.size() >= 4) {
        //             alglib::spline1dfitreport rep;
        //             alglib::spline1dfit(x_arr, y_arr, 10, rho, splines[k], rep);
        //         } else {
        //             // 点数不足，线性插值
        //             //alglib::spline1dbuildlinear(x_arr, y_arr, splines[k]);
        //             throw std::runtime_error("Insufficient points to fit spline for k = " + std::to_string(k));
        //         }
        //     }
        // }

        std::pair<std::vector<double>, std::vector<double>> boson_massSq(Eigen::VectorXd X, double T, ThermalMassScheme ms) const {
            double phi = std::abs(X[0]);

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
                    //std::cout << square(g*gp) * square(phi) * square(phi) - square(g * gp * square(phi))<< std::endl;
                    double mZ2 = solver.eigenvalues()[1];
                    double sw = std::abs(solver.eigenvectors()(0,0));
                    std::vector<double> mSquares = {mh2, mG2, mA2, mH2, mHpm2, mW2, mZ2, mga2, mW2, mZ2, mga2};
                    std::vector<double> mixing_angles = {sw, sw} ;
                    return std::make_pair(mSquares, mixing_angles);
                }
                case ThermalMassScheme::CTterm: { 
                    if (ResumScheme == ResummationScheme::DYQ) {
                        mh2 += delta_mu1sq + 1.5 * delta_lam14 * square(phi);
                        mG2 += delta_mu1sq + 0.5 * delta_lam14 * square(phi);
                        mA2 += delta_mu2sq + 0.5 * delta_lamm4 * square(phi);
                        mH2 += delta_mu2sq + 0.5 * delta_lamp4 * square(phi);
                        mHpm2 += delta_mu2sq + 0.5 * delta_lam34 * square(phi);
                    } else {
                        mh2 += delta_mu1sq + 1.5 * delta_lam1 * square(phi);
                        mG2 += delta_mu1sq + 0.5 * delta_lam1 * square(phi);
                        mA2 += delta_mu2sq + 0.5 * delta_lamm * square(phi);
                        mH2 += delta_mu2sq + 0.5 * delta_lamp * square(phi);
                        mHpm2 += delta_mu2sq + 0.5 * delta_lam3 * square(phi);
                    }
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
                    // if (std::abs(T - cached_T) > 1e-6 ) {
                    //     cached_T = T;
                    //     fitmass(X, T, cached_splines_T, ThermalMassScheme::dMdx);
                    // }
                    // double Mh2 = alglib::spline1dcalc(cached_splines_T[0], phi);
                    // double MG2 = alglib::spline1dcalc(cached_splines_T[1], phi);
                    // double MA2 = alglib::spline1dcalc(cached_splines_T[2], phi);
                    // double MH2 = alglib::spline1dcalc(cached_splines_T[3], phi);
                    // double MHpm2 = alglib::spline1dcalc(cached_splines_T[4], phi);
                    // double MW2L = alglib::spline1dcalc(cached_splines_T[5], phi);
                    // double MZ2L = alglib::spline1dcalc(cached_splines_T[6], phi);
                    // double Mga2L = alglib::spline1dcalc(cached_splines_T[7], phi);
                    // double MW2T = alglib::spline1dcalc(cached_splines_T[8], phi);
                    // double MZ2T = alglib::spline1dcalc(cached_splines_T[9], phi);
                    // double Mga2T = alglib::spline1dcalc(cached_splines_T[10], phi);
                    // double swL = alglib::spline1dcalc(cached_splines_T[11], phi);
                    // double swT = alglib::spline1dcalc(cached_splines_T[12], phi);


                    double Mh2 = alglib::spline2dcalc(mass_splines_.Mh2, phi, T);
                    double MG2 = alglib::spline2dcalc(mass_splines_.MG2, phi, T);
                    double MA2 = alglib::spline2dcalc(mass_splines_.MA2, phi, T);
                    double MH2 = alglib::spline2dcalc(mass_splines_.MH2, phi, T);
                    double MHpm2 = alglib::spline2dcalc(mass_splines_.MHpm2, phi, T);
                    double MW2L = alglib::spline2dcalc(mass_splines_.MW2L, phi, T);
                    double MZ2L = alglib::spline2dcalc(mass_splines_.MZ2L, phi, T);
                    double Mga2L = alglib::spline2dcalc(mass_splines_.Mga2L, phi, T);
                    double MW2T = 0.25 * square(g) * square(phi);//alglib::spline2dcalc(mass_splines_.MW2T, phi, T);
                    double MZ2T = 0.25 * (square(g) + square(gp)) * square(phi);//alglib::spline2dcalc(mass_splines_.MZ2T, phi, T);
                    double Mga2T = 0.0;//alglib::spline2dcalc(mass_splines_.Mga2T, phi, T);
                    double swL = alglib::spline2dcalc(mass_splines_.swL, phi, T);
                    double swT = 0.48084;//alglib::spline2dcalc(mass_splines_.swT, phi, T);

                    
                    std::vector<double> MSquares = {Mh2, MG2, MA2, MH2, MHpm2, MW2L, MZ2L, Mga2L, MW2T, MZ2T, Mga2T};
                    std::vector<double> mixing_angles = {swL, swT};
                    return std::make_pair(MSquares, mixing_angles);
                }
                case ThermalMassScheme::dMdx: {
                    // if (std::abs(T - cached_T) > 1e-6 ) {
                    //     cached_T = T;
                    //     fitmass(X, T, cached_splines_T, ThermalMassScheme::dMdx);
                    // }
                    // double Mh2, dMh2_dx, d2Mh2_dx2;
                    // alglib::spline1ddiff(cached_splines_T[0], phi, Mh2, dMh2_dx, d2Mh2_dx2);
                    // double MG2, dMG2_dx, d2MG2_dx2;
                    // alglib::spline1ddiff(cached_splines_T[1], phi, MG2, dMG2_dx, d2MG2_dx2);
                    // double MA2, dMA2_dx, d2MA2_dx2;
                    // alglib::spline1ddiff(cached_splines_T[2], phi, MA2, dMA2_dx, d2MA2_dx2);
                    // double MH2, dMH2_dx, d2MH2_dx2;
                    // alglib::spline1ddiff(cached_splines_T[3], phi, MH2, dMH2_dx, d2MH2_dx2);
                    // double MHpm2, dMHpm2_dx, d2MHpm2_dx2;
                    // alglib::spline1ddiff(cached_splines_T[4], phi, MHpm2, dMHpm2_dx, d2MHpm2_dx2);
                    // double MW2L, dMW2L_dx, d2MW2L_dx2;
                    // alglib::spline1ddiff(cached_splines_T[5], phi, MW2L, dMW2L_dx, d2MW2L_dx2);
                    // double MZ2L, dMZ2L_dx, d2MZ2L_dx2;
                    // alglib::spline1ddiff(cached_splines_T[6], phi, MZ2L, dMZ2L_dx, d2MZ2L_dx2);
                    // double Mga2L, dMga2L_dx, d2Mga2L_dx2;
                    // alglib::spline1ddiff(cached_splines_T[7], phi, Mga2L, dMga2L_dx, d2Mga2L_dx2);
                    // double MW2T, dMW2T_dx, d2MW2T_dx2;
                    // alglib::spline1ddiff(cached_splines_T[8], phi, MW2T, dMW2T_dx, d2MW2T_dx2);
                    // double MZ2T, dMZ2T_dx, d2MZ2T_dx2;
                    // alglib::spline1ddiff(cached_splines_T[9], phi, MZ2T, dMZ2T_dx, d2MZ2T_dx2);
                    // double Mga2T, dMga2T_dx, d2Mga2T_dx2;
                    // alglib::spline1ddiff(cached_splines_T[10], phi, Mga2T, dMga2T_dx, d2Mga2T_dx2);
                    // double swL, dswL_dx, d2swL_dx2;
                    // alglib::spline1ddiff(cached_splines_T[11], phi, swL, dswL_dx, d2swL_dx2);
                    // double swT, dswT_dx, d2swT_dx2;
                    // alglib::spline1ddiff(cached_splines_T[12], phi, swT, dswT_dx, d2swT_dx2);
                    double Mh2, dMh2_dx, dMh2_dt;
                    alglib::spline2ddiff(mass_splines_.Mh2, phi, T, Mh2, dMh2_dx, dMh2_dt);
                    double MG2, dMG2_dx, dMG2_dt;
                    alglib::spline2ddiff(mass_splines_.MG2, phi, T, MG2, dMG2_dx, dMG2_dt);
                    double MA2, dMA2_dx, dMA2_dt;
                    alglib::spline2ddiff(mass_splines_.MA2, phi, T, MA2, dMA2_dx, dMA2_dt);
                    double MH2, dMH2_dx, dMH2_dt;
                    alglib::spline2ddiff(mass_splines_.MH2, phi, T, MH2, dMH2_dx, dMH2_dt);
                    double MHpm2, dMHpm2_dx, dMHpm2_dt;
                    alglib::spline2ddiff(mass_splines_.MHpm2, phi, T, MHpm2, dMHpm2_dx, dMHpm2_dt);
                    double MW2L, dMW2L_dx, dMW2L_dt;
                    alglib::spline2ddiff(mass_splines_.MW2L, phi, T, MW2L, dMW2L_dx, dMW2L_dt);
                    double MZ2L, dMZ2L_dx, dMZ2L_dt;
                    alglib::spline2ddiff(mass_splines_.MZ2L, phi, T, MZ2L, dMZ2L_dx, dMZ2L_dt);
                    double Mga2L, dMga2L_dx, dMga2L_dt;
                    alglib::spline2ddiff(mass_splines_.Mga2L, phi, T, Mga2L, dMga2L_dx, dMga2L_dt);
                    double dMW2T_dx = 0.5 * square(g)*phi;
                    //alglib::spline2ddiff(mass_splines_.MW2T, phi, T, MW2T, dMW2T_dx, dMW2T_dt);
                    double dMZ2T_dx = 0.5 * (square(g) + square(gp))*phi;
                    //alglib::spline2ddiff(mass_splines_.MZ2T, phi, T, MZ2T, dMZ2T_dx, dMZ2T_dt);
                    double dMga2T_dx = 0.0;
                    //alglib::spline2ddiff(mass_splines_.Mga2T, phi, T, Mga2T, dMga2T_dx, dMga2T_dt);
                    double swL, dswL_dx, dswL_dt;
                    alglib::spline2ddiff(mass_splines_.swL, phi, T, swL, dswL_dx, dswL_dt);
                    double swT = 0.48084, dswT_dx = 0.0, dswT_dt = 0.0;
                    //alglib::spline2ddiff(mass_splines_.swT, phi, T, swT, dswT_dx, dswT_dt);
                    std::vector<double> MSquares, mixing_angles;
                    if (X[0] < 0.0) {
                        MSquares = {-dMh2_dx, -dMG2_dx, -dMA2_dx, -dMH2_dx, -dMHpm2_dx, -dMW2L_dx, -dMZ2L_dx, -dMga2L_dx, -dMW2T_dx, -dMZ2T_dx, -dMga2T_dx};
                        mixing_angles = {-dswL_dx, -dswT_dx};
                    }
                    else {
                        MSquares = {dMh2_dx, dMG2_dx, dMA2_dx, dMH2_dx, dMHpm2_dx, dMW2L_dx, dMZ2L_dx, dMga2L_dx, dMW2T_dx, dMZ2T_dx, dMga2T_dx};
                        mixing_angles = {dswL_dx, dswT_dx};
                    }
                    return std::make_pair(MSquares, mixing_angles);
                } 
                case ThermalMassScheme::dMdT: {
                    // if (std::abs(phi - cached_phi) > 1e-6 ) {
                    //     cached_phi = phi;
                    //     fitmass(X, T, cached_splines_phi, ThermalMassScheme::dMdT);
                    // }
                    // double Mh2, dMh2_dx, d2Mh2_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[0], T, Mh2, dMh2_dx, d2Mh2_dx2);
                    // double MG2, dMG2_dx, d2MG2_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[1], T, MG2, dMG2_dx, d2MG2_dx2);
                    // double MA2, dMA2_dx, d2MA2_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[2], T, MA2, dMA2_dx, d2MA2_dx2);
                    // double MH2, dMH2_dx, d2MH2_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[3], T, MH2, dMH2_dx, d2MH2_dx2);
                    // double MHpm2, dMHpm2_dx, d2MHpm2_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[4], T, MHpm2, dMHpm2_dx, d2MHpm2_dx2);
                    // double MW2L, dMW2L_dx, d2MW2L_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[5], T, MW2L, dMW2L_dx, d2MW2L_dx2);
                    // double MZ2L, dMZ2L_dx, d2MZ2L_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[6], T, MZ2L, dMZ2L_dx, d2MZ2L_dx2);
                    // double Mga2L, dMga2L_dx, d2Mga2L_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[7], T, Mga2L, dMga2L_dx, d2Mga2L_dx2);
                    // double MW2T, dMW2T_dx, d2MW2T_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[8], T, MW2T, dMW2T_dx, d2MW2T_dx2);
                    // double MZ2T, dMZ2T_dx, d2MZ2T_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[9], T, MZ2T, dMZ2T_dx, d2MZ2T_dx2);
                    // double Mga2T, dMga2T_dx, d2Mga2T_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[10], T, Mga2T, dMga2T_dx, d2Mga2T_dx2);
                    // double swL, dswL_dx, d2swL_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[11], T, swL, dswL_dx, d2swL_dx2);
                    // double swT, dswT_dx, d2swT_dx2;
                    // alglib::spline1ddiff(cached_splines_phi[12], T, swT, dswT_dx, d2swT_dx2);
                    double Mh2, dMh2_dx, dMh2_dt;
                    alglib::spline2ddiff(mass_splines_.Mh2, phi, T, Mh2, dMh2_dx, dMh2_dt);
                    double MG2, dMG2_dx, dMG2_dt;
                    alglib::spline2ddiff(mass_splines_.MG2, phi, T, MG2, dMG2_dx, dMG2_dt);
                    double MA2, dMA2_dx, dMA2_dt;
                    alglib::spline2ddiff(mass_splines_.MA2, phi, T, MA2, dMA2_dx, dMA2_dt);
                    double MH2, dMH2_dx, dMH2_dt;
                    alglib::spline2ddiff(mass_splines_.MH2, phi, T, MH2, dMH2_dx, dMH2_dt);
                    double MHpm2, dMHpm2_dx, dMHpm2_dt;
                    alglib::spline2ddiff(mass_splines_.MHpm2, phi, T, MHpm2, dMHpm2_dx, dMHpm2_dt);
                    double MW2L, dMW2L_dx, dMW2L_dt;
                    alglib::spline2ddiff(mass_splines_.MW2L, phi, T, MW2L, dMW2L_dx, dMW2L_dt);
                    double MZ2L, dMZ2L_dx, dMZ2L_dt;
                    alglib::spline2ddiff(mass_splines_.MZ2L, phi, T, MZ2L, dMZ2L_dx, dMZ2L_dt);
                    double Mga2L, dMga2L_dx, dMga2L_dt;
                    alglib::spline2ddiff(mass_splines_.Mga2L, phi, T, Mga2L, dMga2L_dx, dMga2L_dt);
                    double dMW2T_dt = 0.0;
                    //alglib::spline2ddiff(mass_splines_.MW2T, phi, T, MW2T, dMW2T_dx, dMW2T_dt);
                    double dMZ2T_dt = 0.0;
                    //alglib::spline2ddiff(mass_splines_.MZ2T, phi, T, MZ2T, dMZ2T_dx, dMZ2T_dt);
                    double dMga2T_dt = 0.0;
                    //alglib::spline2ddiff(mass_splines_.Mga2T, phi, T, Mga2T, dMga2T_dx, dMga2T_dt);
                    double swL, dswL_dx, dswL_dt;
                    alglib::spline2ddiff(mass_splines_.swL, phi, T, swL, dswL_dx, dswL_dt);
                    double dswT_dt = 0.0;
                    //alglib::spline2ddiff(mass_splines_.swT, phi, T, swT, dswT_dx, dswT_dt);
                    std::vector<double> MSquares = {dMh2_dt, dMG2_dt, dMA2_dt, dMH2_dt, dMHpm2_dt, dMW2L_dt, dMZ2L_dt, dMga2L_dt, dMW2T_dt, dMZ2T_dt, dMga2T_dt};
                    std::vector<double> mixing_angles = {dswL_dx, dswT_dt};
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
            //std::cout << correction << std::endl;
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

            // Boson contribution
            auto bosons = boson_massSq(X, T, ms);
            auto nb = boson_dofs();
            auto& mb2 = bosons.first;
            if (ResumScheme == ResummationScheme::DYQ) {
                SelfEnergy selfenergy(lam2, lamL, mA, mH, mHpm);
                const double mt2_init = 0.5 * square(yt) * square(phi);
                std::vector<double> MB2(13), Pi(11);
                for (int i = 0; i < 8; ++i) {
                    MB2[i] = mb2[i];
                }
                MB2[8] = bosons.second[0];
                MB2[9] = mb2[8];
                MB2[10] = mb2[9];
                MB2[11] = mb2[10];
                MB2[12] = bosons.second[1];
                Pi[0] = selfenergy.Pih3_safe(T, MB2, mt2_init, phi);
                Pi[1] = selfenergy.PiG3(T, MB2, mt2_init, phi);
                Pi[2] = selfenergy.PiA3(T, MB2, phi);
                Pi[3] = selfenergy.PiH3(T, MB2, phi);
                Pi[4] = selfenergy.PiHpm3(T, MB2, phi);
                Pi[5] = selfenergy.PiWL3(T, MB2, mt2_init, phi);
                Pi[6] = square(MB2[8]) * selfenergy.Pi44L3(T, MB2, mt2_init, phi) 
                        + (1-square(MB2[8]))  * selfenergy.Pi33L3(T, MB2, mt2_init, phi) 
                        - 2*MB2[8]*std::sqrt(1-square(MB2[8]))  * selfenergy.Pi34L3(T, MB2, mt2_init, phi);
                Pi[7] = (1-square(MB2[8])) * selfenergy.Pi44L3(T, MB2, mt2_init, phi) 
                        + square(MB2[8]) * selfenergy.Pi33L3(T, MB2, mt2_init, phi) 
                        + 2*MB2[8]*std::sqrt(1-square(MB2[8]))  * selfenergy.Pi34L3(T, MB2, mt2_init, phi);
                Pi[8] = 0.0;
                Pi[9] = 0.0;
                Pi[10] = 0.0;
                for (size_t i = 2; i < 11; ++i) {
                    y += 0.5 * nb[i] * cb[i] *  Ijbp(mb2[i], T, 1)  ;
                    if (mb2[i] >=0.0) y += - 0.5 * nb[i] * cb[i] *0.25*T*std::sqrt(mb2[i])/M_PI;
                }
                // UV terms for transverse modes (indices 8, 9, 10 correspond to mW2T, mZ2T, mga2T)
                y += cb[8] * UV_term(mb2[8], -2.0, 1);
                y += 0.5 * cb[9] * UV_term(mb2[9], -2.0, 1);
                y += 0.5 * cb[10] * UV_term(mb2[10], -2.0, 1);
                //higgs nonzero mode contribution
                y += 0.5 * nb[0] * cb[0] * Ijbp(mb2[0], T, 1);
                // higgs zero mode contribution
                Params_qawc p;
                p.mh2 = mb2[0];
                p.mG2 = mb2[1];
                p.phi = phi;
                p.M2 = mb2[0] + Pi[0];
                p.T = T;
                p.t0 = 0.0;
                p.lam1 = lam1;
                // Gauss-Kronrod 兜底积分：用于 QAWC 失败时或不需要 CPV 时
                auto gauss_kronrod_integral = [&p]() -> double {
                    using boost::math::quadrature::gauss_kronrod;
                    auto integrand = [&p](double t) -> double {
                        double k = p.T*t/(1-t);
                        double PiR = - 4.5*square(p.lam1*p.phi)*ReI20(p.mh2, k, p.T) - 1.5*square(p.lam1*p.phi)*ReI20(p.mG2, k, p.T);
                        double PiI = - 4.5*square(p.lam1*p.phi)*ImI20(p.mh2, k, p.T) - 1.5*square(p.lam1*p.phi)*ImI20(p.mG2, k, p.T);
                        double fk = (p.M2 + PiR ) * (k*k + p.M2 + PiR) + PiI * PiI;
                        double gk = square(k*k + p.M2 + PiR) + square(PiI);
                        return fk/(square(1-t)*gk);
                    };
                    return gauss_kronrod<double, 15>::integrate(integrand, 0.0, 1.0, 20, 1e-4);
                };

                if ((mb2[1]>0.0 && mb2[0]>0.0 && (std::sqrt(mb2[1])*std::sqrt(mb2[0])*p.M2 - std::sqrt(mb2[1])*4.5*square(lam1*phi)*0.25*T/(M_PI* 2.) - std::sqrt(mb2[0])*1.5*square(lam1*phi)*0.25*T/(M_PI* 2.)) < 0)||(mb2[1]==0.0 && mb2[0]>=0.0)||(mb2[1]>=0.0 && mb2[0]==0.0)) {
                    double t0 = find_t0(p);
                    p.t0 = t0;
                    // Cauchy PV via singularity subtraction:
                    //   PV∫ G(t)/(t-t0) dt = G0*ln((1-t0)/t0) + ∫ [G(t)-G0]/(t-t0) dt
                    // 第二项被积函数光滑（t→t0 时趋于 G'(t0)），Gauss-Kronrod 指数收敛，
                    // 避免 QAWC 耗尽子区间预算、也避免 GK 直接对极点深度细分。
                    using boost::math::quadrature::gauss_kronrod;
                    const double k0 = T*t0/(1.0-t0);
                    const double f0  = p.M2 - 4.5*square(lam1*phi)*ReI20(mb2[0], k0, T) - 1.5*square(lam1*phi)*ReI20(mb2[1], k0, T);
                    const double gp0 = 2.0*k0 - 4.5*square(lam1*phi)*dI20dk(mb2[0], k0, T) - 1.5*square(lam1*phi)*dI20dk(mb2[1], k0, T);
                    const double G0  = f0/(T*gp0);   // G(t0)，与 G_qawc 正则化值一致
                    // G'(t0)：扣除项 [G(t)-G0]/(t-t0) 在 t→t0 的连续延拓值。
                    // 用中心差分从光滑一般式估计（避开 |dt|<1e-8 正则化区），O(eps^2) 精度
                    double Gp0 = 0.0;
                    const double eps = std::min({1e-6, 0.3*t0, 0.3*(1.0-t0)});
                    Gp0 = (G_qawc(t0+eps, &p) - G_qawc(t0-eps, &p))/(2.0*eps);
 

                    auto sub_integrand = [&p, G0, Gp0](double t) -> double {
                        const double dt = t - p.t0;
                        // t→t0 时极限为 G'(t0)，做连续延拓，保证被积函数在 t0 处光滑
                        if (std::abs(dt) < 1e-6) return Gp0;
                        return (G_qawc(t, &p) - G0)/dt;
                    };
                    double smooth_part = gauss_kronrod<double, 15>::integrate(sub_integrand, 0.0, 1.0, 15, 1e-4);
                    double CPV = G0*std::log((1.0-t0)/t0) + smooth_part;
                    y += -3.*lam1*phi*square(T)*CPV/(4.*M_PI*M_PI);
                    if (std::isnan(y)||std::isinf(y)) std::cout<< G_qawc(t0, &p) << std::endl;
                }
                else {
                    double integral = gauss_kronrod_integral();
                    y += -3.*lam1*phi*square(T)*integral/(4.*M_PI*M_PI);
                }
            }
            else {
                // Boson contribution: sum(0.5 * nb * cb * Ijb(M2, T, 1))
                for (size_t i = 0; i < 11; ++i) {
                    y += 0.5 * nb[i] * cb[i] * Ijb(mb2[i], T, 1);
                }
                // UV terms for transverse modes (indices 8, 9, 10 correspond to mW2T, mZ2T, mga2T)
                y += cb[8] * UV_term(mb2[8], -2.0, 1);
                y += 0.5 * cb[9] * UV_term(mb2[9], -2.0, 1);
                y += 0.5 * cb[10] * UV_term(mb2[10], -2.0, 1);
            }

            return y;
        }

        double d2V1_dXdT(Eigen::VectorXd X, double T) const {
            double phi = X[0];
            double y = 0.0;

            // Fermion contribution (top quark)
            std::vector<double> mf2 = fermion_massSq(X);  
            std::vector<double> nf = fermion_dofs();  
            std::vector<double> ct = {square(yt) * phi}; 
            for (size_t i = 0; i < nf.size(); ++i) {
                double x = mf2[i] / square(T);
                y -= 0.5 * nf[i] * ct[i] * (- 2 * T * J_F(x, 1) + 2 * x * T * J_F(x, 2)) / square(M_PI);
            }    

            // Boson contribution
            auto bosons = boson_massSq(X, T, ThermalMassScheme::Exact);
            auto nb = boson_dofs();
            const auto& mb2 = bosons.first;
            auto dbosons_dt = boson_massSq(X, T, ThermalMassScheme::dMdT);
            const auto& dmb2_dt = dbosons_dt.first;
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
            for (size_t i = 0; i < 11; ++i) {
                double x = mb2[i] / square(T);
                if (std::abs(cb[i]) <= 1e-10) continue;
                // if (i<2) y += 0.5 * nb[i] * cb[i] * ( 2 * T * J_B(x, 1) + (-2 * x * T ) * J_B(x, 2)) / square(M_PI);
                // else 
                y += 0.5 * nb[i] * cb[i] * ( 2 * T * J_B(x, 1) + (-2 * x * T + dmb2_dt[i]) * J_B(x, 2)) / square(M_PI);
            }
            // UV terms for transverse modes (indices 8, 9, 10 correspond to mW2T, mZ2T, mga2T)
            y += - cb[8] * UV_term(mb2[8], -2.0, 2) * dmb2_dt[8];
            y += - 0.5 * cb[9] * UV_term(mb2[9], -2.0, 2) * dmb2_dt[9];
            y += - 0.5 * cb[10] * UV_term(mb2[10], -2.0, 2) * dmb2_dt[10];

            return y;
        }

        double d2V1_dX2(Eigen::VectorXd X, double T) const {
            double phi = X[0];
            double y = 0.0;

            // Fermion contribution (top quark)
            std::vector<double> mf2 = fermion_massSq(X);  
            std::vector<double> nf = fermion_dofs();  
            std::vector<double> ct = {square(yt) * phi}; 
            std::vector<double> dct_dx = {square(yt)}; 
            std::vector<double> dmf2_dx = {square(yt) * phi};
            for (size_t i = 0; i < nf.size(); ++i) {
                y -= 0.5 * nf[i] * (dct_dx[i]*Ijf(mf2[i], T,  1) - ct[i]*Ijf(mf2[i], T, 2)*dmf2_dx[i]);
            }    

            // Boson contribution
            auto bosons = boson_massSq(X, T, ThermalMassScheme::Exact);
            auto nb = boson_dofs();
            const auto& mb2 = bosons.first;
            auto dbosons_dx = boson_massSq(X, T, ThermalMassScheme::dMdx);
            const auto& dmb2_dx = dbosons_dx.first;

            std::vector<double> cb(11);
            std::vector<double> dcb_dx(11);
            cb[0] = 3.0 * lam1 * phi;           // d(mh2)/d(phi)
            dcb_dx[0] = 3.0 * lam1;        
            cb[1] = lam1 * phi;                  // d(mG2)/d(phi)
            dcb_dx[1] = lam1;
            cb[2] = lamm * phi;                  // d(mA2)/d(phi)
            dcb_dx[2] = lamm;
            cb[3] = lamp * phi;                  // d(mH2)/d(phi)
            dcb_dx[3] = lamp;
            cb[4] = lam3 * phi;                  // d(mHpm2)/d(phi)
            dcb_dx[4] = lam3;
            cb[5] = 0.5 * square(g) * phi;      // d(mW2L)/d(phi)
            dcb_dx[5] = 0.5 * square(g);
            cb[6] = 0.5 * (square(g) + square(gp)) * phi;  // d(mZ2L)/d(phi)
            dcb_dx[6] = 0.5 * (square(g) + square(gp));
            cb[7] = 0.0;                         // d(mga2L)/d(phi) = 0
            dcb_dx[7] = 0.0;
            cb[8] = 0.5 * square(g) * phi;      // d(mW2T)/d(phi)
            dcb_dx[8] = 0.5 * square(g);
            cb[9] = 0.5 * (square(g) + square(gp)) * phi;  // d(mZ2T)/d(phi)
            dcb_dx[9] = 0.5 * (square(g) + square(gp));
            cb[10] = 0.0;                        // d(mga2T)/d(phi) = 0
            dcb_dx[10] = 0.0;

            // Boson contribution: sum(0.5 * nb * cb * Ijb(M2, T, 1))
            for (size_t i = 0; i < 11; ++i) {
                if (std::abs(cb[i]) <= 1e-10) continue;
                // if (i<2) y += 0.5 * nb[i] * dcb_dx[i]*Ijb(mb2[i], T,  1);
                //else 
                y += 0.5 * nb[i] * (dcb_dx[i]*Ijb(mb2[i], T,  1) - cb[i]*Ijb(mb2[i], T, 2)*dmb2_dx[i]);
            }
            // UV terms for transverse modes (indices 8, 9, 10 correspond to mW2T, mZ2T, mga2T)
            y += dcb_dx[8] * UV_term(mb2[8], -2.0, 1) - cb[8] * UV_term(mb2[8], -2.0, 2)*dmb2_dx[8];
            y += 0.5 * dcb_dx[9] * UV_term(mb2[9], -2.0, 1) - cb[9] * UV_term(mb2[9], -2.0, 2)*dmb2_dx[9];
            y += 0.5 * dcb_dx[10] * UV_term(mb2[10], -2.0, 1) - cb[10] * UV_term(mb2[10], -2.0, 2)*dmb2_dx[10];

            return y;
        }


        Eigen::VectorXd dV_dx(Eigen::VectorXd X, double T) const override {
            switch (ResumScheme) {
                case ResummationScheme::None:
                case ResummationScheme::Parwani:
                case ResummationScheme::ArnoldEspinosa: 
                case ResummationScheme::DolanJackiw:
                    return Potential::dV_dx(X, T);
                case ResummationScheme::DYQ: 
                case ResummationScheme::PartialDressing: {
                    double phi = X[0];
                    Eigen::VectorXd dVdX(1);
                    double dV1dX;
                    if (std::abs(cached_T - T) < 1e-4) {
                        if (phi < 0.0) {
                            dV1dX = -alglib::spline1dcalc(cached_dV1dX_splines_T, std::abs(phi));
                        } else {
                            dV1dX = alglib::spline1dcalc(cached_dV1dX_splines_T, std::abs(phi));
                        }
                    } else {
                        dV1dX = dV1_dX(X, T, ThermalMassScheme::Exact);
                    }
                    double dV0dX = mu1sq * phi + 0.5 * lam1 * cube(phi);
                    double dVCTdX = delta_mu1sq * phi + 0.5 * delta_lam1 * cube(phi);
                    dVdX(0) = dV0dX + dVCTdX + dV1dX;
                    // double val = V(X, T);
                    // double d1, d2;
                    // alglib::spline1ddiff(cached_V_spline, phi, val, d1, d2);
                    // dVdX(0) = d1;
                    return dVdX;
                }
                default: 
                    return Potential::dV_dx(X, T);
                
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
                    //return Potential::d2V_dxdt(X, T);
                    double phi = X[0];
                    Eigen::VectorXd result = Eigen::VectorXd::Zero(X.size());
                    
                    // for (int ii = 0; ii < X.size(); ++ii) {
                    //     for (int jj = 0; jj < n_h_xy.size(); ++jj) {
                    //         const double T_shifted = T + n_h_xy[jj] * h ;
                    //         result(ii) += dV1_dX(X, T_shifted, ThermalMassScheme::Exact) * coeff_xy[jj] / h;
                    //     }
                    // }
                    // if (std::abs(cached_phi - phi) > 1e-4) {
                    //     cached_phi = phi;
                    //     int n_points = 100;
                    //     double t_min = 10.0;
                    //     double t_max = 300.0;
                    //     alglib::real_1d_array t_arr;
                    //     t_arr.setlength(n_points);
                    //     for (int i = 0; i < n_points; ++i) {
                    //         t_arr[i] = t_min + (t_max - t_min) * i / (n_points - 1);
                    //     }
                    //     alglib::real_1d_array hessian_arr;
                    //     hessian_arr.setlength(n_points);
                    //     for (int i = 0; i < n_points; ++i) {
                    //         hessian_arr[i] = d2V1_dXdT(X, t_arr[i]);
                    //     }
                    //     //double rho = 0.1;
                    //     alglib::spline1dfitreport rep;
                    //     //alglib::spline1dfit(t_arr, hessian_arr, 5, rho, cached_dV1dX_splines_phi, rep);
                    //     alglib::spline1dbuildcubic(t_arr, hessian_arr, cached_dV1dX_splines_phi);
                    // }
                    // double dV1PD_dx, d2V1PD_dxdt, d2s;
                    // alglib::spline1ddiff(cached_dV1dX_splines_phi, T, dV1PD_dx, d2V1PD_dxdt, d2s);
                    // result(0) = d2V1PD_dxdt;
                    result(0) += d2V1_dXdT(X, T);
                    return result;
                }
                default: 
                    return Potential::d2V_dxdt(X, T);
            }
        }

        Eigen::MatrixXd d2V_dx2(Eigen::VectorXd X, double T) const override { 
            switch (ResumScheme) {
                case ResummationScheme::None:
                case ResummationScheme::Parwani:
                case ResummationScheme::ArnoldEspinosa:
                case ResummationScheme::DolanJackiw:
                    return Potential::d2V_dx2(X, T);
                case ResummationScheme::DYQ:
                case ResummationScheme::PartialDressing: {
                    //return Potential::d2V_dx2(X, T);
                    // Numerical second derivative using analytical first derivative
                    // Mimics the implementation of Potential::d2V_dx2 but uses dV1_dX instead of V
                    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(X.size(), X.size());
                    
                    // for (int ii = 0; ii < X.size(); ++ii) {
                    //     Eigen::VectorXd X_shifted = X;
                    //     for (int jj = 0; jj < n_h_xy.size(); ++jj) {
                    //         X_shifted(ii) = X(ii) + n_h_xy[jj] * h;
                    //         hessian(ii, ii) += dV1_dX(X_shifted, T, ThermalMassScheme::Exact) * coeff_xy[jj] / h;
                    //     }
                    // }
                    // if (std::abs(cached_T - T) > 1e-4) {
                    //     cached_T = T;
                    //     int n_points = 1000;
                    //     double x_min = -1.0;
                    //     double x_max = 400.0;
                    //     alglib::real_1d_array x_arr;
                    //     x_arr.setlength(n_points);
                    //     for (int i = 0; i < n_points; ++i) {
                    //         x_arr[i] = x_min + (x_max - x_min) * i / (n_points - 1);
                    //     }
                    //     alglib::real_1d_array hessian_arr;
                    //     hessian_arr.setlength(n_points);
                    //     for (int i = 0; i < n_points; ++i) {
                    //         Eigen::VectorXd x(1);
                    //         x << x_arr[i];
                    //         hessian_arr[i] = dV1_dX(x, T, ThermalMassScheme::Exact);
                    //     }
                    //     //double rho = 0.01;
                    //     alglib::spline1dfitreport rep;
                    //     alglib::spline1dbuildcubic(x_arr, hessian_arr, cached_dV1dX_splines_T);
                    // }
                    // double dV1PD_dx, d2V1PD_dx2, d2s;
                    // alglib::spline1ddiff(cached_dV1dX_splines_T, X[0], dV1PD_dx, d2V1PD_dx2, d2s);
                    // hessian(0, 0) += d2V1PD_dx2;
                    make_dV1dX_spline(T);
                    double d2V1dX2, d2s, dV1dX;
                    alglib::spline1ddiff(cached_dV1dX_splines_T, std::abs(X[0]), dV1dX, d2V1dX2, d2s);
                    hessian(0, 0) += d2V1dX2;
                    hessian(0, 0) +=  mu1sq + 1.5 * lam1 * square(X[0]);
                    hessian(0, 0) += delta_mu1sq + 1.5 * delta_lam1 * square(X[0]);
                    
                    return hessian;
                }
                default: 
                    return Potential::d2V_dx2(X, T);
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
                case ResummationScheme::DYQ:
                case ResummationScheme::PartialDressing: {
                    // If phi equals reference point phi= 0, return 0
                    using boost::math::quadrature::gauss_kronrod;
                    double phi = std::abs(X[0]);
                    if (std::abs(phi) < std::numeric_limits<double>::min()) {
                        return 0.0;
                    }

                    // if (std::abs(cached_T - T) > 1e-4) {
                    //     cached_T = T;
                    //     int n_points = 2000;
                    //     double x_min = 0.0;
                    //     double x_max = 400.0;
                    //     alglib::real_1d_array x_arr;
                    //     x_arr.setlength(n_points);
                    //     for (int i = 0; i < n_points; ++i) {
                    //         x_arr[i] = x_min + (x_max - x_min) * i / (n_points - 1);
                    //     }
                    //     alglib::real_1d_array hessian_arr;
                    //     hessian_arr.setlength(n_points);
                    //     for (int i = 0; i < n_points; ++i) {
                    //         // Define the integrand as a lambda function
                    //         auto integrand = [&](double x) -> double {
                    //             Eigen::VectorXd X_point(1);
                    //             X_point[0] = x;
                    //             return dV1_dX(X_point, T, ThermalMassScheme::Exact);
                    //         };
                    //         // Perform integration using Gauss-Kronrod quadrature
                    //         // 15-point rule, relative tolerance 1e-4, max iterations 50
                    //         double V1_integrated = gauss_kronrod<double, 15>::integrate(integrand, 0.0, x_arr[i], 50, 1e-6);
                    //         hessian_arr[i] = V1_integrated;
                    //     }
                    //     alglib::spline1dbuildcubic(x_arr, hessian_arr, cached_V_spline);
                    // }         
                    // double V1 = alglib::spline1dcalc(cached_V_spline, X[0]);

                    make_dV1dX_spline(T);

                    // auto integrand = [&](double x) -> double {
                    //             Eigen::VectorXd X_point(1);
                    //             X_point[0] = x;
                    //             return dV1_dX(X_point, T, ThermalMassScheme::Exact);
                    //         };
                    //double V1_integrated = gauss_kronrod<double, 15>::integrate(integrand, 0.0, phi, 50, 1e-4);
                    double V1_integrated = alglib::spline1dintegrate(cached_dV1dX_splines_T, phi);
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
            for (int i = 0; i < 8; ++i) {
                MB2_vec[i] = Mb2[i];
            }
            MB2_vec[8] = sw[0];
            MB2_vec[9] = Mb2[8];
            MB2_vec[10] = Mb2[9];
            MB2_vec[11] = Mb2[10];
            MB2_vec[12] = sw[1];
            double eq1 = - dV1_dX(vev, 0.0, ThermalMassScheme::Tree) / v0;
            //std::cout << eq1 << std::endl;
            double deltaZ = selfenergy.DPih3_Dp2(square(mh), MB2_vec, square(mt)) ;
            double se_term = selfenergy.Pih3Pdependent(square(mh), MB2_vec, square(mt), v0) + selfenergy.Pih4(.0, MB2_vec);
            double eq2 = deltaZ * square(mh) - se_term;
            delta_lam1 =  (eq2 - eq1)  / square(v0);
            delta_mu1sq = (3.0*eq1 - eq2) / 2.0;
            delta_lam14 = - (delta_mu1sq + selfenergy.Pih4(.0, MB2_vec) ) / (1.5 * square(v0)) ;

            delta_lamp = - Gamma_hHH();
            delta_mu2sq = - (selfenergy.PiH3(.0, MB2_vec, v0) + selfenergy.PiH4(.0, MB2_vec) + 0.5 * delta_lamp * square(v0));
            delta_lam3 = - 2.0 * (selfenergy.PiHpm3(.0, MB2_vec, v0) + selfenergy.PiHpm4(.0, MB2_vec) + delta_mu2sq) / square(v0);
            delta_lamm = - 2.0 * (selfenergy.PiA3(.0, MB2_vec, v0) + selfenergy.PiA4(.0, MB2_vec) + delta_mu2sq) / square(v0);
            delta_lamm4 = - (delta_mu2sq + selfenergy.PiA4(.0, MB2_vec) ) / (0.5 * square(v0)) ;
            delta_lamp4 = - (delta_mu2sq + selfenergy.PiH4(.0, MB2_vec) ) / (0.5 * square(v0)) ;
            delta_lam34 = - (delta_mu2sq + selfenergy.PiHpm4(.0, MB2_vec) ) / (0.5 * square(v0)) ;

        }


 
        /**
         * Initialize mass splines by solving gap equations on a grid asynchronously.
         * Saves results to disk and loads them into spline interpolators.
         */
        void init_mass_splines (bool PD) {
            double phimin = 0.0, phimax = 400.0;
            double Tmin = 50.0, Tmax = 300.0;
            int n_phi = 400, n_T = 250;
            std::string spline_data_path = "/home/dayun/data"; // Adjust path as needed
            std::filesystem::create_directories(spline_data_path);     
            std::string M2_dat_path;
            std::string bad_points_path;
            if (PD) {
                M2_dat_path = spline_data_path + "/M2_PD_" + paramNumber + ".txt";
                bad_points_path = spline_data_path + "/bad_points_PD_" + paramNumber + ".txt";
            }
            else {
                M2_dat_path = spline_data_path + "/M2_" + paramNumber + ".txt";
                bad_points_path = spline_data_path + "/bad_points_" + paramNumber + ".txt";
            }
            const int n_mass = 9; // mh2, mG2, mA2, mH2, mHpm2, mW2L, mZ2L, mga2L, swL
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
                auto worker = [this, n_mass, &yM_flat, &bad_points_list, &bad_points_mutex, &completed_points, total_points, start_time, PD](
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
                            gapEqResult result = selfenergy.solve_gap_equations(phi, T, 1e-2, bosons_bare, bosons_init, 300, PD); // 1% tolerance
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
            create_mass_splines(xphi, xT, yM_flat, n_phi, n_T, n_mass, bad_points_path);
            //std::cout << "Mass splines has created." << std::endl;
        }

    private:

        mutable double cached_T = -1e10;
        mutable double cached_phi = -1e10;
        //mutable std::array<alglib::spline1dinterpolant, 13> cached_splines_T;
        //mutable std::array<alglib::spline1dinterpolant, 13> cached_splines_phi;
        mutable alglib::spline1dinterpolant cached_dV1dX_splines_T;
        mutable alglib::spline1dinterpolant cached_dV1dX_splines_phi;
        mutable alglib::spline1dinterpolant cached_V_spline;

        MassSplines mass_splines_;

        void make_dV1dX_spline(double T) const {
            if (std::abs(cached_T - T) > 1e-4) {
                int n_points = 2000;
                double x_min = 0.0;
                double x_max = 400.0;
                alglib::real_1d_array x_arr;
                alglib::real_1d_array y_arr;
                x_arr.setlength(n_points);
                y_arr.setlength(n_points);
                for (int i = 0; i < n_points; ++i) {
                    x_arr[i] = x_min + (x_max - x_min) * i / (n_points - 1);
                    y_arr[i] = dV1_dX(Eigen::VectorXd::Constant(1, x_arr[i]), T, ThermalMassScheme::Exact);
                    if (std::isnan(y_arr[i])||std::isinf(y_arr[i])) std::cout << "Warning: dV1_dX is nan or inf at x = " << x_arr[i] << ", T = " << T << std::endl;
                }
                try {
                    alglib::spline1dbuildcubic(x_arr, y_arr, cached_dV1dX_splines_T);
                } catch (const alglib::ap_error& e)
                {
                    std::cout << e.msg << std::endl;
                    std::terminate();
                }
                cached_T = T;
            }
        }
        
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
                            int n_phi, int n_T, int n_mass, std::string bad_points_path) {
            // Build bad point mask
            std::vector<std::vector<bool>> is_bad(n_phi, std::vector<bool>(n_T, false));

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
                            int start_j = std::max(0, j - 25);
                            int end_j = std::min(n_T, j + 26);
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
                            int start_i = std::max(0, i - 25);
                            int end_i = std::min(n_phi, i + 26);
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

                // alglib::real_1d_array f_alglib;
                // f_alglib.setcontent(n_phi * n_T, f_flat.data());
                alglib::real_2d_array xy;
                xy.setlength(n_phi * n_T, 3); 
                for (int j = 0; j < n_T; ++j) {
                    for (int i = 0; i < n_phi; ++i) {
                        int idx = j * n_phi + i;
                        xy(idx, 0) = xphi_alglib[i];
                        xy(idx, 1) = xT_alglib[j];
                        xy(idx, 2) = f_flat[idx];
                    }
                }

                // 3.5 Build bicubic spline (vector version, D=1)
                alglib::ae_int_t d = 1;
                double lambdav = 0.01;
                alglib::ae_int_t n_points = n_phi * n_T;
                alglib::spline2dbuilder state;
                alglib::spline2dbuildercreate(d, state);
                alglib::spline2dbuildersetpoints(state, xy, n_points);
                alglib::spline2dbuildersetgrid(state, 10, 10);
                alglib::spline2dbuildersetalgoblocklls(state, lambdav);

                alglib::spline2dfitreport rep;
                alglib::spline2dfit(state, mass_splines_.get(k), rep);
                
                //std::cout << "Begin create Mass fit " << k << "." << std::endl;
                // 准备散乱点 (x, y, z) – 列优先遍历
                // alglib::real_2d_array xy;
                // xy.setlength(n_phi * n_T, 3);
                // for (int j = 0; j < n_T; ++j) {
                //     for (int i = 0; i < n_phi; ++i) {
                //         int idx = j * n_phi + i;
                //         xy[idx][0] = xphi_alglib[i];
                //         xy[idx][1] = xT_alglib[j];
                //         xy[idx][2] = f_flat[idx];   // f_flat 已经是列优先
                //     }
                // }

                // alglib::rbfmodel rbf;
                // alglib::rbfreport rep;
                // alglib::rbfcreate(2, 1, rbf);
                // alglib::rbfsetpoints(rbf, xy);
                // double rbase = 100.0;  // 数据范围: phi[0,400], T[0,300]
                // int nlayers = 3;    // 层数，可调整
                // alglib::rbfsetalgomultilayer(rbf, rbase, nlayers);
                // alglib::rbfbuildmodel(rbf, rep);

                // // 存储模型（假设你有一个容器 alglib::rbfmodel 数组）
                // mass_rbfs_.get(k) = rbf;
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
            std::cout << "Spline built successfully.  " << std::endl;
        }

        double Gamma_hHH() const {
            // Define the integrand using a Lambda expression
            auto integrand = [this](double x) -> double {
                double factor = x * (1.0 - x) * square(mh);
                
                double mh2star     = square(mh) - factor;
                double mG2star     =  - factor;
                double mA2star     = square(mA) - factor;
                double mH2star     = square(mH) - factor;
                double mHpm2star   = square(mHpm) - factor;
                double mW2star    =  0.25*square(g) *square(v0)- factor;
                double mZ2star    =  0.25*(square(g) + square(gp) )*square(v0) - factor;

                double result = 0.0;
                result += - 1.5 * lamp * lam1 * Ijb(mh2star, 0.0, 2) ;
                result += - (0.5 * lamm + lam3) * lam1 * Ijb(mG2star, 0.0, 2) ;
                result += - 0.5 * lam2 * lamm * Ijb(mA2star, 0.0, 2) ;
                result += - 1.5 * lam2 * lamp * Ijb(mH2star, 0.0, 2) ;
                result += - lam2 * lam3 * Ijb(mHpm2star, 0.0, 2) ;
                result -= 0.125 * square(square(g)+ square(gp)) * (3.0*Ijb(mZ2star, 0.0, 2) + UV_term(mZ2star, -2.0, 2) );
                result -= 0.25 * pow_4(g) * (3.0*Ijb(mW2star, 0.0, 2) + UV_term(mW2star, -2.0, 2) );

                //std::cout << "DEBUG: Gamma_hHH() result = " << result << std::endl;
                return result;
            };
            
            double result = boost::math::quadrature::gauss_kronrod<double, 15>::integrate(integrand, 0.0, 1.0, 20, 1e-4);
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
        double delta_lam14 = 0;
        double delta_mu1sq = 0;
        double delta_mu2sq = 0;
        double delta_lamm = 0;
        double delta_lamp = 0;
        double delta_lam3 = 0;
        double delta_lamm4 = 0;
        double delta_lamp4 = 0;
        double delta_lam34 = 0;

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
            IDM_DR(double lam2_input_, double lamL_input_, double mA_input_, double mH_input_, double mHpm_input_, ConterTermParameters counterterm_params_) {
                lam2_input = lam2_input_;
                lamL_input = lamL_input_;
                mA_input = mA_input_;
                mH_input = mH_input_;
                mHpm_input = mHpm_input_;
                lam1_input = square(mh/v0) + counterterm_params_.delta_lam1;
                mu1sq_input = - 0.5 * square(mh) + counterterm_params_.delta_mu1sq;

                double delta_lam4 = 0.5*( counterterm_params_.delta_lamp + counterterm_params_.delta_lamm) - counterterm_params_.delta_lam3;
                double delta_lam5 = 0.5*( counterterm_params_.delta_lamp - counterterm_params_.delta_lamm) ;
                lam4_input = (square(mA_input) + square(mH_input) - 2 * square(mHpm_input))/ square(v0) + delta_lam4;
                lam5_input = (square(mH_input) - square(mA_input))/ square(v0) + delta_lam5;
                lam3_input = 2 * lamL_input - lam4_input - lam5_input + counterterm_params_.delta_lam3;
                mu2sq_input = 0.5* (2 * square(mHpm_input) - lam3_input * square(v0)) + counterterm_params_.delta_mu2sq;
                lamp_input = lam3_input + lam4_input + lam5_input + counterterm_params_.delta_lamp;
                lamm_input = lam3_input + lam4_input - lam5_input + counterterm_params_.delta_lamm;


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
                double lambdaVLL_1 = (square(gsq) * T) / (4. * square(M_PI));// 2*lam-WWWW
                double lambdaVLL_2 = -(gsq * gpsq * T) / (4. * square(M_PI));// lam-WWBB
                double lambdaVLL_3 = -(181. * square(gpsq) * T) / (36. * square(M_PI));// 6*lam-BBBB
                double lambdaVLL_4 = -(3. * gsq * gssq * T) / (4. * square(M_PI)); // lam-WWGG 
                double lambdaVLL_5 = -(11. * gpsq * gssq * T) / (12. * square(M_PI)); // lam-BBGG
                //double lambdaVLL_6 = -(sqrt(gssq* gpsq)  * gssq * T) / (4. * square(M_PI));
                double lambdaVLL_7 = (square(gssq) * T) / (4. * square(M_PI));// 2*lam-GGGG
                double lambdaVL_1 = -((gssq * square(yt) * T) / (4. * square(M_PI)));// lam-phi-phi-GG
                double lamppgg = lambdaVL_1;
                double lambdaVL_2 = (sqrt(gsq) * sqrt(gpsq) * T) / (192. * square(M_PI)) * (
                    gsq * (5. + 21. * Lb - 12. * Lf) - 
                    gpsq * (-21. + Lb + 20. * Lf) + 
                    12. * (8. * square(M_PI) + square(yt) + lam1 + lam4)
                ); // lam-phi-phi-BW
                double lamppbw = lambdaVL_2;
                double lambdaVL_3 = (sqrt(gsq) * sqrt(gpsq) * T) / (192. * square(M_PI)) * (
                    gsq * (5. + 21. * Lb - 12. * Lf) - 
                    gpsq * (-21. + Lb + 20. * Lf) + 
                    12. * (8. * square(M_PI) + lam2 + lam4)
                );  // lam-eta-eta-BW
                double lambdaVL_4 = -1. / (192. * square(M_PI)) * gpsq * T * (
                    -9. * gsq + gpsq * (-39. + 2. * Lb + 40. * Lf) - 
                    4. * (24. * square(M_PI) - 17. * square(yt) + 9. * lam1 + 6. * lam3 + 3. * lam4)
                ); // lam-phi-phi-BB
                double lamppbb = lambdaVL_4;
                double lambdaVL_5 = -1. / (192. * square(M_PI)) * gpsq * T * (
                    -9. * gsq + gpsq * (-39. + 2. * Lb + 40. * Lf) - 
                    12. * (8. * square(M_PI) + 3. * lam2 + 2. * lam3 + lam4)
                ); // lam-eta-eta-BB
                double lambdaVL_6 = (gsq * T) / (192. * square(M_PI)) * (
                    gsq * (73. + 42. * Lb - 24. * Lf) + 
                    3. * (gpsq + 4. * (8. * square(M_PI) - 3. * square(yt) + 3. * lam1 + 2. * lam3 + lam4))
                ); // lam-phi-phi-WW
                double lamppww = lambdaVL_6;
                double lambdaVL_7 = (gsq * T) / (192. * square(M_PI)) * (
                    gsq * (73. + 42. * Lb - 24. * Lf) + 
                    3. * (gpsq + 4. * (8. * square(M_PI) + 3. * lam2 + 2. * lam3 + lam4))
                ); // lam-eta-eta-WW
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
                        8. * gpsq3d * (3. * lam23d + 2. * lam33d + lam43d) - 
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
                // if ( matching_flag == 0 ) {
                //     return {gpsq*T, gsq*T, gssq*T, lam1*T, mu1sq3d_LO, lam2*T, lam3*T, lam4*T, lam5*T, mu2sq3d_LO};
                // } else if ( matching_flag == 1 ) {
                //     return {gpsq3d, gsq3d, gssq3d, lam13d, mu1sq3d, lam23d, lam33d, lam43d, lam53d, mu2sq3d};
                // }
                if ( matching_flag == 0 ) {
                     return {gpsq*T, gsq*T, gssq*T, lam1*T, mu1sq3d_LO, lam2*T, lam3*T, lam4*T, lam5*T, mu2sq3d_LO, musqSU3_LO, musqSU2_LO, musqU1_LO, 0., 0., 0., 0.};
                }
                if ( matching_flag == 1 ) {
                    return {gpsq3d, gsq3d, gssq3d, lam13d, mu1sq3d, lam23d, lam33d, lam43d, lam53d, mu2sq3d, musqSU3, musqSU2, musqU1, lamppww, lamppbb, lamppbw, lamppgg};
                }
                //---------------------------------
                // integrate out temporal scalar
                //---------------------------------
                // couplings
                double RG_scale_3DUS = gsq * T;

                // integrate out the second doublet
                double lam13dUS = lam13d - (1. / (16. * M_PI)) * (
                    (2. * (2. * square(lam33d) + 2. * lam33d * lam43d + square(lam43d) + square(lam53d))) / sqrt(mu2sq3d) + 
                    (8. * square(lambdaVL_1)) / sqrt(musqSU3) + 
                    (4. * square(lambdaVL_2)) / (sqrt(musqSU2) + sqrt(musqU1)) + 
                    square(lambdaVL_4) / sqrt(musqU1) + 
                    (3. * square(lambdaVL_6)) / sqrt(musqSU2)
                );

                double gsq3dUS = gsq3d - (square(gsq3d) * (1. / sqrt(mu2sq3d) + 2. / sqrt(musqSU2))) / (48. * M_PI);
                double gpsq3dUS = gpsq3d - square(gpsq3d) / (48. * M_PI * sqrt(mu2sq3d));
                double gssq3dUS = gssq3d - square(gssq3d) / (16. * M_PI * sqrt(musqSU3));
                // masses
                double mu1sq3dUS_LO = mu1sq3d - (1. / (8. * M_PI)) * (
                    4. * lam33d * sqrt(mu2sq3d) + 
                    2. * lam43d * sqrt(mu2sq3d) + 
                    8. * sqrt(musqSU3) * lambdaVL_1 + 
                    sqrt(musqU1) * lambdaVL_4 + 
                    3. * sqrt(musqSU2) * lambdaVL_6
                );
                double mu1sq3dUS_NLO = 1. / (128. * square(M_PI)) * (
                    6. * gsq3d * lam33d + 2. * gpsq3d * lam33d + 
                    24. * lam23d * lam33d - 8. * square(lam33d) + 
                    3. * gsq3d * lam43d + gpsq3d * lam43d + 
                    12. * lam23d * lam43d - 8. * lam33d * lam43d - 
                    8. * square(lam43d) - 12. * square(lam53d) - 
                    (3. * square(gsq3d) + square(gpsq3d) - 
                    12. * gsq3d * (2. * lam33d + lam43d) - 
                    4. * gpsq3d * (2. * lam33d + lam43d) + 
                    8. * (2. * square(lam33d) + 2. * lam33d * lam43d + 
                        2. * square(lam43d) + 3. * square(lam53d))) * log(RG_scale_3D / (2. * sqrt(mu2sq3d))) + 
                    48. * gsq3d * lambdaVL_1 + 
                    192. * gsq3d * log(RG_scale_3D / (2. * sqrt(musqSU3))) * lambdaVL_1 - 
                    16. * square(lambdaVL_1) - 
                    32. * log(RG_scale_3D / (2. * sqrt(musqSU3))) * square(lambdaVL_1) - 
                    12. * square(lambdaVL_2) - 
                    24. * log(RG_scale_3D / (sqrt(musqSU2) + sqrt(musqU1))) * square(lambdaVL_2) - 
                    2. * square(lambdaVL_4) - 
                    4. * log(RG_scale_3D / (2. * sqrt(musqU1))) * square(lambdaVL_4) + 
                    12. * gsq3d * lambdaVL_6 - 6. * square(lambdaVL_6) - 
                    6. * log(RG_scale_3D / (2. * sqrt(musqSU2))) * (square(gsq3d) - 8. * gsq3d * lambdaVL_6 + 
                    2. * square(lambdaVL_6)) + 
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
                double mu1sq3dUS_beta = 1. / (256. * square(M_PI)) * (
                    -51. * square(gsq3dUS) + 
                    5. * square(gpsq3dUS) + 
                    18. * gsq3dUS * (gpsq3dUS - 4. * lam13dUS) - 
                    24. * gpsq3dUS * lam13dUS + 
                    48. * square(lam13dUS)
                );
                double mu1Sq3dUS = mu1sq3dUS_LO + mu1sq3dUS_NLO + mu1sq3dUS_beta * log(RG_scale_3DUS / RG_scale_3D);

// 保留 惰性二重态
                // double lam13dUS = lam13d - (1. / (16. * M_PI)) * (
                //     (8. * lambdaVL_1 * lambdaVL_1) / sqrt(musqSU3) + 
                //     (4. * lambdaVL_2 * lambdaVL_2) / (sqrt(musqSU2) + sqrt(musqU1)) + 
                //     (lambdaVL_4 * lambdaVL_4) / sqrt(musqU1) + 
                //     (3. * lambdaVL_6 * lambdaVL_6) / sqrt(musqSU2)
                // );
                // double lam23dUS = lam23d - (1. / (16. * M_PI)) * (
                //     (4. * lambdaVL_3 * lambdaVL_3) / (sqrt(musqSU2) + sqrt(musqU1)) + 
                //     (lambdaVL_5 * lambdaVL_5) / sqrt(musqU1) + 
                //     (3. * lambdaVL_7 * lambdaVL_7) / sqrt(musqSU2)
                // );
                // double lam33dUS = lam33d - (1. / (16. * M_PI)) * (
                //     - (4. * lambdaVL_2 * lambdaVL_3) / (sqrt(musqSU2) + sqrt(musqU1)) + 
                //     (lambdaVL_4 * lambdaVL_5) / sqrt(musqU1) + 
                //     (3. * lambdaVL_6 * lambdaVL_7) / sqrt(musqSU2)
                // );
                // double lam43dUS = lam43d - (lambdaVL_2 * lambdaVL_3) / (2. * M_PI * (sqrt(musqSU2) + sqrt(musqU1)));
                // double lam53dUS = lam53d;
                // double gsq3dUS = gsq3d - (gsq3d * gsq3d) / (24. * M_PI * sqrt(musqSU2));
                // double gpsq3dUS = gpsq3d;
                // double gssq3dUS = gssq3d - (gssq3d * gssq3d) / (16. * M_PI * sqrt(musqSU3));

                // double mu1Sq3dUS_LO = mu1sq3d - (8. * sqrt(musqSU3) * lambdaVL_1 + sqrt(musqU1) * lambdaVL_4 + 3. * sqrt(musqSU2) * lambdaVL_6) / (8. * M_PI);
                // double mu1Sq3dUS_NLO = (1. / (128. * square(M_PI))) * (
                //     48. * gssq3d * lambdaVL_1 + 
                //     32. * log(RG_scale_3D / (2. * sqrt(musqSU3))) * (6. * gssq3d - lambdaVL_1) * lambdaVL_1 - 
                //     16. * lambdaVL_1 * lambdaVL_1 - 
                //     12. * lambdaVL_2 * lambdaVL_2 - 
                //     24. * log(RG_scale_3D / (sqrt(musqSU2) + sqrt(musqU1))) * lambdaVL_2 * lambdaVL_2 - 
                //     2. * lambdaVL_4 * lambdaVL_4 - 
                //     4. * log(RG_scale_3D / (2. * sqrt(musqU1))) * lambdaVL_4 * lambdaVL_4 + 
                //     12. * gsq3d * lambdaVL_6 - 
                //     6. * lambdaVL_6 * lambdaVL_6 - 
                //     6. * log(RG_scale_3D / (2. * sqrt(musqSU2))) * (gsq3d * gsq3d - 8. * gsq3d * lambdaVL_6 + 2. * lambdaVL_6 * lambdaVL_6) + 
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
                // double mu1Sq3dUS_beta = (1. / (256. * square(M_PI))) * (
                //     -45. * gsq3dUS * gsq3dUS + 
                //     7. * gpsq3dUS * gpsq3dUS + 
                //     48. * lam23dUS * lam23dUS - 
                //     8. * gpsq3dUS * (3. * lam23dUS + 2. * lam33dUS + lam43dUS) + 
                //     32. * (lam33dUS * lam33dUS + lam33dUS * lam43dUS + lam43dUS * lam43dUS) + 
                //     6. * gsq3dUS * (3. * gpsq3dUS - 4. * (3. * lam23dUS + 2. * lam33dUS + lam43dUS)) + 
                //     48. * lam53dUS * lam53dUS
                // );
                // double mu1Sq3dUS = mu1Sq3dUS_LO + mu1Sq3dUS_NLO + mu1Sq3dUS_beta * log(RG_scale_3DUS / RG_scale_3D);
                // double mu2sq3dUS_LO = mu2sq3d - (sqrt(musqU1) * lambdaVL_5 + 3. * sqrt(musqSU2) * lambdaVL_7) / (8. * M_PI);
                // double mu2sq3dUS_NLO = (1. / (128. * square(M_PI))) * (
                //     -12. * lambdaVL_3 * lambdaVL_3 - 
                //     24. * log(RG_scale_3D / (sqrt(musqSU2) + sqrt(musqU1))) * lambdaVL_3 * lambdaVL_3 - 
                //     2. * lambdaVL_5 * lambdaVL_5 - 
                //     4. * log(RG_scale_3D / (2. * sqrt(musqU1))) * lambdaVL_5 * lambdaVL_5 + 
                //     12. * gsq3d * lambdaVL_7 - 
                //     6. * lambdaVL_7 * lambdaVL_7 - 
                //     6. * log(RG_scale_3D / (2. * sqrt(musqSU2))) * (gsq3d * gsq3d - 8. * gsq3d * lambdaVL_7 + 2. * lambdaVL_7 * lambdaVL_7) + 
                //     15. * lambdaVL_7 * lambdaVLL_1 + 
                //     (3. * sqrt(musqSU2) * lambdaVL_5 * lambdaVLL_2) / sqrt(musqU1) + 
                //     (3. * sqrt(musqU1) * lambdaVL_7 * lambdaVLL_2) / sqrt(musqSU2) + 
                //     lambdaVL_5 * lambdaVLL_3 + 
                //     (24. * sqrt(musqSU3) * lambdaVL_7 * lambdaVLL_4) / sqrt(musqSU2) + 
                //     (8. * sqrt(musqSU3) * lambdaVL_5 * lambdaVLL_5) / sqrt(musqU1)
                // );
                // double mu2sq3dUS_beta = (1. / (256. * square(M_PI))) * (
                //     -45. * gsq3dUS * gsq3dUS + 
                //     7. * gpsq3dUS * gpsq3dUS + 
                //     48. * lam13dUS * lam13dUS - 
                //     8. * gpsq3dUS * (3. * lam13dUS + 2. * lam33dUS + lam43dUS) + 
                //     32. * (lam33dUS * lam33dUS + lam33dUS * lam43dUS + lam43dUS * lam43dUS) + 
                //     6. * gsq3dUS * (3. * gpsq3dUS - 4. * (3. * lam13dUS + 2. * lam33dUS + lam43dUS)) + 
                //     48. * lam53dUS * lam53dUS
                // );
                // double mu2sq3dUS = mu2sq3dUS_LO + mu2sq3dUS_NLO + mu2sq3dUS_beta * log(RG_scale_3DUS / RG_scale_3D);

                if ( matching_flag == 2 ) {
                    //return {gpsq3dUS, gsq3dUS, gssq3dUS, lam13dUS, lam23dUS, lam33dUS, lam43dUS, lam53dUS, mu1Sq3dUS, mu2sq3dUS};
                    return {gpsq3d, gsq3d, gssq3d, lam13d, mu1sq3d, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
                }
                return {0., 0., 0., 0., 0., 0., 0., 0., 0.0, 0.0};
            }

            double V(Eigen::VectorXd X, double T) const override {
    
                const std::vector<double> par = get_3d_parameters(T);
                double gpsq = par[0];
                double gsq = par[1];
                double gssq = par[2];
                double lam1 = par[3];
                //std::cout << "lam1 = " << lam1 << std::endl;
                double mu1sq = par[4];
                double lam2 = par[5];
                double lam3 = par[6];
                double lam4 = par[7];
                double lam5 = par[8];
                double mu2sq = par[9];
                double musqSU3 = par[10];
                double musqSU2 = par[11];
                double musqU1 = par[12];
                double lamppww = par[13];
                double lamppbb = par[14];
                double lamppbw = par[15];
                double lamppgg = par[16];

                double phi = X[0]/sqrt(T + 1e-15);
                // B0W30 mixing
                double M33 = 0.5 * lamppww * phi * phi + musqSU2;
                double M44 = 0.5 * lamppbb * phi * phi + musqU1;
                double M34 = - lamppbw * phi * phi; 
                Eigen::Matrix2d M2matrix;
                M2matrix << M33, M34,
                             M34, M44;
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(M2matrix);
                double Mga2L = solver.eigenvalues()[0];
                double MZ2L = solver.eigenvalues()[1];

               

                double veffLO = 0.5* pow(phi,2.)*mu1sq + 0.125*pow(phi,4.)*lam1 ;
                

                if ( potential_flag == 0 ) {
                    return T * veffLO;
                }

                double veffNLO = 
                    - pow(std::max(mu1sq + 0.5 * lam1 * phi * phi, 0.0), 1.5) / (4. * M_PI) - 
                    pow(std::max(mu1sq + 1.5 * lam1 * phi * phi, 0.0), 1.5) / (12. * M_PI) - 
                    pow(std::max(mu2sq + 0.5 * lam3 * phi * phi, 0.0), 1.5) / (6. * M_PI) - 
                    pow(std::max(mu2sq + 0.5 * (lam3 + lam4 - lam5) * phi * phi, 0.0), 1.5) / (12. * M_PI) - 
                    pow(std::max(mu2sq + 0.5 * (lam3 + lam4 + lam5) * phi * phi, 0.0), 1.5) / (12. * M_PI) - 
                    2. * ( pow(std::max(gsq * phi * phi, 0.0), 1.5) / (48. * M_PI) + pow(std::max((gsq + gpsq) * phi * phi, 0.0), 1.5) / (96. * M_PI)) ;
                    8. * pow(std::max(musqSU3 + 0.5* lamppgg* phi * phi, 0.0), 1.5 )/(12. * M_PI) - 
                    pow(std::max(musqSU2 + 0.5* lamppww* phi * phi, 0.0), 1.5 )/(6. * M_PI) -
                    pow(std::max(Mga2L, 0.0), 1.5 )/(12. * M_PI) - pow(std::max(MZ2L, 0.0), 1.5 )/(12. * M_PI);

                // ensure the normalization is correct, that is, when phi = 0, Veff = 0
                double veff_norm = pow(std::max(mu1sq, 0.0) , 1.5) / (4. * M_PI) + 
                    pow(std::max(mu1sq, 0.0) , 1.5) / (12. * M_PI) + 
                    pow(std::max(mu2sq, 0.0) , 1.5) / (6. * M_PI) + 
                    pow(std::max(mu2sq, 0.0) , 1.5) / (12. * M_PI) + 
                    pow(std::max(mu2sq, 0.0) , 1.5) / (12. * M_PI) ;
                    8. * pow(std::max(musqSU3, 0.0) , 1.5) / (12. * M_PI) + 
                    pow(std::max(musqSU2, 0.0) , 1.5) / (6. * M_PI) +
                    pow(std::max(musqSU2, 0.0) , 1.5) / (12. * M_PI) + 
                    pow(std::max(musqU1, 0.0) , 1.5) / (12. * M_PI);


                if ( potential_flag == 1 ) {
                    return T * veffLO + T * veffNLO + T * veff_norm;
                }

                // if ( potential_flag == 2 ) {
                //     DRIDM_NNLO VNNLO;
                //     std::complex<double> veffNNLO = VNNLO.V2(X, T, par);
                //     return T * veffLO.real() + T * veffNLO.real() + T * veffNNLO.real();
                // }

                return 0.;
            }


    };



    



   




}  // namespace EffectivePotential

#endif