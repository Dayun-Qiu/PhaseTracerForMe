#include "IDM.hpp"
#include "example/IDM/SelfEnergy.hpp"
#include "phase_finder.hpp"
#include "thermal_funcs.h"
#include "transition_finder.hpp"
#include "gravwave_calculator.hpp"
#include "thermal_function.hpp"
#include "derivatives.h"
#include <boost/log/trivial.hpp>
#include <iostream>
#include <gsl/gsl_errno.h>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <vector>


struct phase_params {
    double voT;
    double alpha;
    double beta_H; 
};

bool parse_command_line_args(int argc, char *argv[], double& lam2, double& lamL, double& mA, double& mH, double& mHpm, std::string& paramNumber, std::string& Resum) {

    std::map<std::string, std::string> args_map;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) == "--" && i + 1 < argc) {
            std::string key = arg.substr(2);
            std::string value = argv[i + 1];
            args_map[key] = value;
            i++; 
        }
    }

    try {
        auto get_double = [&args_map](const std::string& key, double default_val) -> double {
            auto it = args_map.find(key);
            if (it != args_map.end()) {
                try {
                    return std::stod(it->second);
                } catch (const std::exception&) {
                    std::cerr << "Warning: Invalid double value for key '" << key << "'. Using default." << std::endl;
                    return default_val;
                }
            }
            return default_val;
        };
        
        auto get_string = [&args_map](const std::string& key, const std::string& default_val) -> std::string {
            if (args_map.find(key) != args_map.end()) {
                return args_map.at(key);
            }
            return default_val;
        };

        // Use default values if arguments are not provided
        lam2 = get_double("lam2", lam2);
        lamL = get_double("lamL", lamL);
        mA = get_double("mA", mA);
        mH = get_double("mH", mH);
        mHpm = get_double("mHpm", mHpm);
        paramNumber = get_string("paramNumber", paramNumber);
        Resum = get_string("Resum", Resum);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
        return false;
    }
}

// 只保留从高温对称相出发的物理相变：false vacuum 在 Tnuc 处位于原点 (|phi| < tol)。
// 影子转换（亚稳支之间的 0-1、无势垒的 1-2、高势能支之间的 3-4）的 false vacuum 都远离原点，会被排除。
static std::vector<PhaseTracer::Transition> physical_transitions(
    const std::vector<PhaseTracer::Transition> &t, double tol = 0.5) {
    std::vector<PhaseTracer::Transition> out;
    for (const auto &ti : t) {
        if (ti.false_vacuum_TN.size() == 0 || ti.true_vacuum_TN.size() == 0) continue;
        // 物理相变：两个真空之一必须是高温对称相（原点）。
        // 影子转换（0-1、1-2、3-4）的两端真空都远离原点，被排除。
        const double v1 = ti.true_vacuum_TN[0];
        const double v2 = ti.false_vacuum_TN[0];
        if (std::min(std::abs(v1), std::abs(v2)) > tol) continue;
        out.push_back(ti);
    }
    return out;
}

void phase_parameters(PhaseTracer::TransitionFinder &tf, double &alpha, double &beta_H,
                      const std::vector<PhaseTracer::Transition> &keep) {
    // create grav wave calculator object
    PhaseTracer::GravWaveCalculator gc(tf);

    // 只对物理转换计算谱
    gc.set_transitions(keep);

    // find all spectrums
    gc.set_dof(110.75);
    try {
        auto spectrums = gc.calc_spectrums();
        alpha = spectrums[0].alpha;
        beta_H = spectrums[0].beta_H;
    } catch (const std::exception &e) {
        std::cerr << "[Warning] GW spectrum calculation failed: " << e.what() << std::endl;
        alpha = std::nan("");
        beta_H = std::nan("");
    }
    //std::cout << gc;
}

phase_params calculate_phasetransition(EffectivePotential::Potential &model) {

    // Make PhaseFinder object and find the phases
    PhaseTracer::PhaseFinder pf(model);
    pf.set_upper_bounds({400.0});
    pf.set_lower_bounds({-400.0});
    pf.set_guess_points({Eigen::VectorXd::Constant(1, 246.22)});
    pf.set_t_high(300.0);
    pf.set_t_low(50.0); // Jb(x) , x < 3
    pf.set_v(246.22);
    pf.set_check_hessian_singular(false);
    // 放宽场跳变判据：样条 wiggle 造成的伪双谷合并（如 150.7->157.6, 跳变 4.6%）会被误判为
    // 新相，把同一物理相劈裂成两个，导致 0<->2 共存窗口被截短、成核温度搜索失败（输出 NaN）。
    // 默认 x_rel_jump=1e-2；取 5e-2 使阈值 = 0.5 + 0.05*157.6 = 8.38 > 6.9，伪跳变被吸收。
    pf.set_x_rel_jump(0.05);
    pf.set_check_vacuum_at_high(true);
    pf.set_check_vacuum_at_low(false);
    
    
    // auto mass_splines = idm.get_mass_splines();
    // std::cout << "Test..." << std::endl;
    // std::cout << alglib::spline2dcalc(mass_splines.Mh2, 246.22, 0.0) << std::endl;
    // std::cout << "Done!" << std::endl;

    pf.find_phases();
    //std::cout <<pf << std::endl;  // 暂时禁用打印

    // create action calculator object
    PhaseTracer::ActionCalculator ac(model);
    //ac.set_action_calculator(PhaseTracer::ActionMethod::BubbleProfiler);



    //createtransitionfinderobject
    PhaseTracer::TransitionFinder tf(pf, ac);
    //tf.set_check_subcritical_transitions(true);

    //findtransitions
    tf.find_transitions();
    //std::cout << "TransitionFinder completed. " << std::endl;

    // extract transitions
    auto t = tf.get_transitions();
    //std::cout <<tf;
    //std::cout << "计算完成。 "  << std::endl;
    phase_params result;

    // 只保留从高温对称相出发的物理相变（影子转换被排除）
    auto t_phys = physical_transitions(t);
    if (t_phys.size() < 1) {
        result.voT = std::nan("");
        result.beta_H = std::nan("");
        result.alpha = std::nan("");
        return result;
    }

    if (t_phys[0].true_vacuum_TN.size() == 0) {
        result.voT = std::nan("");
        result.beta_H = std::nan("");
        result.alpha = std::nan("");
        return result;
    }

    double alpha, beta_H;
    phase_parameters(tf, alpha, beta_H, t_phys);

    double nuc_temp = t_phys[0].TN;
    double nuc_vev1 =  t_phys[0].true_vacuum_TN[0];
    double nuc_vev2 =  t_phys[0].false_vacuum_TN[0];
    std::cout << std::fixed << std::setprecision(6) << t_phys[0].TN << " " << t_phys[0].TC << std::endl;
    
    // Avoid division by zero
    if (nuc_temp <= 1e-6) {
        result.voT = std::nan("");
        result.beta_H = std::nan("");
        result.alpha = std::nan("");
        return result;
    }

    result.voT = std::max(nuc_vev1, nuc_vev2) / nuc_temp;
    result.beta_H = beta_H;
    result.alpha = alpha;
    return result;
}


void test(EffectivePotential::IDM &model) {
    double phi = 50;  
    double T =  100.0;  
    auto params = model.get_params();
    EffectivePotential::SelfEnergy self_energy(params.lam2, params.lamL, params.mA, params.mH, params.mHpm);
    auto bosons_init = model.boson_massSq(Eigen::VectorXd::Constant(1, phi), T, EffectivePotential::ThermalMassScheme::Tree); // Initial guess
    auto bosons_bare = model.boson_massSq(Eigen::VectorXd::Constant(1, phi), T, EffectivePotential::ThermalMassScheme::CTterm);
    EffectivePotential::gapEqResult result = self_energy.solve_gap_equations(phi, T, 1e-2, bosons_bare, bosons_init, 300);
    std::cout << "messages:" << result.message << std::endl;
    std::cout << "loss:" << result.loss << std::endl;
    //std::cout << "x: " << result.x[8] << " " << result.x[9] << " " << result.x[10] << std::endl;  
}


// void plot_data(double lam2, double lamL, double mA, double mH, double mHpm) {
//     EffectivePotential::IDM idm;
//     idm.init_params(lam2, lamL, mA, mH, mHpm, "1");
//     idm.calc_conterterms();
//     idm.init_mass_splines();
//     double phimin = 0.0, phimax = 400.0;
//     double Tmin = 0.0, Tmax = 300.0;
//     int n_phi = 800, n_T = 600;
//     const int n_mass = 13; // mh2, mG2, mA2, mH2, mHpm2, mW2L, mZ2L, mga2L, mW2T, mZ2T, mga2T, swL, swT
//     // Generate grid coordinates
//     std::vector<double> xphi(n_phi);
//     std::vector<double> xT(n_T);
//     for (int i = 0; i < n_phi; ++i) xphi[i] = phimin + i * (phimax - phimin) / (n_phi - 1);
//     for (int j = 0; j < n_T; ++j) xT[j] = Tmin + j * (Tmax - Tmin) / (n_T - 1);
//     // 构建样条
//     EffectivePotential::MassSplines mass_splines = idm.get_mass_splines();
//     //EffectivePotential::MassRBFs mass_rbfs_ = idm.get_mass_rbfs();
//     for (int k = 0; k < n_mass; ++k) {
//         std::string interp_file = "/home/dayun/data/interp_" + std::to_string(k) + ".txt";
//         std::ofstream fout(interp_file);
//         if (!fout) {
//             std::cerr << "Cannot write to " << interp_file << std::endl;
//             continue;
//         }
//         fout << "# phi T interp\n";
//         for (int i = 0; i < n_phi; ++i) {
//             for (int j = 0; j < n_T; ++j) {
//                 double interp = alglib::spline2dcalc(mass_splines.Mh2, xphi[i], xT[j]);
//                 fout << xphi[i] << " " << xT[j] << " " << interp << "\n";
//             }
//         }
//         std::cout << "Interpolation values written to " << interp_file << std::endl;
//         fout.close();
//     }
// }


int main(int argc, char *argv[]) {
    // 关掉 GSL 默认错误处理器（默认会 abort），让我们用返回码处理失败
    gsl_set_error_handler_off();

    //std::cout << EffectivePotential::J_F(0) << std::endl;
    // Set default values for parameters
    double lam2 = 0.2;
    double lamL = 0.0015;
    double mA = 300.0;
    double mH = 65.0;
    double mHpm = 300.0;
    std::string paramNumber = "1";
    std::string resum = "None";

    bool input_loaded = false;

    if (argc > 1) {
        input_loaded = parse_command_line_args(argc, argv, lam2, lamL, mA, mH, mHpm, paramNumber, resum);
        if (input_loaded) {
            std::cerr << "[Info] Parameters loaded from command line." << std::endl;
        } else {
            std::cerr << "[Warning] Failed to load parameters from command line. Using default values." << std::endl;
        }
    } else {
        std::cerr << "[Info] No command line arguments provided. Using default values." << std::endl;
    }

    LOGGER(debug);

    phase_params result;

    if (resum == "DR") {
        EffectivePotential::IDM idm_pre;
        // Initialize the IDM parameters
        idm_pre.init_params(lam2, lamL, mA, mH, mHpm, paramNumber);

        // Check perturbativity and vacuum stability
        if (!idm_pre.check_perturbativity()) {
            std::cerr << "DEBUG: Perturbativity check failed for mA=" << mA << " mH=" << mH << std::endl;
            std::cout << std::nan("")<< std::endl;
            return 1;
        }
        
        if (!idm_pre.check_vacuum_stability()) {
            std::cerr << "DEBUG: Vacuum stability check failed for mA=" << mA << " mH=" << mH << std::endl;
            std::cout << std::nan("")<< std::endl;
            return 1;
        }

        // Calculate the conterterms
        idm_pre.calc_conterterms();
        auto counterterm_params = idm_pre.get_counterterm_params();

        EffectivePotential::IDM_DR idm(lam2, lamL, mA, mH, mHpm, counterterm_params);
        // 生成势能数据
        std::vector<double> phis(500), Vs(500), Vs2(500), Vs3(500), Vs4(500), Vs5(500), Vs6(500);
        for (int i = 0; i < 500; ++i) {
            phis[i] = 0.7 * i ;
            Vs[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 60.);
            Vs2[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 90.);
            Vs3[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 120.);
            Vs4[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 150.);
            Vs5[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 180.);
            Vs6[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 240.);

            //std::cout << "phi:" << phis[i] << " V:" << Vs[i] << std::endl;
            //std::vector<double> params = idm.get_3d_parameters(300);
            //std::cout << "mu1sq:" << params[4] << std::endl;
        }
        std::vector<double> params = idm.get_3d_parameters(10);
        std::cout << "mu1sq:" << params[4] << std::endl;

        // 保存数据
        std::ofstream fout("/home/dayun/data/Vs.txt", std::ofstream::out | std::ofstream::trunc);
        fout << "# phi V\n";
        for (int i = 0; i < 500; ++i) {
            fout << phis[i] << " " << Vs[i] << " " << Vs2[i] << " " << Vs3[i] << " " << Vs4[i] << " " << Vs5[i] << " " << Vs6[i] << "\n";
        }
        fout.close();

        //return 0;
        result = calculate_phasetransition(idm);
    } else {
        EffectivePotential::IDM idm;
        // Initialize the IDM parameters
        idm.init_params(lam2, lamL, mA, mH, mHpm, paramNumber);

        // Check perturbativity and vacuum stability
        if (!idm.check_perturbativity()) {
            std::cerr << "DEBUG: Perturbativity check failed for mA=" << mA << " mH=" << mH << std::endl;
            std::cout << std::nan("")<< std::endl;
            return 1;
        }
        
        if (!idm.check_vacuum_stability()) {
            std::cerr << "DEBUG: Vacuum stability check failed for mA=" << mA << " mH=" << mH << std::endl;
            std::cout << std::nan("")<< std::endl;
            return 1;
        }

        // Calculate the conterterms
        idm.calc_conterterms();

        // Set resummation scheme
        if (resum == "None") {
            idm.set_resummation_scheme(EffectivePotential::ResummationScheme::None);
        } else if (resum == "AE") {
            idm.set_resummation_scheme(EffectivePotential::ResummationScheme::ArnoldEspinosa);
        } else if (resum == "Parwani") {
            idm.set_resummation_scheme(EffectivePotential::ResummationScheme::Parwani);
        } else if (resum == "DJ") {
            idm.set_resummation_scheme(EffectivePotential::ResummationScheme::DolanJackiw);
        } else if (resum == "PD") {
            idm.set_resummation_scheme(EffectivePotential::ResummationScheme::PartialDressing);
            idm.init_mass_splines(true);
        } else if (resum == "DYQ") {
            idm.set_resummation_scheme(EffectivePotential::ResummationScheme::DYQ);
            idm.init_mass_splines(false);
        } else {
            std::cerr << "Warning: Invalid resummation scheme '" << resum << "'. Defaulting to no resummation." << std::endl;
            idm.set_resummation_scheme(EffectivePotential::ResummationScheme::None);
        }
        // 生成势能数据
        std::vector<double> phis(1000), Vs1(500), Vs2(500), Vs3(500), Vs4(500), Vs5(500), Vs6(500);
        //auto mass_splines_ = idm.get_mass_splines();
        idm.set_h_4(true);
        auto params = idm.get_params();
        std::cout << "lam1:" << params.lam1 << std::endl;
        std::cout << "mu1sq:" << params.mu1sq << std::endl;
        std::cout << "mu2sq:" << params.mu2sq << std::endl;
        std::cout << "lam3:" << params.lam3 << std::endl;
        std::cout << "lam4:" << params.lam4 << std::endl;
        std::cout << "lam5:" << params.lam5 << std::endl;
        EffectivePotential::SelfEnergy self_energy(params.lam2, params.lamL, params.mA, params.mH, params.mHpm);
        std::vector<double> MB2_vec(13);
        for (int i = 0; i < 500; ++i) {
            phis[i] = 0.7 *i ;
            //Vs[i] = D2_J_B_bessel(phis[i]);
            Vs1[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 60.) - idm.V(Eigen::VectorXd::Constant(1, 0.), 60.);
            Vs2[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 79.99 ) - idm.V(Eigen::VectorXd::Constant(1, 0.), 79.99 );
            Vs3[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 96.01  ) - idm.V(Eigen::VectorXd::Constant(1, 0.), 96.01 );
            Vs4[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 120) - idm.V(Eigen::VectorXd::Constant(1, 0.), 120);
            Vs5[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 150.) - idm.V(Eigen::VectorXd::Constant(1, 0.), 150);
            Vs6[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 240.) - idm.V(Eigen::VectorXd::Constant(1, 0.), 240);
            //Vs2[i] = idm.dV_dx(Eigen::VectorXd::Constant(1, phis[i]), 90)[0];
            //std::cout << Vs1[i] << std::endl;
            //Vs1[i] = alglib::spline2dcalc(idm.get_mass_splines().get(0), phis[i], 72.5);
            //Vs2[i] = alglib::spline2dcalc(idm.get_mass_splines().get(1), phis[i], 72.5);
            //Vs1[i] =  EffectivePotential::J_B(square(phis[i]/200.), 1);
            //Vs2[i] =  EffectivePotential::J_B(phis[i]/10., 1);
            // Vs[i] = idm.boson_massSq(Eigen::VectorXd::Constant(1, phis[i]), 10., EffectivePotential::ThermalMassScheme::dMdx).first[4];
            // auto [Mb2, sw] = idm.boson_massSq(Eigen::VectorXd::Constant(1, phis[i]), 300, EffectivePotential::ThermalMassScheme::Exact);
            // auto [Mbt2, swt] = idm.boson_massSq(Eigen::VectorXd::Constant(1, phis[i]), 100, EffectivePotential::ThermalMassScheme::CTterm);
            // for (int j = 0; j < 8; ++j) {
            //     MB2_vec[j] = Mb2[j];
            // }
            // MB2_vec[8] = sw[0];
            // MB2_vec[9] = Mb2[8];
            // MB2_vec[10] = Mb2[9];
            // MB2_vec[11] = Mb2[10];
            // MB2_vec[12] = sw[1];
            //Vs1[i] = Mbt2[0] + self_energy.Pih4(0., MB2_vec) +self_energy.Pih3(0., MB2_vec, 0.5*params.yt*params.yt*phis[i]*phis[i], phis[i]);
            // if (std::isnan(Vs2[i])||std::isinf(Vs2[i])) {
            //     for (int j = 0; j < 11; ++j) {
            //         std::cout << bosons_init[j] << " ";
            //     }
            //     std::cout << std::endl;
            // }
            //Vs2[i] = MB2_vec[0];//+ ;
        }
        //std::cout<< (EffectivePotential::J_B_algb(0.1, 1) - D1_J_B_taylor(0.1, 1))/EffectivePotential::J_B_algb(0.1, 1)<<std::endl;

        // 保存数据
        std::ofstream fout("/home/dayun/data/Vs.txt", std::ofstream::out | std::ofstream::trunc);
        fout << "# phi V V2\n";
        for (int i = 0; i < 500; ++i) {
            fout << phis[i] << " " << Vs1[i] << " " << Vs2[i] << " " << Vs3[i] << " " << Vs4[i] << " " << Vs5[i] << " " << Vs6[i]  <<"\n";
        }
        fout.close();
        // plot_data(lam2, lamL, mA, mH, mHpm);
        // test(idm);
        // std::cout << idm.dV_dx(Eigen::VectorXd::Constant(1, 0.7), 60.)(0) << std::endl;
        // std::cout << idm.V(Eigen::VectorXd::Constant(1, 0.7), 60.) << std::endl;
        //return 0;
        // auto [Mb2, sw] = idm.boson_massSq(Eigen::VectorXd::Constant(1, 0.), 100, EffectivePotential::ThermalMassScheme::HighT);
        // std::cout << Mb2[0] << std::endl;
        return 0;
        result = calculate_phasetransition(idm);
    }


    if (std::isnan(result.voT)) {
        std::cout << "nan" << " " << "nan" << " " << "nan" << std::endl;
    } else {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << result.voT << " " << result.alpha << " " << result.beta_H << std::endl;
    }
    //plot_data(lam2, lamL, mA, mH, mHpm);

    return 0;
}

