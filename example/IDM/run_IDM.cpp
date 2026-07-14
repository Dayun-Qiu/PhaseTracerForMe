#include "IDM.hpp"
#include "phase_finder.hpp"
#include "transition_finder.hpp"
#include "gravwave_calculator.hpp"
#include <boost/log/trivial.hpp>
#include <iostream>
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

void phase_parameters(PhaseTracer::TransitionFinder &tf, double &alpha, double &beta_H) {
    // create grav wave calculator object
    PhaseTracer::GravWaveCalculator gc(tf);

    // find all spectrums
    gc.set_dof(110.75);
    auto spectrums = gc.calc_spectrums();
    alpha = spectrums[0].alpha;
    beta_H = spectrums[0].beta_H;
    //std::cout << gc;
}

phase_params calculate_phasetransition(EffectivePotential::Potential &model) {

    // Make PhaseFinder object and find the phases
    PhaseTracer::PhaseFinder pf(model);
    pf.set_upper_bounds({400.0});
    pf.set_lower_bounds({-400.0});
    pf.set_guess_points({Eigen::VectorXd::Constant(1, 246.22)});
    pf.set_t_high(300.0);
    pf.set_t_low(10.0);
    pf.set_v(246.22);
    pf.set_check_hessian_singular(true);
    pf.set_check_vacuum_at_high(true);
    
    
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

    if (t.size() < 1) {
        // std::cerr << "DEBUG: No transitions found for mA=" << mA << " mH=" << mH << std::endl; 
        result.voT = std::nan("");
        result.beta_H = std::nan("");
        result.alpha = std::nan("");
        return result;
    }

    if (t[0].true_vacuum_TN.size() == 0) {
        result.voT = std::nan("");
        result.beta_H = std::nan("");
        result.alpha = std::nan("");
        return result;
    }

    double alpha, beta_H;
    phase_parameters(tf, alpha, beta_H);

    double nuc_temp = t[0].TN;
    double nuc_vev1 =  t[0].true_vacuum_TN[0];
    double nuc_vev2 =  t[0].false_vacuum_TN[0];
    std::cout << t[0].TN << " " << t[0].TC <<std::endl;
    
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
    double phi = 106.265664;  
    double T =  120.401338;  
    auto params = model.get_params();
    EffectivePotential::SelfEnergy self_energy(params.lam2, params.lamL, params.mA, params.mH, params.mHpm);
    auto bosons_init = model.boson_massSq(Eigen::VectorXd::Constant(1, phi), T, EffectivePotential::ThermalMassScheme::Tree); // Initial guess
    auto bosons_bare = model.boson_massSq(Eigen::VectorXd::Constant(1, phi), T, EffectivePotential::ThermalMassScheme::CTterm);
    EffectivePotential::gapEqResult result = self_energy.solve_gap_equations(phi, T, 1e-2, bosons_bare, bosons_init, 100);
    std::cout << "messages:" << result.message << std::endl;
    std::cout << "loss:" << result.loss << std::endl;
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
        EffectivePotential::IDM_DR idm(lam2, lamL, mA, mH, mHpm);
        // 生成势能数据
        // std::vector<double> phis(100), Vs(100);
        // for (int i = 0; i < 100; ++i) {
        //     phis[i] = 0.9 * (i+1) ;
        //     Vs[i] = idm.V(Eigen::VectorXd::Constant(1, phis[i]), 300.);
        //     // std::vector<double> params = idm.get_3d_parameters(300);
        //     // std::cout << "lam1:" << params[3] << std::endl;
        // }

        // // 保存数据
        // std::ofstream fout("/home/dayun/data/Vs.txt", std::ofstream::out | std::ofstream::trunc);
        // fout << "# phi V\n";
        // for (int i = 0; i < 100; ++i) {
        //     fout << phis[i] << " " << Vs[i] << "\n";
        // }
        // fout.close();

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
            idm.init_mass_splines();
        } else {
            std::cerr << "Warning: Invalid resummation scheme '" << resum << "'. Defaulting to no resummation." << std::endl;
            idm.set_resummation_scheme(EffectivePotential::ResummationScheme::None);
        }
        // 生成势能数据
        // std::vector<double> phis(100), Vs(100);
        // //auto mass_splines_ = idm.get_mass_splines();

        // for (int i = 0; i < 100; ++i) {
        //     phis[i] = 3*i +1;
        //     Vs[i] = idm.d2V_dx2(Eigen::VectorXd::Constant(1, phis[i]), 117.499)(0,0);
        //     //Vs[i] = alglib::rbfcalc2(mass_rbfs_.get(3), phis[i], 110.);
        //     //Vs[i] = idm.boson_massSq(Eigen::VectorXd::Constant(1, phis[i]), 10., EffectivePotential::ThermalMassScheme::dMdx).first[4];
        // }


        // // 保存数据
        // std::ofstream fout("/home/dayun/data/Vs.txt", std::ofstream::out | std::ofstream::trunc);
        // fout << "# phi V\n";
        // for (int i = 0; i < 100; ++i) {
        //     fout << phis[i] << " " << Vs[i] << "\n";
        // }
        // fout.close();
        //plot_data(lam2, lamL, mA, mH, mHpm);
        //test(idm);
        //return 0;
        result = calculate_phasetransition(idm);
    }


    if (std::isnan(result.voT)) {
        std::cout << "nan" << " " << "nan" << " " << "nan" << std::endl;
    } else {
        std::cout << result.voT << " " << result.alpha << " " << result.beta_H << std::endl;
    }
    //plot_data(lam2, lamL, mA, mH, mHpm);

    return 0;
}

