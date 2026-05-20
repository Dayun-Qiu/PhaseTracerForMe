#include "IDM.hpp"
#include "phase_finder.hpp"
#include "transition_finder.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>


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

double calculate_v_over_T(double lam2, double lamL, double mA, double mH, double mHpm, const std::string& paramNumber, const std::string& resum) {
    // Force fatal logging to avoid cluttering stdout which is parsed by Python
    LOGGER(fatal);

    EffectivePotential::IDM idm;
    // Initialize the IDM parameters
    idm.init_params(lam2, lamL, mA, mH, mHpm, paramNumber);

    // Check perturbativity and vacuum stability
    if (!idm.check_perturbativity()) {
        std::cerr << "DEBUG: Perturbativity check failed for mA=" << mA << " mH=" << mH << std::endl;
        return std::nan("");
    }
    
    if (!idm.check_vacuum_stability()) {
        std::cerr << "DEBUG: Vacuum stability check failed for mA=" << mA << " mH=" << mH << std::endl;
        return std::nan("");
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

    // Make PhaseFinder object and find the phases
    PhaseTracer::PhaseFinder pf(idm);
    pf.set_upper_bounds({400.0});
    pf.set_lower_bounds({-400.0});
    pf.set_t_high(300.0);
    pf.set_v(246.22);
    
    try {
        pf.find_phases();
    } catch (const std::exception& e) {
        std::cerr << "DEBUG: PhaseFinder exception for mA=" << mA << " mH=" << mH << ": " << e.what() << std::endl;
        return std::nan("");
    }
    
    // create action calculator object
    PhaseTracer::ActionCalculator ac(idm);

    //createtransitionfinderobject
    PhaseTracer::TransitionFinder tf(pf, ac);

    

    //findtransitions
    try {
        tf.find_transitions();
    } catch (const std::exception& e) {
        std::cerr << "DEBUG: TransitionFinder exception for mA=" << mA << " mH=" << mH << ": " << e.what() << std::endl;
        return std::nan("");
    }

    // extract transitions
    auto t = tf.get_transitions();
    if (t.size() < 1) {
        // std::cerr << "DEBUG: No transitions found for mA=" << mA << " mH=" << mH << std::endl; 
        return std::nan("");
    }

    if (t[0].true_vacuum_TN.size() == 0) {
        std::cerr << "DEBUG: Transition data invalid (empty vacuum vectors) for mA=" << mA << " mH=" << mH << std::endl;
        return std::nan("");
    }

    double nuc_temp = t[0].TN;
    double nuc_vev1 =  t[0].true_vacuum_TN[0];
    double nuc_vev2 =  t[0].false_vacuum_TN[0];
    
    // Avoid division by zero
    if (nuc_temp <= 1e-6) {
        std::cerr << "DEBUG: Nucleation temperature too low for mA=" << mA << " mH=" << mH << std::endl;
        return std::nan("");
    }

    return std::max(nuc_vev1, nuc_vev2) / nuc_temp;
}

int main(int argc, char *argv[]) { 
    // Force fatal logging to avoid cluttering stdout which is parsed by Python
    LOGGER(debug);

    double v_over_T;

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

    v_over_T = calculate_v_over_T(lam2, lamL, mA, mH, mHpm, paramNumber, resum);

    // Output only the result for Python to parse
    if (std::isnan(v_over_T)) {
        std::cout << "nan" << std::endl;
    } else {
        std::cout << v_over_T << std::endl;
    }

    return 0;
}

