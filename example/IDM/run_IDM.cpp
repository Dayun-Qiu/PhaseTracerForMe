#include "IDM.hpp"
#include "phase_finder.hpp"
#include "transition_finder.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>


int get_inputs(double& lam2, double& lamL, double& mA, double& mH, double& mHpm, std::string& paramNumber, std::string& Resum);

// 新增：核心计算函数，接受参数并返回 v_over_T
double calculate_v_over_T(double lam2, double lamL, double mA, double mH, double mHpm, const std::string& paramNumber, const std::string& resum) {
    // Force fatal logging to avoid cluttering stdout which is parsed by Python
    LOGGER(fatal);

    // 【新增】输入参数合法性检查，防止非法质量值导致后续计算崩溃
    if (mA <= 0.0 || mH <= 0.0 || mHpm <= 0.0) {
        std::cerr << "DEBUG: Invalid mass parameters (must be positive): mA=" << mA << ", mH=" << mH << ", mHpm=" << mHpm << std::endl;
        return std::nan("");
    }

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
    //pf.set_upper_bounds({400.0});
    //pf.set_lower_bounds({0.0});
    //pf.set_t_high(300.0);
    //pf.set_v(246.22);
    //pf.set_seed(486149);
    
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
        // 这可能只是表示没有发生相变，不一定是错误
        // std::cerr << "DEBUG: No transitions found for mA=" << mA << " mH=" << mH << std::endl; 
        return std::nan("");
    }

    // 【修正】检查 t[0] 中的关键数据成员是否为空，防止访问空容器导致 Segmentation Fault
    // 原逻辑错误：if (size || size) 意味着如果有数据则报错。
    // 正确逻辑：如果数据为空，则无法访问 [0]，应返回 NaN。
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

    double lam2, lamL, mA, mH, mHpm;
    std::string paramNumber, resum;

    // Get input parameters from file
    if (get_inputs(lam2, lamL, mA, mH, mHpm, paramNumber, resum) != 0) {
        std::cout << "nan" << std::endl;
        return 0;
    }

    // 调用新的核心函数
    v_over_T = calculate_v_over_T(lam2, lamL, mA, mH, mHpm, paramNumber, resum);

    // Output only the result for Python to parse
    if (std::isnan(v_over_T)) {
        std::cout << "nan" << std::endl;
    } else {
        std::cout << v_over_T << std::endl;
    }

    return 0;
}

int get_inputs(double& lam2, double& lamL, double& mA, double& mH, double& mHpm, std::string& paramNumber, std::string& Resum) {
    std::ifstream f("/home/dayun/IDM_input.txt");
    if (!f) {
        std::cerr << "Error: cannot open input file\n";
        return 1;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key, value;
        if (!std::getline(iss, key, '=')) continue;
        
        std::getline(iss, value);  

        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
        };
        trim(key); trim(value);
        
        if (key == "paramNumber") paramNumber = value;
        else if (key == "Resum") Resum = value ;
        else {
            try {
                double d = std::stod(value);
                if (key == "lam2") lam2 = d;
                else if (key == "lamL") lamL = d;
                else if (key == "mA") mA = d;
                else if (key == "mH") mH = d;
                else if (key == "mHpm") mHpm = d;
            } catch (const std::exception& e) {
                std::cerr << "Warning: invalid number for key '" << key << "': " << e.what() << std::endl;
                return 1;
            }
        }
    }
    return 0;
}

