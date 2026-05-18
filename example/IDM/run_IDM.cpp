#include "IDM.hpp"
#include "phase_finder.hpp"
#include "transition_finder.hpp"
#define GAMMA_E 0.577215661901532

int get_inputs(double& lam2, double& lamL, double& mA, double& mH, double& mHpm, std::string& paramNumber, std::string& Resum);

int main(int argc, char *argv[]) { 
    const bool debug_mode = argc > 1 && strcmp(argv[1], "-d") == 0;
    // Set level of screen  output
    if (debug_mode) {
        LOGGER(debug);
    } else {
        LOGGER(fatal);
    }

    double v_over_T;

    double lam2, lamL, mA, mH, mHpm;
    std::string paramNumber, resum;

    // Get input parameters from file
    if (get_inputs(lam2, lamL, mA, mH, mHpm, paramNumber, resum) != 0) {
        std::cerr << "Error: Failed to get input parameters." << std::endl;
        return 1;
    }

    EffectivePotential::IDM idm;
    // Initialize the IDM parameters
    idm.init_params(lam2, lamL, mA, mH, mHpm, paramNumber);

    // Check perturbativity and vacuum stability
    if (!idm.check_perturbativity() || !idm.check_vacuum_stability()) {
        std::cout << "np.nan" << std::endl;
        return 0;
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
    pf.set_lower_bounds({0.0});
    pf.set_t_high(300.0);
    pf.set_v(246.22);
    pf.find_phases();
    //std::cout << pf;

    // create action calculator object
    PhaseTracer::ActionCalculator ac(idm);

    //createtransitionfinderobject
    PhaseTracer::TransitionFinder tf(pf, ac);

    //findtransitions
    tf.find_transitions();

    // extract transitions
    auto t = tf.get_transitions();
    //double crit_temp = t[0].TC;
    double nuc_temp = t[0].TC;
    double nuc_vev1 =  t[0].true_vacuum[0];
    double nuc_vev2 =  t[0].false_vacuum[0];

    std::cout << nuc_vev1/nuc_temp << std::endl;
    std::cout << nuc_vev2/nuc_temp << std::endl;
    

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


