#ifndef POTENTIAL_IDM_HPP_INCLUDED
#define POTENTIAL_IDM_HPP_INCLUDED

/*
   IDM
*/

#include <fstream>
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "potential.hpp"
#include "SelfEnergy.hpp"
using namespace std;


namespace EffectivePotential {
    class IDM : public Potential {
    public:

        size_t get_n_scalars() const override { return 1; }

        bool forbidden(Eigen::VectorXd X) const override { return X[0] < -0.1; } 















        double V(Eigen::VectorXd x, double T) const override {

            return 0;
        }





    private:
        // constants
        const double v0 = 246.22; // GeV
        const double m_h = 125.10; // GeV
        const double mt = 172.76; // GeV
        const double g = 0.65175;
        const double gp = 0.35742;
        const double gstar = 110.75;
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

        double delta_lam1 = 0;
        double delta_mu1sq = 0;
        double delta_mu2sq = 0;
        double delta_lamm = 0;
        double delta_lamp = 0;
        double delta_lam3 = 0;

    };









}  // namespace EffectivePotential

#endif