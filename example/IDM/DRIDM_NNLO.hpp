#include <vector>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include "pow.hpp"



class DRIDM_NNLO {
    public:
        std::complex<double> V2( Eigen::VectorXd X, double T, std::vector<double> par ) {
            std::complex<double> gpsq(par[0], 0);
            std::complex<double> gsq(par[1], 0);
            std::complex<double> gssq(par[2], 0);
            std::complex<double> lam1(par[3], 0);
            std::complex<double> mu1sq(par[4], 0);

            std::complex<double> phi(X[0]/sqrt(T + 1e-15),0);

            std::complex<double> RG_scale_3DUS = gsq * T;

            std::complex<double> veffNNLO_part1 = // === 第一部分：基础项 ===
                pow(gsq, 3) * pow(phi, 2) / (48. * (gsq + gpsq) * square(M_PI)) + 
                square(gsq) * gpsq * pow(phi, 2) / (48. * (gsq + gpsq) * square(M_PI)) + 
                square(gsq) * sqrt(gsq * pow(phi, 2)) * sqrt((gsq + gpsq) * pow(phi, 2)) / (24. * (gsq + gpsq) * square(M_PI)) + 
                3. * gsq * sqrt(gsq * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (64. * square(M_PI)) + 
                // 两项相同，合并
                2. * square(gsq - gpsq) * sqrt((gsq + gpsq) * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * (gsq + gpsq) * square(M_PI)) + 
                (gsq + gpsq) * sqrt((gsq + gpsq) * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) + 
                15. * lam1 * (mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) + 
                gsq * sqrt(gsq * pow(phi, 2)) * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) / (64. * square(M_PI)) + 
                (gsq + gpsq) * sqrt((gsq + gpsq) * pow(phi, 2)) * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) + 
                3. * lam1 * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) / (64. * square(M_PI)) + 
                3. * lam1 * (mu1sq + 1.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) + 
                
                // === 对数项 ===
                pow(gsq, 3) * pow(phi, 2) * (0.5 + log(2. * RG_scale_3DUS / sqrt(gsq * pow(phi, 2)))) / (128. * (gsq + gpsq) * square(M_PI)) + 
                square(gsq) * gpsq * pow(phi, 2) * (0.5 + log(2. * RG_scale_3DUS / sqrt(gsq * pow(phi, 2)))) / (128. * (gsq + gpsq) * square(M_PI)) + 
                4. * gpsq * (-89. * pow(gsq, 3) * pow(phi, 6) / (2048. * square(M_PI)) + 3. * pow(gsq, 3) * pow(phi, 6) * (0.5 + log(RG_scale_3DUS / sqrt(gsq * pow(phi, 2)))) / (64. * square(M_PI)) + 
                3. * pow(gsq, 3) * pow(phi, 6) * (0.5 + log(2. * RG_scale_3DUS / sqrt(gsq * pow(phi, 2)))) / (256. * square(M_PI))) / (3. * gsq * (gsq + gpsq) * pow(phi, 4)) + 
                square(gsq) * pow(phi, 2) * (0.5 + log(2. * RG_scale_3DUS / sqrt((gsq + gpsq) * pow(phi, 2)))) / (256. * square(M_PI)) ;

            std::complex<double> veffNNLO_part2 = // === 第一项：g²gp² 相关项 ===
                gsq * gpsq * (
                    - (2. * mu1sq + lam1 * pow(phi, 2)) / (32. * square(M_PI)) - 
                    2. * (
                        - (mu1sq + 0.5 * lam1 * pow(phi, 2)) / (16. * square(M_PI)) - 
                        (2. * mu1sq + lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / (2. * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI))
                    )) / (2. * (gsq + gpsq)) - 
                
                // === 第二项：λ₁²Φ² 对数项 ===
                3. * square(lam1) * pow(phi, 2) * (0.5 + log(RG_scale_3DUS / (3. * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2))))) / (64. * square(M_PI)) + 
                
                // === 第三项：复杂大项 ===
                1. / (3. * square(gsq + gpsq) * pow(phi, 6)) * 8. * (
                    40. / 3. * (
                        pow(gsq, 3) * (gsq + gpsq) * pow(phi, 8) / (8192. * square(M_PI)) + 
                        square(gsq) * square(gsq + gpsq) * pow(phi, 8) / (16384. * square(M_PI))
                    ) + 
                    1. / 3. * (
                        // 子项1
                        1. / (128. * square(M_PI)) * gsq * pow(phi, 2) * sqrt(gsq * pow(phi, 2)) * sqrt((gsq + gpsq) * pow(phi, 2)) * (
                            3. * pow(gsq, 2) * pow(phi, 4) / 16. - 
                            11. / 16. * gsq * (gsq + gpsq) * pow(phi, 4) - 
                            15. / 16. * square(gsq + gpsq) * pow(phi, 4)
                        ) + 
                        // 子项2
                        (gsq * (gsq + gpsq) * pow(phi, 4) * (
                            -7. / 2. * pow(gsq, 2) * pow(phi, 4) + 
                            15. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + 
                            3. / 16. * square(gsq + gpsq) * pow(phi, 4)
                        )) / (256. * square(M_PI)) + 
                        // 子项3
                        1. / 128. * pow(gsq, 4) * pow(phi, 8) * (
                            1. / (32. * square(M_PI)) - 
                            3. * (0.5 + log(2. * RG_scale_3DUS / sqrt(gsq * pow(phi, 2)))) / (16. * square(M_PI))
                        ) + 
                        // 子项4
                        1. / 256. * pow(gsq + gpsq, 4) * pow(phi, 8) * (
                            1. / (32. * square(M_PI)) - 
                            3. * (0.5 + log(2. * RG_scale_3DUS / sqrt((gsq + gpsq) * pow(phi, 2)))) / (16. * square(M_PI))
                        ) + 
                        // 子项5
                        2. * square(-0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2)) * (
                            1. / (64. * square(M_PI)) * (
                                -1.5 * gsq * (gsq + gpsq) * pow(phi, 4) - 
                                2. * (pow(gsq, 2) * pow(phi, 4) / 16. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.)
                            ) + 
                            1. / (16. * square(M_PI)) * 3. * (
                                pow(gsq, 2) * pow(phi, 4) / 16. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.
                            ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + 0.5 * sqrt((gsq + gpsq) * pow(phi, 2)))))) + 
                        // 子项6
                        (-0.25 * gsq * (gsq + gpsq) * pow(phi, 4) + 1. / 16. * square(gsq + gpsq) * pow(phi, 4)) * (
                            -1. / (64. * square(M_PI)) * (
                                3. * (-0.5 * pow(gsq, 2) * pow(phi, 4) - gsq * (gsq + gpsq) * pow(phi, 4)) - 
                                2. * (pow(gsq, 2) * pow(phi, 4) / 2. + 3. / 4. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.)
                            ) - 
                            1. / (16. * square(M_PI)) * 3. * (
                                pow(gsq, 2) * pow(phi, 4) / 2. + 3. / 4. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.
                            ) * (0.5 + log(RG_scale_3DUS / (sqrt(gsq * pow(phi, 2)) + 0.5 * sqrt((gsq + gpsq) * pow(phi, 2)))))
                )));

            std::complex<double> veffNNLO_part3 = 
                1. / (3. * square(gsq + gpsq) * pow(phi, 6)) * 8. * (
                    40. / 3. * (
                        pow(gsq, 3) * (gsq + gpsq) * pow(phi, 8) / (8192. * square(M_PI)) + 
                        square(gsq) * square(gsq + gpsq) * pow(phi, 8) / (16384. * square(M_PI))
                    ) + 
                    1. / 3. * (
                        // 子项1
                        1. / (128. * square(M_PI)) * gsq * pow(phi, 2) * sqrt(gsq * pow(phi, 2)) * sqrt((gsq + gpsq) * pow(phi, 2)) * (
                            3. * pow(gsq, 2) * pow(phi, 4) / 16. - 
                            11. / 16. * gsq * (gsq + gpsq) * pow(phi, 4) - 
                            15. / 16. * square(gsq + gpsq) * pow(phi, 4)
                        ) + 
                        // 子项2
                        (gsq * (gsq + gpsq) * pow(phi, 4) * (
                            -7. / 2. * pow(gsq, 2) * pow(phi, 4) + 
                            15. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + 
                            3. / 16. * square(gsq + gpsq) * pow(phi, 4)
                        )) / (256. * square(M_PI)) + 
                        // 子项3
                        1. / 128. * pow(gsq, 4) * pow(phi, 8) * (
                            1. / (32. * square(M_PI)) - 
                            3. * (0.5 + log(2. * RG_scale_3DUS / sqrt(gsq * pow(phi, 2)))) / (16. * square(M_PI))
                        ) + 
                        // 子项4
                        1. / 256. * pow(gsq + gpsq, 4) * pow(phi, 8) * (
                            1. / (32. * square(M_PI)) - 
                            3. * (0.5 + log(2. * RG_scale_3DUS / sqrt((gsq + gpsq) * pow(phi, 2)))) / (16. * square(M_PI))
                        ) + 
                        // 子项5
                        2. * square(0.25 * gsq * pow(phi, 2) - 0.25 * (gsq + gpsq) * pow(phi, 2)) * (
                            1. / (64. * square(M_PI)) * (
                                -1.5 * gsq * (gsq + gpsq) * pow(phi, 4) - 
                                2. * (pow(gsq, 2) * pow(phi, 4) / 16. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.)
                            ) + 
                            1. / (16. * square(M_PI)) * 3. * (
                                pow(gsq, 2) * pow(phi, 4) / 16. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.
                            ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + 0.5 * sqrt((gsq + gpsq) * pow(phi, 2)))))) + 
                        // 子项6
                        (pow(gsq, 2) * pow(phi, 4) / 16. + square(0.25 * gsq * pow(phi, 2) - 0.25 * (gsq + gpsq) * pow(phi, 2)) - 0.5 * gsq * pow(phi, 2) * (0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2))) * (
                            -1. / (64. * square(M_PI)) * (
                                3. * (-0.5 * gsq * (gsq + gpsq) * pow(phi, 4) - 2. * gsq * pow(phi, 2) * (0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2))) - 
                                2. * (pow(gsq, 2) * pow(phi, 4) / 8. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16. + 1.5 * gsq * pow(phi, 2) * (0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2)))
                            ) - 
                            1. / (16. * square(M_PI)) * 3. * (
                                pow(gsq, 2) * pow(phi, 4) / 8. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16. + 1.5 * gsq * pow(phi, 2) * (0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2))
                            ) * (0.5 + log(RG_scale_3DUS / (sqrt(gsq * pow(phi, 2)) + 0.5 * sqrt((gsq + gpsq) * pow(phi, 2)))))
                )));

            std::complex<double> veffNNLO_part4 = 
                1. / (3. * square(gsq + gpsq) * pow(phi, 6)) * 8. * (
                    40. / 3. * (
                        pow(gsq, 3) * (gsq + gpsq) * pow(phi, 8) / (8192. * square(M_PI)) + 
                        square(gsq) * square(gsq + gpsq) * pow(phi, 8) / (16384. * square(M_PI))
                    ) + 
                    1. / 3. * (
                        // 子项1
                        1. / (128. * square(M_PI)) * gsq * pow(phi, 2) * sqrt(gsq * pow(phi, 2)) * sqrt((gsq + gpsq) * pow(phi, 2)) * (
                            3. * pow(gsq, 2) * pow(phi, 4) / 16. - 
                            11. / 16. * gsq * (gsq + gpsq) * pow(phi, 4) - 
                            15. / 16. * square(gsq + gpsq) * pow(phi, 4)
                        ) + 
                        // 子项2
                        (gsq * (gsq + gpsq) * pow(phi, 4) * (
                            -7. / 2. * pow(gsq, 2) * pow(phi, 4) + 
                            15. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + 
                            3. / 16. * square(gsq + gpsq) * pow(phi, 4)
                        )) / (256. * square(M_PI)) + 
                        // 子项3
                        1. / 128. * pow(gsq, 4) * pow(phi, 8) * (
                            1. / (32. * square(M_PI)) - 
                            3. * (0.5 + log(2. * RG_scale_3DUS / sqrt(gsq * pow(phi, 2)))) / (16. * square(M_PI))
                        ) + 
                        // 子项4
                        1. / 256. * pow(gsq + gpsq, 4) * pow(phi, 8) * (
                            1. / (32. * square(M_PI)) - 
                            3. * (0.5 + log(2. * RG_scale_3DUS / sqrt((gsq + gpsq) * pow(phi, 2)))) / (16. * square(M_PI))
                        ) + 
                        // 子项5: ((g²Φ²)/4 - 1/4(g²+gp²)Φ²)^2 * (...)
                        square(0.25 * gsq * pow(phi, 2) - 0.25 * (gsq + gpsq) * pow(phi, 2)) * (
                            1. / (64. * square(M_PI)) * (
                                -1.5 * gsq * (gsq + gpsq) * pow(phi, 4) - 
                                2. * (pow(gsq, 2) * pow(phi, 4) / 16. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.)
                            ) + 
                            1. / (16. * square(M_PI)) * 3. * (
                                pow(gsq, 2) * pow(phi, 4) / 16. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.
                            ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + 0.5 * sqrt((gsq + gpsq) * pow(phi, 2)))))) + 
                        // 子项6: (-(1/4)g²Φ² + 1/4(g²+gp²)Φ²)^2 * (...)
                        square(-0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2)) * (
                            1. / (64. * square(M_PI)) * (
                                -1.5 * gsq * (gsq + gpsq) * pow(phi, 4) - 
                                2. * (pow(gsq, 2) * pow(phi, 4) / 16. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.)
                            ) + 
                            1. / (16. * square(M_PI)) * 3. * (
                                pow(gsq, 2) * pow(phi, 4) / 16. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16.
                            ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + 0.5 * sqrt((gsq + gpsq) * pow(phi, 2)))))) + 
                        // 子项7: (g⁴Φ⁴/16 + (-(1/4)g²Φ² + 1/4(g²+gp²)Φ²)^2 - 1/2g²Φ²((g²Φ²)/4 + 1/4(g²+gp²)Φ²)) * (...)
                        (pow(gsq, 2) * pow(phi, 4) / 16. + square(-0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2)) - 0.5 * gsq * pow(phi, 2) * (0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2))) * (
                            -1. / (64. * square(M_PI)) * (
                                3. * (-0.5 * gsq * (gsq + gpsq) * pow(phi, 4) - 2. * gsq * pow(phi, 2) * (0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2))) - 
                                2. * (pow(gsq, 2) * pow(phi, 4) / 8. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16. + 1.5 * gsq * pow(phi, 2) * (0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2)))
                            ) - 
                            1. / (16. * square(M_PI)) * 3. * (
                                pow(gsq, 2) * pow(phi, 4) / 8. + 3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + square(gsq + gpsq) * pow(phi, 4) / 16. + 1.5 * gsq * pow(phi, 2) * (0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2))
                            ) * (0.5 + log(RG_scale_3DUS / (sqrt(gsq * pow(phi, 2)) + 0.5 * sqrt((gsq + gpsq) * pow(phi, 2)))))
                )));

            std::complex<double> veffNNLO_part5 = 
                // === 第一部分：1/(4(g²+gp²)) * g² * gp² * (...) ===
                1. / (4. * (gsq + gpsq)) * gsq * gpsq * (
                    3. * gsq * pow(phi, 2) / (128. * square(M_PI)) + 
                    2. * (
                        sqrt(gsq * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (32. * square(M_PI)) - 
                        (mu1sq + 0.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)))) / (16. * square(M_PI)) + 
                        (mu1sq - 0.75 * gsq * pow(phi, 2) + 0.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI))
                    )
                ) + 
                
                // === 第二部分：1/(2(g²+gp²)²Φ²) * gp⁴ * (...) ===
                1. / (2. * square(gsq + gpsq) * pow(phi, 2)) * pow(gpsq, 2) * (
                    gsq * (gsq + gpsq) * pow(phi, 4) / (128. * square(M_PI)) - 
                    sqrt(gsq * pow(phi, 2)) * sqrt((gsq + gpsq) * pow(phi, 2)) * (-mu1sq + 0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2) - 0.5 * lam1 * pow(phi, 2)) / (64. * square(M_PI)) + 
                    (gsq + gpsq) * pow(phi, 2) * sqrt(gsq * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) + 
                    gsq * pow(phi, 2) * sqrt((gsq + gpsq) * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) - 
                    square(mu1sq + 0.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)))) / (16. * square(M_PI)) + 
                    square(-mu1sq + 0.25 * gsq * pow(phi, 2) - 0.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI)) + 
                    square(-mu1sq + 0.25 * (gsq + gpsq) * pow(phi, 2) - 0.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt((gsq + gpsq) * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI)) - 
                    1. / (16. * square(M_PI)) * (
                        pow(gsq, 2) * pow(phi, 4) / 16. + 
                        3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + 
                        square(-mu1sq + 0.25 * (gsq + gpsq) * pow(phi, 2) - 0.5 * lam1 * pow(phi, 2)) - 
                        0.5 * gsq * pow(phi, 2) * (mu1sq + 0.5 * lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + 0.5 * sqrt((gsq + gpsq) * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)))))
                );

            std::complex<double> veffNNLO_part6 = 
                // === 第一部分：1/(2(g²+gp²)²Φ²) * gp⁴ * (...) ===
                1. / (2. * square(gsq + gpsq) * pow(phi, 2)) * pow(gpsq, 2) * (
                    gsq * (gsq + gpsq) * pow(phi, 4) / (128. * square(M_PI)) - 
                    sqrt(gsq * pow(phi, 2)) * sqrt((gsq + gpsq) * pow(phi, 2)) * (-mu1sq + 0.25 * gsq * pow(phi, 2) + 0.25 * (gsq + gpsq) * pow(phi, 2) - 0.5 * lam1 * pow(phi, 2)) / (64. * square(M_PI)) + 
                    (gsq + gpsq) * pow(phi, 2) * sqrt(gsq * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) + 
                    gsq * pow(phi, 2) * sqrt((gsq + gpsq) * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) - 
                    square(mu1sq + 0.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)))) / (16. * square(M_PI)) + 
                    square(-mu1sq + 0.25 * gsq * pow(phi, 2) - 0.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI)) + 
                    square(-mu1sq + 0.25 * (gsq + gpsq) * pow(phi, 2) - 0.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt((gsq + gpsq) * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI)) - 
                    1. / (16. * square(M_PI)) * (
                        3. / 8. * gsq * (gsq + gpsq) * pow(phi, 4) + 
                        1. / 16. * square(gsq + gpsq) * pow(phi, 4) + 
                        square(-mu1sq + 0.25 * gsq * pow(phi, 2) - 0.5 * lam1 * pow(phi, 2)) - 
                        0.5 * (gsq + gpsq) * pow(phi, 2) * (mu1sq + 0.5 * lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + 0.5 * sqrt((gsq + gpsq) * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)))))
                ) + 
                
                // === 第二部分：1/Φ² * (...) ===
                1. / pow(phi, 2) * (
                    - gsq * pow(phi, 2) * sqrt(gsq * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) + 
                    sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) * (
                        - gsq * pow(phi, 2) * sqrt(gsq * pow(phi, 2)) / (32. * M_PI) + 
                        gsq * pow(phi, 2) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (16. * M_PI)
                    ) / (4. * M_PI) - 
                    (pow(gsq, 2) * pow(phi, 4) / 16. - 0.5 * gsq * pow(phi, 2) * (2. * mu1sq + lam1 * pow(phi, 2))) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + 2. * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI))
                );

            std::complex<double> veffNNLO_part7 = 
                // === 第一部分：1/(4(g²+gp²)²Φ²) * (g²-gp²)² * (...) ===
                1. / (4. * square(gsq + gpsq) * pow(phi, 2)) * square(gsq - gpsq) * (
                    - (gsq + gpsq) * pow(phi, 2) * sqrt((gsq + gpsq) * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) + 
                    sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) * (
                        - (gsq + gpsq) * pow(phi, 2) * sqrt((gsq + gpsq) * pow(phi, 2)) / (32. * M_PI) + 
                        (gsq + gpsq) * pow(phi, 2) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (16. * M_PI)
                    ) / (4. * M_PI) - 
                    1. / (16. * square(M_PI)) * (
                        1. / 16. * square(gsq + gpsq) * pow(phi, 4) - 
                        0.5 * (gsq + gpsq) * pow(phi, 2) * (2. * mu1sq + lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt((gsq + gpsq) * pow(phi, 2)) + 2. * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)))))
                ) + 
                
                // === 第二部分：1/(4(g²+gp²)²Φ²) * (-g²+gp²)² * (...)（与第一部分相同）===
                1. / (4. * square(gsq + gpsq) * pow(phi, 2)) * square(-gsq + gpsq) * (
                    - (gsq + gpsq) * pow(phi, 2) * sqrt((gsq + gpsq) * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (128. * square(M_PI)) + 
                    sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) * (
                        - (gsq + gpsq) * pow(phi, 2) * sqrt((gsq + gpsq) * pow(phi, 2)) / (32. * M_PI) + 
                        (gsq + gpsq) * pow(phi, 2) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (16. * M_PI)
                    ) / (4. * M_PI) - 
                    1. / (16. * square(M_PI)) * (
                        1. / 16. * square(gsq + gpsq) * pow(phi, 4) - 
                        0.5 * (gsq + gpsq) * pow(phi, 2) * (2. * mu1sq + lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt((gsq + gpsq) * pow(phi, 2)) + 2. * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)))))
                ) + 
                
                // === 第三部分：1/(2Φ²) * (...) ===
                1. / (2. * pow(phi, 2)) * (
                    pow(gsq, 2) * pow(phi, 4) / (128. * square(M_PI)) - 
                    gsq * pow(phi, 2) * (-mu1sq + 0.5 * gsq * pow(phi, 2) - 1.5 * lam1 * pow(phi, 2)) / (64. * square(M_PI)) + 
                    gsq * pow(phi, 2) * sqrt(gsq * pow(phi, 2)) * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) / (64. * square(M_PI)) - 
                    square(mu1sq + 1.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)))) / (16. * square(M_PI)) + 
                    square(-mu1sq + 0.25 * gsq * pow(phi, 2) - 1.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2))))) / (8. * square(M_PI)) - 
                    1. / (16. * square(M_PI)) * (
                        7. * pow(gsq, 2) * pow(phi, 4) / 16. + 
                        square(-mu1sq + 0.25 * gsq * pow(phi, 2) - 1.5 * lam1 * pow(phi, 2)) - 
                        0.5 * gsq * pow(phi, 2) * (mu1sq + 1.5 * lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (sqrt(gsq * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)))))
                );

            std::complex<double> veffNNLO_part8 = 
                // === 第一部分：1/(4Φ²) * (...) ===
                1. / (4. * pow(phi, 2)) * (
                    square(gsq + gpsq) * pow(phi, 4) / (128. * square(M_PI)) - 
                    (gsq + gpsq) * pow(phi, 2) * (-mu1sq + 0.5 * (gsq + gpsq) * pow(phi, 2) - 1.5 * lam1 * pow(phi, 2)) / (64. * square(M_PI)) + 
                    (gsq + gpsq) * pow(phi, 2) * sqrt((gsq + gpsq) * pow(phi, 2)) * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) / (64. * square(M_PI)) - 
                    square(mu1sq + 1.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)))) / (16. * square(M_PI)) + 
                    1. / (8. * square(M_PI)) * square(-mu1sq + 0.25 * (gsq + gpsq) * pow(phi, 2) - 1.5 * lam1 * pow(phi, 2)) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt((gsq + gpsq) * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2))))) - 
                    1. / (16. * square(M_PI)) * (
                        7. / 16. * square(gsq + gpsq) * pow(phi, 4) + 
                        square(-mu1sq + 0.25 * (gsq + gpsq) * pow(phi, 2) - 1.5 * lam1 * pow(phi, 2)) - 
                        0.5 * (gsq + gpsq) * pow(phi, 2) * (mu1sq + 1.5 * lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (sqrt((gsq + gpsq) * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)))))
                ) + 
                
                // === 第二部分：1/(2Φ²) * (...) ===
                1. / (2. * pow(phi, 2)) * (
                    - sqrt(gsq * pow(phi, 2)) * (0.25 * gsq * pow(phi, 2) - lam1 * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (32. * square(M_PI)) + 
                    sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) * (
                        gsq * pow(phi, 2) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (16. * M_PI) - 
                        sqrt(gsq * pow(phi, 2)) * (0.25 * gsq * pow(phi, 2) + lam1 * pow(phi, 2)) / (8. * M_PI)
                    ) / (4. * M_PI) + 
                    square(lam1) * pow(phi, 4) * (0.5 + log(RG_scale_3DUS / (sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI)) - 
                    1. / (16. * square(M_PI)) * (
                        pow(gsq, 2) * pow(phi, 4) / 16. + 
                        square(lam1) * pow(phi, 4) - 
                        0.5 * gsq * pow(phi, 2) * (2. * mu1sq + 2. * lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)))))
                );

            std::complex<double> veffNNLO_part9 = 
                // === 第一部分：1/(2Φ²) * (...) ===
                1. / (2. * pow(phi, 2)) * (
                    - sqrt(gsq * pow(phi, 2)) * (0.25 * gsq * pow(phi, 2) + lam1 * pow(phi, 2)) * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) / (32. * square(M_PI)) + 
                    sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) * (
                        - sqrt(gsq * pow(phi, 2)) * (0.25 * gsq * pow(phi, 2) - lam1 * pow(phi, 2)) / (8. * M_PI) + 
                        gsq * pow(phi, 2) * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) / (16. * M_PI)
                    ) / (4. * M_PI) + 
                    square(lam1) * pow(phi, 4) * (0.5 + log(RG_scale_3DUS / (sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI)) - 
                    1. / (16. * square(M_PI)) * (
                        pow(gsq, 2) * pow(phi, 4) / 16. + 
                        square(lam1) * pow(phi, 4) - 
                        0.5 * gsq * pow(phi, 2) * (2. * mu1sq + 2. * lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt(gsq * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)))))
                ) + 
                
                // === 第二部分：1/(4Φ²) * (...) ===
                1. / (4. * pow(phi, 2)) * (
                    - sqrt((gsq + gpsq) * pow(phi, 2)) * (0.25 * (gsq + gpsq) * pow(phi, 2) - lam1 * pow(phi, 2)) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (32. * square(M_PI)) + 
                    sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) * (
                        (gsq + gpsq) * pow(phi, 2) * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) / (16. * M_PI) - 
                        sqrt((gsq + gpsq) * pow(phi, 2)) * (0.25 * (gsq + gpsq) * pow(phi, 2) + lam1 * pow(phi, 2)) / (8. * M_PI)
                    ) / (4. * M_PI) + 
                    square(lam1) * pow(phi, 4) * (0.5 + log(RG_scale_3DUS / (sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI)) - 
                    1. / (16. * square(M_PI)) * (
                        1. / 16. * square(gsq + gpsq) * pow(phi, 4) + 
                        square(lam1) * pow(phi, 4) - 
                        0.5 * (gsq + gpsq) * pow(phi, 2) * (2. * mu1sq + 2. * lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt((gsq + gpsq) * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)))))
                );

            std::complex<double> veffNNLO_part10 = 
                // === 第一部分：1/(4Φ²) * (...) ===
                1. / (4. * pow(phi, 2)) * (
                    - sqrt((gsq + gpsq) * pow(phi, 2)) * (0.25 * (gsq + gpsq) * pow(phi, 2) + lam1 * pow(phi, 2)) * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) / (32. * square(M_PI)) + 
                    sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) * (
                        - sqrt((gsq + gpsq) * pow(phi, 2)) * (0.25 * (gsq + gpsq) * pow(phi, 2) - lam1 * pow(phi, 2)) / (8. * M_PI) + 
                        (gsq + gpsq) * pow(phi, 2) * sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)) / (16. * M_PI)
                    ) / (4. * M_PI) + 
                    square(lam1) * pow(phi, 4) * (0.5 + log(RG_scale_3DUS / (sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2))))) / (16. * square(M_PI)) - 
                    1. / (16. * square(M_PI)) * (
                        1. / 16. * square(gsq + gpsq) * pow(phi, 4) + 
                        square(lam1) * pow(phi, 4) - 
                        0.5 * (gsq + gpsq) * pow(phi, 2) * (2. * mu1sq + 2. * lam1 * pow(phi, 2))
                    ) * (0.5 + log(RG_scale_3DUS / (0.5 * sqrt((gsq + gpsq) * pow(phi, 2)) + sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2)))))
                ) - 
                
                // === 第二部分：- 3λ₁²Φ²(0.5 + Log[...])/(64π²) ===
                3. * square(lam1) * pow(phi, 2) * (0.5 + log(RG_scale_3DUS / (2. * sqrt(mu1sq + 0.5 * lam1 * pow(phi, 2)) + sqrt(mu1sq + 1.5 * lam1 * pow(phi, 2))))) / (64. * square(M_PI));
  
            std::complex<double> veffNNLO = veffNNLO_part1 + veffNNLO_part2 + veffNNLO_part3 + veffNNLO_part4 + veffNNLO_part5 + veffNNLO_part6 + veffNNLO_part7 + veffNNLO_part8 + veffNNLO_part9 + veffNNLO_part10;
            return veffNNLO;
        }
    
};