add_executable(run_IDM run_IDM.cpp SelfEnergy.cpp)

target_include_directories(run_IDM PRIVATE
    /home/dayun/thermal_funcs/src
)

target_link_directories(run_IDM PRIVATE
    /home/dayun/thermal_funcs/lib
)

target_link_libraries(run_IDM PRIVATE
    thermal_funcs
    gsl
    gslcblas
)

set_target_properties(run_IDM PROPERTIES
    BUILD_RPATH /home/dayun/thermal_funcs/lib
    INSTALL_RPATH /home/dayun/thermal_funcs/lib
)
