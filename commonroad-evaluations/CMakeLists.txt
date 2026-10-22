cmake_minimum_required(VERSION 3.20)
project(lex_stl_planner_bindings)

set(CMAKE_CXX_STANDARD 17)

# Enable LTO and specify the number of parallel jobs
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Enables LTO
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=2") #2 parallel jobs
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=2")

# Find pybind11, Eigen, OpenMP, and YAML-CPP
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(YAML-CPP REQUIRED)

# Find Pagmo2 (required for CMA-ES planner)
find_package(Pagmo REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/config
    ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/planner
    ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/system
    ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/cost
    ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/stl
    ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/utils
    ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/bindings
    ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/profiling
    ${CMAKE_SOURCE_DIR}/external/frenetix
    ${CMAKE_SOURCE_DIR}/external/third_party/lhslib/src/lhslib
    ${CMAKE_SOURCE_DIR}/external/third_party/lhslib/src/bclib
    ${EIGEN3_INCLUDE_DIR}
)

# Add source files dynamically
file(GLOB_RECURSE SOURCES
    ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/*.cpp
)

# Set output directory for .so file
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lex_stl_planner/lib)

# Create the Python module
pybind11_add_module(lex_stl_planner_bindings ${SOURCES})

# Link against Eigen, OpenMP, YAML-CPP, and Pagmo2
target_link_libraries(lex_stl_planner_bindings PRIVATE Eigen3::Eigen)
target_link_libraries(lex_stl_planner_bindings PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(lex_stl_planner_bindings PRIVATE yaml-cpp)
target_link_libraries(lex_stl_planner_bindings PRIVATE Pagmo::pagmo)

# Lightweight regression target for the header-only FOM-IGO xNES updater.
option(BUILD_FOM_IGO_TESTS "Build FOM-IGO unit tests" ON)
if(BUILD_FOM_IGO_TESTS)
    enable_testing()
    add_executable(fom_igo_block_updater_test
        ${CMAKE_SOURCE_DIR}/lex_stl_planner/tests/fom_igo_block_updater_test.cpp
    )
    target_include_directories(fom_igo_block_updater_test PRIVATE
        ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/planner
        ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(fom_igo_block_updater_test PRIVATE Eigen3::Eigen)
    add_test(NAME fom_igo_block_updater_test COMMAND fom_igo_block_updater_test)

    # Exercise YAML parsing of the canonical planner.igo section and ensure
    # legacy MPPI sampling settings do not leak into the active IGO backend.
    add_executable(fom_igo_config_test
        ${CMAKE_SOURCE_DIR}/lex_stl_planner/tests/fom_igo_config_test.cpp
    )
    target_include_directories(fom_igo_config_test PRIVATE
        ${CMAKE_SOURCE_DIR}/lex_stl_planner/cpp/config
        ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(fom_igo_config_test PRIVATE Eigen3::Eigen yaml-cpp)
    add_test(NAME fom_igo_config_test COMMAND fom_igo_config_test)
    set_tests_properties(fom_igo_config_test PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# Add external dependencies
if(EXISTS ${CMAKE_SOURCE_DIR}/external/third_party/CMakeLists.txt)
    add_subdirectory(external/third_party)
    # Link LHS library if available
    if(TARGET lhslib)
        target_link_libraries(lex_stl_planner_bindings PRIVATE lhslib)
        message(STATUS "LHS library linked")
    endif()
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/external/clcs/CMakeLists.txt)
    add_subdirectory(external/clcs)
    if(TARGET crccosy)
        target_link_libraries(lex_stl_planner_bindings PRIVATE crccosy)
        message(STATUS "CLCS library linked")
    endif()
endif()

# Add frenetix library
if(EXISTS ${CMAKE_SOURCE_DIR}/external/frenetix/CMakeLists.txt)
    add_subdirectory(external/frenetix)
    if(TARGET frenetix)
        target_link_libraries(lex_stl_planner_bindings PRIVATE frenetix)
        message(STATUS "Frenetix library linked")
        # Ensure frenetix is built after crccosy if it exists
        if(TARGET crccosy)
            add_dependencies(frenetix crccosy)
        endif()
    endif()
endif()