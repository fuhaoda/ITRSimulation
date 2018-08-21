cmake_minimum_required(VERSION 3.12)
project(ITR2)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(ITR2
        DataTable.cpp
        DataTable.h
        FillUpA.cpp
        FillUpA.h
        FillUpA_Case1.cpp
        FillUpA_Case1.h
        FillUpX.cpp
        FillUpX.h
        FillUpX_Case1.cpp
        FillUpX_Case1.h
        FillUpY.cpp
        FillUpY.h
        FillUpY_Case1.cpp
        FillUpY_Case1.h
        main.cpp
        SimulateData.cpp
        SimulateData.h)
