//
// Created by Haoda Fu on 7/16/18.
/*
 * This is the base class to generate X. It requires a random engine as input.
 */

#ifndef ITR2_FILLUPX_H
#define ITR2_FILLUPX_H

#include "DataTable.h"
#include <random>

#include<iostream>
#include <vector>




class FillUpX{
public:
    //We use single random engine for simulation scenario. So we require a random engine reference input
    explicit FillUpX(std::default_random_engine & generator):generator{generator}{};
    virtual ~FillUpX()= default;
    void fillX(DataTable &);
protected:

    std::default_random_engine & generator;
    std::normal_distribution<> rnorm{0,1};
    std::uniform_real_distribution<> runifDouble{0,1};
    std::uniform_int_distribution<> runifInt{0,1};

private:
    virtual void fill_VarX_Cont(DataTable &);
    virtual void fill_VarX_Ord(DataTable &);
    virtual void fill_VarX_Nom(DataTable &);
};






#endif //ITR2_FILLUPX_H
