//
// Created by Haoda Fu on 7/22/18.
/*
 * This is the base class to generate action. It requires a random engine as input.
 */

#ifndef ITR2_FILLUPA_H
#define ITR2_FILLUPA_H

#include "DataTable.h"
#include <random>

#include<iostream>
#include <vector>



class FillUpA {
public:
    explicit FillUpA(std::default_random_engine & generator):generator{generator}{};
    virtual ~FillUpA()= default;
    virtual void fillA(DataTable &);
protected:

    std::default_random_engine & generator;
    std::normal_distribution<> rnorm{0,1};
    std::uniform_real_distribution<> runifDouble{0,1};
    std::uniform_int_distribution<> runifInt{0,1};
    std::binomial_distribution<int> rbinom{1,0.5};
};


#endif //ITR2_FILLUPA_H
