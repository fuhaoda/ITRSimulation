//
// Created by Haoda Fu on 7/22/18.
/*
 * This is the base class to generate responses.  It also generate counterfactual Y for each treatment.
 * Therefore, we need to generate data in order. Generate X first then A, then Y.
 */

#ifndef ITR2_FILLUPY_H
#define ITR2_FILLUPY_H

#include "DataTable.h"
#include <random>

#include<iostream>
#include <vector>
#include <set>



class FillUpY {
public:
    explicit FillUpY(std::default_random_engine & generator):generator{generator}{};
    virtual ~FillUpY()= default;
    void fillY(DataTable &);
    //this function generated couterfactual Y, the definition of this function depends on the DataTable class
    //the return type vector<vector<double>> is the the type of Y, and the int trt is algined with A.
    //trt, column, row
    std::vector<std::vector<std::vector<double>>> counterfactualYs(DataTable &);

protected:
    std::vector<std::vector<double>> yEachTrt(DataTable &, int trt);
    std::default_random_engine & generator;
    std::normal_distribution<> rnorm{0,1};
    std::uniform_real_distribution<> runifDouble{0,1};
    std::uniform_int_distribution<> runifInt{0,1};
private:

    virtual std::vector<double> valueY(size_t row, DataTable &, int trt);

};


#endif //ITR2_FILLUPY_H
