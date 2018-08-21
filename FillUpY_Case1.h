//
// Created by Haoda Fu on 8/5/18.
//

#ifndef ITR2_FILLUPY_CASE1_H
#define ITR2_FILLUPY_CASE1_H

#include "FillUpY.h"

class FillUpY_Case1: public FillUpY {
public:
    explicit FillUpY_Case1(std::default_random_engine & generator):FillUpY(generator){};
    ~FillUpY_Case1() override = default;
private:
    std::vector<double> valueY(size_t row, DataTable &, int trt) override;
};


#endif //ITR2_FILLUPY_CASE1_H
