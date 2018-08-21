//
// Created by Haoda Fu on 8/5/18.
//

#ifndef ITR2_FILLUPACASE1_H
#define ITR2_FILLUPACASE1_H

#include "FillUpA.h"

class FillUpA_Case1:public FillUpA{
public:
    explicit FillUpA_Case1(std::default_random_engine & generator):FillUpA(generator){};
    ~FillUpA_Case1() override = default;
    void fillA(DataTable &) override;
};


#endif //ITR2_FILLUPACASE1_H
