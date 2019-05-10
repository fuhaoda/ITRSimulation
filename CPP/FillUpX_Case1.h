//
// Created by Haoda Fu on 7/17/18.
//

#ifndef ITR2_CASE1FILLUPX_H
#define ITR2_CASE1FILLUPX_H

#include "FillUpX.h"

class FillUpX_Case1:public FillUpX{

public:
    explicit FillUpX_Case1(std::default_random_engine & generator):FillUpX(generator){};
    ~FillUpX_Case1() override = default;
private:
    void fill_VarX_Cont(DataTable &) override;
    void fill_VarX_Ord(DataTable &) override;
    void fill_VarX_Nom(DataTable &) override;
};




#endif //ITR2_CASE1FILLUPX_H
