//
// Created by Haoda Fu on 8/5/18.
//

#include "FillUpA_Case1.h"

void FillUpA_Case1::fillA(DataTable & dataM) {
    size_t sampleSize = dataM.getSampleSize();
    auto & A = dataM.ref_VarA();
    double tempMean=0;
    for(size_t i=0;i<sampleSize;++i){
        tempMean = -2.5+3*dataM.getVarX_Cont().at(0).at(i)+dataM.getVarX_Ord().at(0).at(i)+dataM.getVarX_Nom().at(0).at(i);
        tempMean = exp(tempMean);
        tempMean = tempMean/(1+tempMean);

        std::binomial_distribution<int>::param_type p{1,tempMean};
        rbinom.param(p);
        A.at(i) = rbinom(generator)+1;

    }
}
