//
// Created by Haoda Fu on 8/5/18.
//

#include "FillUpY_Case1.h"

std::vector<double> FillUpY_Case1::valueY(size_t row, DataTable & dataM, int trt) {
    std::vector<double> Yi(dataM.getVarY().size(),0);
    size_t yDim=dataM.getVarY().size();
    for(size_t item=0;item < yDim; ++item){
        Yi.at(item) = (double)trt+(double(trt)-1.5)*(double)(dataM.getVarX_Cont().at(1).at(row)>0.7 && dataM.getVarX_Nom().at(0).at(row)==0)+2*dataM.getVarX_Cont().at(0).at(row)+rnorm(generator);
    }
    return Yi;
}
