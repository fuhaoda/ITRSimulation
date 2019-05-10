//
// Created by Haoda Fu on 7/22/18.
//

#include "FillUpA.h"
using namespace std;

void FillUpA::fillA(DataTable & dataM) {
    uniform_int_distribution<>::param_type p{1,2};


    runifInt.param(p);
    for(auto & item:dataM.ref_VarA()){
        item=runifInt(generator);
    }
}
