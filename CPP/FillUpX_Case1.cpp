//
// Created by Haoda Fu on 7/19/18.
//

#include "FillUpX_Case1.h"
void FillUpX_Case1::fill_VarX_Cont(DataTable & dataM) {
    auto & varX_Cont = dataM.ref_VarX_Cont();

    for(auto & item:varX_Cont){
        for(auto & value:item){
            value = runifDouble(generator);
        }
    }
}

void FillUpX_Case1::fill_VarX_Ord(DataTable & dataM) {

    auto & varX_Nom = dataM.ref_VarX_Nom();
    for(auto & item:varX_Nom){
        for(auto & value:item){
            value = runifInt(generator);
        }
    }
}

void FillUpX_Case1::fill_VarX_Nom(DataTable & dataM) {
    auto & varX_Ord = dataM.ref_VarX_Ord();
    for(auto & item:varX_Ord){
        for(auto & value:item){
            value = runifInt(generator);
        }
    }
}