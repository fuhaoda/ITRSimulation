//
// Created by Haoda Fu on 7/16/18.
//

#include "FillUpX.h"
using namespace std;

void FillUpX::fillX(DataTable & dataM){

    if(dataM.getN_varX_Cont()>0){
        fill_VarX_Cont(dataM);
    }

    if(dataM.getN_varX_Ord()>0){
        fill_VarX_Ord(dataM);
    }

    if(dataM.getN_varX_Nom()>0){
        fill_VarX_Nom(dataM);
    }
}



void FillUpX::fill_VarX_Cont(DataTable & dataM) {
    //direct operate on the original data reference
    auto & varX_Cont = dataM.ref_VarX_Cont();

    for(auto & item:varX_Cont){
        for(auto & value:item){
            value = runifDouble(generator);
        }
    }
}

void FillUpX::fill_VarX_Nom(DataTable & dataM) {
    auto & varX_Nom = dataM.ref_VarX_Nom();
    for(auto & item:varX_Nom){
        for(auto & value:item){
            value = runifInt(generator);
        }
    }
}

void FillUpX::fill_VarX_Ord(DataTable & dataM) {
    auto & varX_Ord = dataM.ref_VarX_Ord();
    for(auto & item:varX_Ord){
        for(auto & value:item){
            value = runifInt(generator);
        }
    }
}
