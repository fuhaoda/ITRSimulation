//
// Created by Haoda Fu on 7/22/18.
//

#include "FillUpY.h"
using namespace std;

void FillUpY::fillY(DataTable & dataM) {
    auto & Y = dataM.ref_VarY();
    size_t sampleSize = dataM.getSampleSize();
    size_t yDim = dataM.getVarY().size();
    vector<double> Yobs(dataM.getVarY().size(),0);
    for(size_t row=0;row<sampleSize;++row){
        Yobs = valueY(row,dataM,dataM.getVarA().at(row));
        for(size_t item=0;item < yDim; ++item){
            Y.at(item).at(row) = Yobs.at(item);
        }
    }
}

vector<vector<double>> FillUpY::yEachTrt(DataTable & dataM, int trt) {
    auto counterfactualY = dataM.getVarY();
    size_t sampleSize = dataM.getSampleSize();
    size_t yDim = dataM.getVarY().size();
    vector<double> Yobs(dataM.getVarY().size(),0);
    for(size_t row=0;row<sampleSize;++row){
        Yobs = valueY(row,dataM,trt);
        for(size_t item=0;item < yDim; ++item){
            counterfactualY.at(item).at(row) = Yobs.at(item);
        }
    }
    return counterfactualY;
}

vector<double> FillUpY::valueY(size_t row, DataTable & dataM, int trt) {
    vector<double> Yi(dataM.getVarY().size(),0);
    size_t yDim=dataM.getVarY().size();
    for(size_t item=0;item < yDim; ++item){
        Yi.at(item) = runifDouble(generator);
    }
    return Yi;
}

vector<vector<vector<double>>> FillUpY::counterfactualYs(DataTable & dataM) {
    set<int> trts{};
    for(auto item:dataM.getVarA()){
        trts.insert(item);
    }
    vector<vector<vector<double>>> Yobs(trts.size(),dataM.getVarY());
    size_t indx=0;
    for(auto trtCode:trts){
        Yobs.at(indx++)=yEachTrt(dataM, trtCode);
    }
    return Yobs;
}


