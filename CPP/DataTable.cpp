//
// Created by Haoda Fu on 5/28/18.
//

#include "DataTable.h"
using namespace std;

const vector<vector<double>> &DataTable::getVarY() const {
    return varY;
}

void DataTable::setVarY(const vector<vector<double>> &varY) {
    DataTable::varY = varY;
}

const vector<int> &DataTable::getVarA() const {
    return varA;
}

void DataTable::setVarA(const vector<int> &varA) {
    DataTable::varA = varA;
}

const vector<vector<double>> &DataTable::getVarX_Cont() const {
    return varX_Cont;
}

void DataTable::setVarX_Cont(const vector<vector<double>> &varX_Cont) {
    DataTable::varX_Cont = varX_Cont;
}

const vector<vector<int>> &DataTable::getVarX_Ord() const {
    return varX_Ord;
}

void DataTable::setVarX_Ord(const vector<vector<int>> &varX_Ord) {
    DataTable::varX_Ord = varX_Ord;
}

const vector<vector<int>> &DataTable::getVarX_Nom() const {
    return varX_Nom;
}

void DataTable::setVarX_Nom(const vector<vector<int>> &varX_Nom) {
    DataTable::varX_Nom = varX_Nom;
}

const vector<unsigned int> &DataTable::getSubjectID() const {
    return subjectID;
}

void DataTable::setSubjectID(const vector<unsigned int> &subjectID) {
    DataTable::subjectID = subjectID;
}

DataTable::DataTable(size_t sampleSize, NumOfVars inNumOfVars):
        varA{vector<int>(sampleSize)},subjectID{vector<unsigned int>(sampleSize)},
        varX_Cont{vector<vector<double>>(inNumOfVars.nCont,vector<double>(sampleSize))},
        varY{vector<vector<double>>(inNumOfVars.nVarY,vector<double>(sampleSize))},
        varX_Ord{vector<vector<int>>(inNumOfVars.nOrd,vector<int>(sampleSize))},
        varX_Nom{vector<vector<int>>(inNumOfVars.nNom,vector<int>(sampleSize))}, sampleSize{sampleSize},
        n_varX_Cont{inNumOfVars.nCont},n_varX_Ord{inNumOfVars.nOrd},n_varX_Nom{inNumOfVars.nNom},n_varY{inNumOfVars.nVarY}{

}

size_t DataTable::getSampleSize() const {
    return sampleSize;
}

size_t DataTable::getN_varX_Cont() const {
    return n_varX_Cont;
}

size_t DataTable::getN_varX_Ord() const {
    return n_varX_Ord;
}

size_t DataTable::getN_varX_Nom() const {
    return n_varX_Nom;
}

vector<vector<double>> &DataTable::ref_VarX_Cont() {
    return varX_Cont;
}

vector<vector<int>> &DataTable::ref_VarX_Ord() {
    return varX_Ord;
}

vector<vector<int>> &DataTable::ref_VarX_Nom() {
    return varX_Nom;
}

vector<vector<double>> &DataTable::ref_VarY() {
    return varY;
}

vector<int> &DataTable::ref_VarA() {
    return varA;
}







