//
// Created by Haoda Fu on 5/28/18.
//


#include "SimulateData.h"
using namespace std;

SimulateData::SimulateData(size_t sampleSizeTraining, NumOfVars inNumOfVars, unsigned randomSeed):
        seed{randomSeed}, inNumOfVars{inNumOfVars}, yDim{inNumOfVars.nVarY},
        myTrainingData{sampleSizeTraining,inNumOfVars},
        myTestingData{sampleSizeTesting,inNumOfVars}{}

void SimulateData::setXAY(unsigned choiceX, unsigned choiceA, unsigned choiceY) {
    setX(choiceX);
    setA(choiceA);
    setY(choiceY);
}

void SimulateData::setX(unsigned choiceX) {
    delete fillUpMyX;
    switch(choiceX){
        case 1:
            fillUpMyX = new FillUpX_Case1(generator);
            break;

        default:
            fillUpMyX = new FillUpX(generator);
            break;
    }

}



void SimulateData::setA(unsigned choiceA) {
    delete fillUpMyA;
    switch(choiceA){
        case 1:
            fillUpMyA = new FillUpA_Case1(generator);
            break;
        default:
            fillUpMyA = new FillUpA(generator);
            break;
    }

}

void SimulateData::setY(unsigned choiceY) {
    delete fillUpMyY;
    switch(choiceY){
        case 1:
            fillUpMyY = new FillUpY_Case1(generator);
            break;
        default:
            fillUpMyY = new FillUpY(generator);
            break;
    }
}

void SimulateData::simulateTraining() {
    fillUpMyX->fillX(myTrainingData);
    fillUpMyA->fillA(myTrainingData);
    fillUpMyY->fillY(myTrainingData);
    vector<unsigned> pid(myTrainingData.getSampleSize());
    iota(pid.begin(),pid.end(),1);
    myTrainingData.setSubjectID(pid);
}


void SimulateData::simulateTesting() {
    fillUpMyX->fillX(myTestingData);
    fillUpMyA->fillA(myTestingData);
    counterfactual_Ys = fillUpMyY->counterfactualYs(myTestingData);

    trts.clear();
    for(auto item:myTestingData.getVarA()){
        trts.insert(make_pair(item,0));
    }
    size_t trtCode=0;
    for(auto &item:trts){
        item.second=trtCode++;
    }


    size_t trtID=0;

    for(size_t row=0; row<sampleSizeTesting;++row){
        trtID=trts.at(myTestingData.getVarA().at(row));//
        for(size_t item=0;item < yDim; ++item){
            myTestingData.ref_VarY().at(item).at(row)=counterfactual_Ys.at(trtID).at(item).at(row);
        }
    }

    vector<unsigned> pid(myTestingData.getSampleSize());
    iota(pid.begin(),pid.end(),1);
    myTestingData.setSubjectID(pid);
}



void SimulateData::defaultSaveData(string fileName) {
    simulateTraining();
    savedAsTraining(fileName);
    simulateTesting();
    savedAsTesting(fileName);
}

void SimulateData::savedAsTraining(string fileName) {
    ofstream myFile;
    try{
        myFile.open(fileName+".csv");
        size_t nRows = myTrainingData.getSampleSize();
        saveFullTitle(myFile);
        for(size_t iter=0;iter<nRows;++iter){
            saveElement(myTrainingData.getSubjectID().at(iter),myFile);
            saveMatrix(myTrainingData.getVarY(),myFile,iter);
            saveElement(myTrainingData.getVarA().at(iter),myFile);
            saveMatrix(myTrainingData.getVarX_Cont(),myFile,iter);
            saveMatrix(myTrainingData.getVarX_Ord(),myFile,iter);
            saveMatrix(myTrainingData.getVarX_Nom(),myFile,iter);
            myFile.seekp(-1, std::ios_base::cur);
            myFile << endl;
        }
        myFile.close();
    } catch (const std::exception& e){
       cout << "ERROR!:" << e.what() << endl;
    }
}

void SimulateData::printData(const DataTable & dataM, bool head) {
    size_t nRows = dataM.getSampleSize();
    if(head) nRows=min(nRows, (size_t)20);
    for(size_t iter=0;iter<nRows;++iter){
        printElement(dataM.getSubjectID().at(iter));
        printMatrix(dataM.getVarY(),iter);
        printElement(dataM.getVarA().at(iter));
        printMatrix(dataM.getVarX_Cont(),iter);
        printMatrix(dataM.getVarX_Ord(),iter);
        printMatrix(dataM.getVarX_Nom(),iter);
        cout << endl;
    }
}

void SimulateData::savedAsTesting(string fileName) {
    ofstream myFile;
    try{
        myFile.open(fileName+"_X.csv");
        size_t nRows = myTestingData.getSampleSize();
        saveTitleX(myFile);
        for(size_t iter=0;iter<nRows;++iter){
            saveElement(myTestingData.getSubjectID().at(iter),myFile);
            saveMatrix(myTestingData.getVarX_Cont(),myFile,iter);
            saveMatrix(myTestingData.getVarX_Ord(),myFile,iter);
            saveMatrix(myTestingData.getVarX_Nom(),myFile,iter);
            myFile.seekp(-1, std::ios_base::cur);
            myFile << endl;
        }
        myFile.close();

        //save all the unique treatment codes into a (ordered) set

        myFile.open(fileName+"_Ys.csv");
        saveTitleYs(myFile);
        for(size_t iter=0;iter<nRows;++iter){
            saveElement(myTestingData.getSubjectID().at(iter),myFile);
            saveElement(myTestingData.getVarA().at(iter),myFile);
            size_t indx=0;
            for(auto trtCode:trts){
                saveMatrix(counterfactual_Ys.at(indx++),myFile,iter);
            }
            myFile.seekp(-1, std::ios_base::cur);
            myFile << endl;
        }
        myFile.close();

    } catch (const std::exception& e){
        cout << "ERROR!:" << e.what() << endl;
    }
}


template<typename T>
void SimulateData::printElement(T t){
    cout << left << setw(printWidth) << setfill(separator) <<setprecision(3) << t;
}

template<typename T>
void SimulateData::printMatrix(const T & matrix, size_t row) {
    size_t nCol = matrix.size();
    for(size_t i=0;i<nCol;++i){
      printElement(matrix.at(i).at(row));
    }
}

template<typename T>
void SimulateData::saveElement(T t, ofstream & myFile) {
    myFile << t << ",";
}

template<typename T>
void SimulateData::saveMatrix(const T & matrix, ofstream & myFile, size_t row) {
    size_t nCol=matrix.size();
    for(size_t i=0;i<nCol;++i){
        saveElement(matrix.at(i).at(row), myFile);
    }
}

void SimulateData::saveFullTitle(ofstream & myFile) {

    myFile << "SubID" <<",";
    if(myTrainingData.getVarY().size()==1){
        myFile << "Y"<<",";
    } else{
        for(size_t i=0;i<myTrainingData.getVarY().size();++i){
            myFile << "Y"<<i+1<<",";
        }
    }

    myFile << "Trt" <<",";
    for(size_t i=0;i<myTrainingData.getVarX_Cont().size();++i){
        myFile << "X_Cont"<<i+1<<",";
    }
    for(size_t i=0;i<myTrainingData.getVarX_Ord().size();++i){
        myFile << "X_Ord"<<i+1<<",";
    }
    for(size_t i=0;i<myTrainingData.getVarX_Nom().size();++i){
        myFile << "X_Nom"<<i+1<<",";
    }
    myFile.seekp(-1, std::ios_base::cur);
    myFile << endl;
}

size_t SimulateData::getSampleSizeTraining() const {
    return sampleSizeTraining;
}

void SimulateData::setSampleSizeTraining(size_t sampleSizeTraining) {
    SimulateData::sampleSizeTraining = sampleSizeTraining;
    myTrainingData = DataTable(sampleSizeTraining,inNumOfVars);
}

size_t SimulateData::getSampleSizeTesting() const {
    return sampleSizeTesting;
}

void SimulateData::setSampleSizeTesting(size_t sampleSizeTesting) {
    SimulateData::sampleSizeTesting = sampleSizeTesting;
    myTestingData = DataTable(sampleSizeTesting,inNumOfVars);
}

const DataTable &SimulateData::getTrainingData(){
    simulateTraining();
    return myTrainingData;
}

const DataTable &SimulateData::getTestingData(){
    simulateTesting();
    return myTestingData;
}

void SimulateData::saveTitleX(ofstream & myFile) {
    myFile << "SubID" <<",";
    for(size_t i=0;i<myTrainingData.getVarX_Cont().size();++i){
        myFile << "X_Cont"<<i+1<<",";
    }
    for(size_t i=0;i<myTrainingData.getVarX_Ord().size();++i){
        myFile << "X_Ord"<<i+1<<",";
    }
    for(size_t i=0;i<myTrainingData.getVarX_Nom().size();++i){
        myFile << "X_Nom"<<i+1<<",";
    }
    myFile.seekp(-1, std::ios_base::cur);
    myFile << endl;
}

void SimulateData::saveTitleYs(ofstream & myFile) {
    myFile << "SubID" <<",";
    myFile <<  "Trt" <<",";
    for(auto trtCode:trts){
        if(myTrainingData.getVarY().size()==1){
            myFile << "Y" << "(" << trtCode.first << ")" << ",";
        } else {
            for (size_t i = 0; i < myTrainingData.getVarY().size(); ++i) {
                myFile << "Y" << i + 1 << "(" << trtCode.first << ")" << ",";
            }
        }
    }
    myFile.seekp(-1, std::ios_base::cur);
    myFile << endl;
}

const vector<vector<vector<double>>> &SimulateData::getCounterfactual_Ys() const {
    return counterfactual_Ys;
}











