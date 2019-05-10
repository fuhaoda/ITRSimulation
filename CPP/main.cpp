#include <iostream>
#include "SimulateData.h"

#include <chrono>

using namespace std;
int main() {
    std::cout << "Hello, World!" << std::endl;

    NumOfVars inNumOfVars;
    inNumOfVars.nCont=5;

    SimulateData fhd(30,inNumOfVars,0);
    fhd.setXAY(0,0,0);

    DataTable X=fhd.getTrainingData();


    auto timeNow = chrono::high_resolution_clock::now();

    // random seed using time.
    // unsigned(timeNow.time_since_epoch().count()%100000)

    fhd.printData(X,true);

    fhd.savedAsTraining("abc");
    fhd.getTestingData();
    fhd.savedAsTesting("haha");
    fhd.defaultSaveData("didi");

    return 0;
}