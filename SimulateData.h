//
// Created by Haoda Fu on 5/28/18.
//

/* This is the main class to generate simulated data.
 * It is important to generate training and testing data before save the data. Otherwise, default values (likely to be 0)
 *   will be saved. The following two functions should be called to generate and get the simulated data.
 *   const DataTable &getTrainingData();
 *   const DataTable &getTestingData();
 */


#ifndef ITR2_SIMULATEDATA_H
#define ITR2_SIMULATEDATA_H

#include <iomanip>
#include <fstream>
#include "DataTable.h"
#include "FillUpX.h"
#include "FillUpA.h"
#include "FillUpY.h"

#include "FillUpX_Case1.h"
#include "FillUpA_Case1.h"
#include "FillUpY_Case1.h"

#include <random>
#include<iostream>
#include<map>


class SimulateData {
private:
    //default sample size for training and testing
    size_t sampleSizeTraining=500;
    size_t sampleSizeTesting = 50000;
    //save the treatment ID with their corresponding index, the index starts from 0, 1,2,... It is used to index counterfactual effects
    std::map<int,size_t> trts{};
    //this multi-dimension vector is to save the counterfactual effects.
    // The first dimension is used to save responses for different treatment.
    // The second dimension is used to save rows for different patients
    // The third dimension is used to save for different dimension of response Y. For most cases, it has only 1 dimension.
    std::vector<std::vector<std::vector<double>>> counterfactual_Ys{};
    // Input variables contains the number of covariates and responses
    NumOfVars inNumOfVars;
    //random number seed
    unsigned seed = 0;
    //control for printing out the data on the screen
    const int printWidth = 8;
    const char separator    = ' ';

    //the dimension of the response Y
    size_t yDim;

    //the random simulation engine for each class. This simulation engine will pass by reference into data generation functions
    std::default_random_engine generator{seed};

    //training and testing data. User can split the training data for some validation/tuning purpose by themselves.
    DataTable myTrainingData;
    DataTable myTestingData;

    //We use strategy pattern to allow us dynamically hook up different ways to generate X, A, and Y.
    //The default generation is simply to generate random number to fill in
    FillUpX * fillUpMyX = new FillUpX(generator);
    FillUpA * fillUpMyA = new FillUpA(generator);
    FillUpY * fillUpMyY = new FillUpY(generator);

    //Set ways to generate X, A, and Y.
    void setX(unsigned);
    void setA(unsigned);
    void setY(unsigned);


    //simulate the training and testing data
    void simulateTraining();
    void simulateTesting();

    //print and save data
    template<typename T> void printElement(T t);
    template<typename T> void printMatrix(const T & t, size_t row);

    template<typename T> void saveElement(T, std::ofstream &);
    template<typename T> void saveMatrix(const T&, std::ofstream &, size_t row);
    void saveFullTitle(std::ofstream &);
    void saveTitleX(std::ofstream &);
    void saveTitleYs(std::ofstream &);

public:

    //constructor and destructor. The input is the sample size for training data, dimensions of the covariates, and random seeds.
    explicit SimulateData(size_t sampleSizeTraining, NumOfVars inNumOfVars, unsigned randomSeed);

    ~SimulateData(){
        delete fillUpMyX;
        delete fillUpMyA;
        delete fillUpMyY;
    }



    //set the ways to generate X A and Y. Default ways to generate values will be called if we didn't call this function
    void setXAY(unsigned choiceX, unsigned choiceA, unsigned choiceY);


    //generate and get training and testing data.
    const DataTable &getTrainingData();
    const DataTable &getTestingData();

    //get the counterfactual responses.
    // The first dimension is used to save responses for different treatment.
    // The second dimension is used to save rows for different patients
    // The third dimension is used to save for different dimension of response Y. For most cases, it has only 1 dimension.
    const std::vector<std::vector<std::vector<double>>> &getCounterfactual_Ys() const;

    //getter and setter for sample size
    size_t getSampleSizeTraining() const;
    void setSampleSizeTraining(size_t sampleSizeTraining);
    size_t getSampleSizeTesting() const;
    void setSampleSizeTesting(size_t sampleSizeTesting);


    /// *** The following are utilities for print and save the data ***///

    //print Data is a general utility which requires an input so that we can use it to print any DataTable
    void printData(const DataTable &, bool head);

    // Save data as a single CSV file for training data. We don't need to provide the filename extension
    // The saved training data is the current training data
    void savedAsTraining(std::string);

    // Save data as two multiple files for testing data with true optimal values.
    void savedAsTesting(std::string);

    // Default ways to automatically generate training and testing data.
    // The testing dataset is generated with 50K samples.
    //training and testing data are generated within this function.
    void defaultSaveData(std::string);


};


#endif //ITR2_SIMULATEDATA_H
