//
// Created by Haoda Fu on 5/28/18.
//

// This is a class to hold data. Data contains 3 parts: X, A, and Y
// X: X are covariates. There are three types of covariates: continous, ordinal catagorical variable, and nominal categorical.
// A: Actions which are a integer type. Treatment is a special case of action.
// Y: Y is response. It can be multicatgory. For example, survival outcome often contains observed time and event indicator.
// In our program, we save different types of X into different vectors.

// This header file contains two classes: NumOfVars and DataTable
// NumOfVars defines the dimension of variables for each type.
// DataTable is used to organize the data.

//Todo: we didn't implement move constructor, copy assignment constructor, copy assignment operator

#ifndef ITR2_DATATABLE_H
#define ITR2_DATATABLE_H

#include <vector>

/*
 * This class define the number of covariates and responses
 */
struct NumOfVars{
    size_t nCont=1; //number of continuous variable
    size_t nOrd=1; //number of ordinal categorical variable
    size_t nNom=1; //number nominal categorical variable
    size_t nVarY=1; //number of responses
};

class DataTable {
private:
    //For all the vector of vector below. The first index is for the rows,
    // and the second index is for the columns of the covariates or the responses.
    std::vector<std::vector<double>> varY;
    std::vector<int> varA;
    std::vector<std::vector<double>> varX_Cont;
    std::vector<std::vector<int>> varX_Ord;
    std::vector<std::vector<int>> varX_Nom;
    std::vector<unsigned int> subjectID;
    size_t sampleSize;
    size_t n_varX_Cont;
    size_t n_varX_Ord;
    size_t n_varX_Nom;
    size_t n_varY;
public:

    //standard setter and getters
    size_t getSampleSize() const;

    size_t getN_varX_Cont() const;

    size_t getN_varX_Ord() const;

    size_t getN_varX_Nom() const;

    const std::vector<std::vector<double>> &getVarY() const;

    void setVarY(const std::vector<std::vector<double>> &varY);

    const std::vector<int> &getVarA() const;

    void setVarA(const std::vector<int> &varA);

    const std::vector<std::vector<double>> &getVarX_Cont() const;

    void setVarX_Cont(const std::vector<std::vector<double>> &varX_Cont);

    const std::vector<std::vector<int>> &getVarX_Ord() const;

    void setVarX_Ord(const std::vector<std::vector<int>> &varX_Ord);

    const std::vector<std::vector<int>> &getVarX_Nom() const;

    void setVarX_Nom(const std::vector<std::vector<int>> &varX_Nom);

    const std::vector<unsigned int> &getSubjectID() const;

    void setSubjectID(const std::vector<unsigned int> &subjectID);


    //expose the reference so that we can directly operate on it.

    std::vector<std::vector<double>> &ref_VarX_Cont();
    std::vector<std::vector<int>> &ref_VarX_Ord();
    std::vector<std::vector<int>> &ref_VarX_Nom();
    std::vector<std::vector<double>> &ref_VarY();
    std::vector<int> &ref_VarA();

    //ctor
    DataTable(size_t sampleSize, NumOfVars inNumOfVars);
};


#endif //ITR2_DATATABLE_H
