#ifndef SOLVER_H_
#define SOLVER_H_

#include <vector>

#include "mpsReader.h"

class Solver {
public:
    Solver(mpsReader reader);
    ~Solver() = default;

    bool solve();
    bool solveFromBasicSolution(vector<int> start_basis, VectorXd basic_solution);
private:
    std::vector<int> calculateNonBasicFromBasic(std::vector<int> basis);
    
    mpsReader m_reader;
};

#endif
