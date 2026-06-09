#ifndef SOLVER_H_
#define SOLVER_H_

#include <vector>

#include "mpsReader.h"

enum class ResultType {
    FEASIBLE,
    INFEASIBLE,
    UNBOUNDED,
    NONE
};

struct SolutionResult {
    ResultType type;
    VectorXd variables;
    double objective;

    vector<int> basis;
};

class Solver {
public:
    Solver(mpsReader reader);
    ~Solver() = default;

    SolutionResult solve();
private:
    SolutionResult solveFromBasicSolution(vector<int> start_basis, VectorXd basic_solution, bool phase1);
    std::vector<int> calculateNonBasicFromBasic(const std::vector<int>& basis) const;

    void updateB0(Eigen::SparseMatrix<double>& B, const vector<int>& basis) const;
    
    mpsReader m_reader;

    
};

#endif
