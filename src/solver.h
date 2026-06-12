#ifndef SOLVER_H_
#define SOLVER_H_

#include <vector>
#include "eigen3/Eigen/UmfPackSupport"
#include "mpsReader.h"

enum class ResultType {
    FEASIBLE,
    INFEASIBLE,
    UNBOUNDED,
    NONE
};

class SolverException : public std::runtime_error {
public:
    explicit SolverException(ResultType type, const std::string& message) : std::runtime_error(message) {
        this->m_err = type;       
    }

    ResultType result() {
        return this->m_err;
    }
private:
    ResultType m_err;
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
    struct SolverETA {
        int index; // 0 to row-1
        VectorXd etavector;
    };

    struct EnteringCollumnData {
        double best_ya;
        int entering_col;
        int entering_col_id;
    };

    struct LeavingCollumnData {
        double t;
        int leaving_col;
        int leaving_col_id;
    };

    SolutionResult solveFromBasicSolution(vector<int> start_basis, VectorXd basic_solution, bool phase1);

    void initializeSolvers();
    void updateBcosts();
        
    EnteringCollumnData findEntering() const;
    LeavingCollumnData findLeaving(int entering_col, double best_ya) const;
    void ftran(int entering_col);
    void btran();
    void updateSolution(int entering_col, double t);
    void swapCollumns(int entering_col, int leaving_col);
    
    std::vector<int> calculateNonBasicFromBasic(const std::vector<int>& basis) const;

    void updateB0();

    int updatePhaseCosts();
    
    mpsReader m_reader;

    VectorXd m_activeCosts;
    VectorXd m_activeLB;
    VectorXd m_activeUB;

    // solver variables
    VectorXd m_solution;
    VectorXd m_y;
    VectorXd m_d;

    vector<int> m_basis;
    vector<int> m_nonbasis;
    VectorXd m_bcosts;

    vector<SolverETA> m_etas;

    Eigen::SparseMatrix<double> m_B;
    Eigen::SparseMatrix<double> m_Asv;

    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> m_nsolver; // normal solver (B)
    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> m_tsolver; // transpose solver (B^t)
};

#endif
