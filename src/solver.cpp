#include "solver.h"
#include "eigen3/Eigen/UmfPackSupport"

#define EPSILON 1e-9
#define SMEPSILON 1e-12

#define EIGEN_ERRDETECT(solver, returnval, err_msg) if((solver).info() != Eigen::Success) { \
    std::cerr << (err_msg); \
    return (returnval); \
}

struct Eta {
    int index; // 0 to row-1
    VectorXd etavector;
};

Solver::Solver(mpsReader reader) {
    this->m_reader = reader;
    //this->m_reader.c *= -1;
}

SolutionResult Solver::solve() {
    static SolutionResult invalid_solution = {.type = ResultType::INFEASIBLE};

    const int rows = this->m_reader.A.rows();
    const int cols = this->m_reader.A.cols();
    const int nonbasic_count = cols - rows;

    // use the created artificial variables from reader as starting basic solution
    vector<int> start_basis(rows);
    for(int i = 0; i < rows; i++) {
        start_basis[i] = nonbasic_count + i;
    }

    // generate basic solution
    VectorXd solution = VectorXd::Zero(cols);
    for(int i = 0; i < nonbasic_count; i++) { // we only iterate to the cols - rows because the rest is in the basis
        solution[i] = max(this->m_reader.lb[i], std::numeric_limits<double>::lowest());
    }

    VectorXd rhs = this->m_reader.b - this->m_reader.A.block(0, 0, rows, nonbasic_count) * solution.head(nonbasic_count);

    Eigen::SparseMatrix<double> B(rows, rows);
    this->updateB0(B, start_basis);

    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> solver(B);
    EIGEN_ERRDETECT(solver, invalid_solution, "failed to compute first phase B matrix");

    solution.tail(rows) = solver.solve(rhs);
    EIGEN_ERRDETECT(solver, invalid_solution, "failed to solve for initial basic solution");

    // create a new mpsReader with updated costs and bound values
    mpsReader subproblemReader = this->m_reader;

    // configure initial costs
    subproblemReader.c.setZero();
    for(int i = 0; i < cols; i++) {
        if(solution[i] < this->m_reader.lb[i] - EPSILON) {
            subproblemReader.c[i] = 1;
        }
        else if(solution[i] > this->m_reader.ub[i] + EPSILON) {
            subproblemReader.c[i] = -1;
        }
    }

    Solver subSolver(subproblemReader);
    SolutionResult result = subSolver.solveFromBasicSolution(start_basis, solution, true);
    if(result.type == ResultType::INFEASIBLE) {
        cout << "infeasible" << endl;
        return invalid_solution;
    }
    else if(result.type == ResultType::UNBOUNDED) {
        cout << "unbounded" << endl;
        return {
            .type = ResultType::UNBOUNDED
        };
    }

    cout << "Variables: " << result.variables << "\nObjective: " << result.objective << endl;
    return this->solveFromBasicSolution(result.basis, result.variables, false);
}

SolutionResult Solver::solveFromBasicSolution(vector<int> basis, VectorXd solution, bool phase1) {
    static SolutionResult invalid_solution = {.type = ResultType::INFEASIBLE};
    const int rows = this->m_reader.A.rows();
    const int cols = this->m_reader.A.cols();
    
    Eigen::SparseMatrix<double> A = this->m_reader.A.sparseView();

    vector<int> nonbasic = calculateNonBasicFromBasic(basis);
    vector<Eta> etas;

    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> normal_solver;
    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> transp_solver;

    // calculate B0
    Eigen::SparseMatrix<double> B(rows, rows);
    this->updateB0(B, basis);

    // create basis costs vector
    VectorXd bcosts = VectorXd(rows);
    for(int i = 0; i < rows; i++) {
        bcosts[i] = this->m_reader.c[basis[i]]; // fill with basis cost values
    }

    // compute solvers
    normal_solver.compute(B);
    EIGEN_ERRDETECT(normal_solver, invalid_solution, "failed to compute normal solver")

    transp_solver.compute(B.transpose());
    EIGEN_ERRDETECT(transp_solver, invalid_solution, "failed to compute transpose solver")

    // STEP 1: calculate starting y with yB = cb
    VectorXd y = transp_solver.solve(bcosts);
    EIGEN_ERRDETECT(transp_solver, invalid_solution, "failed to calculate dual vector's linear system");

    SolutionResult result = {};
    result.type = ResultType::FEASIBLE;
    
    while(true) {
        // TODO: maybe recalculate the B and clear etas after some iterations to avoid precisions errors

        // Setup iteration
        VectorXd iteration_costs = this->m_reader.c;
        VectorXd iteration_lb = this->m_reader.lb;
        VectorXd iteration_ub = this->m_reader.ub;
        if(phase1) {
            for(int i = 0; i < cols; i++) {
                if(solution[i] > iteration_ub[i] + EPSILON) {
                    iteration_costs[i] = -1;
                    iteration_ub[i] = std::numeric_limits<double>::infinity();
                }
                else if(solution[i] < iteration_lb[i] - EPSILON) {
                    iteration_costs[i] = 1;
                    iteration_lb[i] = -std::numeric_limits<double>::infinity();
                }
                else {
                    iteration_costs[i] = 0;
                }
            }

            // update bcosts
            for(int i = 0; i < rows; i++) {
                bcosts[i] = iteration_costs[basis[i]];
            }
        }
        
        // STEP 2: Choose entering collumn
        int entering_col = -1;
        double best_cr = -1;
        for(int colid : nonbasic) {
            double cost = iteration_costs[colid];
            double ub = iteration_ub[colid];
            double lb = iteration_lb[colid];

            double cr = A.col(colid).dot(y);

            if((cr < cost - EPSILON && solution[colid] < ub - EPSILON) || (cr > cost + EPSILON && solution[colid] > lb + EPSILON)) {
                entering_col = colid;
                best_cr = cr;
                break;
            }
        }

        if(entering_col == -1) {
            // no way to improve
            break;
        }

        // STEP 3: Solve Bd = a
        VectorXd d = normal_solver.solve(A.col(entering_col)); // set starting rhs value
        EIGEN_ERRDETECT(normal_solver, invalid_solution, "failed to calculate start factorization of d vector");

        for(Eta& eta : etas) {
            double val = d[eta.index] / eta.etavector[eta.index];
            d -= eta.etavector * val;
            d[eta.index] = val;
        }

        // STEP 4: Define t and find leaving collumn
        double multiplier = (best_cr < iteration_costs[entering_col] - SMEPSILON) ? 1 : -1;
        double entering_ub = iteration_ub[entering_col];
        double entering_lb = iteration_lb[entering_col];
        double entering_bound = multiplier > 0 ? entering_ub : entering_lb;
        double bound_clamp = abs(solution[entering_col] - entering_bound);
        double t = std::numeric_limits<double>::infinity();
        int leaving_col = -1;

       
        for(int i = 0; i < rows; i++) {
            int colid = basis[i];

            double ub = iteration_ub[colid];
            double lb = iteration_lb[colid];

            if(abs(d[i]) < SMEPSILON) { // if d[i] is 0 then maxt will be infinite
                continue;
            }
            
            double bound = (-d[i] * multiplier) > 0 ? ub : lb;
            double maxt = abs((solution[colid] - bound) / (d[i]));
            maxt = min(maxt, bound_clamp);

            if(maxt == std::numeric_limits<double>::infinity()) {
                continue;
            }

            if((abs(t - maxt) <= SMEPSILON && colid < leaving_col) || (abs(t - maxt) > SMEPSILON && maxt < t - SMEPSILON)) {
                t = maxt;
                leaving_col = colid;
            }
        }        

        if(leaving_col == -1 || abs(t) == std::numeric_limits<double>::infinity()) {
            result.type = ResultType::UNBOUNDED;
            break;
        }

        // STEP 5: Replace values
        double old_solution_val = solution[entering_col];
        solution[entering_col] += multiplier * t;
        
        for(int i = 0; i < rows; i++) {
            solution[basis[i]] -= multiplier * t * d[i];
        }

        if(t >= EPSILON) { // bound switch test
            if(abs(solution[entering_col] - entering_ub) < SMEPSILON || abs(solution[entering_col] - entering_lb) < SMEPSILON) {
                if(abs(old_solution_val - entering_ub) < SMEPSILON || abs(old_solution_val - entering_lb) < SMEPSILON) {
                    continue;
                }
            }
        }

        // swap collumns
        int basis_id = std::find(basis.begin(), basis.end(), leaving_col) - basis.begin();
        int nonbasic_id = std::find(nonbasic.begin(), nonbasic.end(), entering_col) - nonbasic.begin();

        basis[basis_id] = entering_col;
        nonbasic[nonbasic_id] = leaving_col;

        bcosts[basis_id] = iteration_costs[entering_col]; // update costs value

        // get eta vector
        etas.push_back({
            .index = basis_id,
            .etavector = d
        });

        // update the y vector
        VectorXd rhs_val = bcosts;
        for(int i = etas.size()-1; i >= 0; i--) {
            Eta& eta = etas[i];
            double cost = rhs_val[eta.index];
            double val = eta.etavector[eta.index];
            rhs_val[eta.index] = (cost * (1 + val) - eta.etavector.dot(rhs_val)) / val;
        }

        y = transp_solver.solve(rhs_val);
        EIGEN_ERRDETECT(transp_solver, invalid_solution, "failed to calculate dual vector's linear system");
    }

    result.variables = solution;
    result.basis = basis;
    result.objective = this->m_reader.c.dot(result.variables);

    return result;
}

vector<int> Solver::calculateNonBasicFromBasic(const vector<int>& basis) const {
    const int cols = this->m_reader.A.cols();
    const int rows = this->m_reader.A.rows();
    vector<bool> mask(cols, false);
    for(int i : basis) {
        mask[i] = true; // set mask for identifying base indices
    }

    int l = 0;
    vector<int> output(cols - rows);
    for(int i = 0; i < cols; i++) {
        if(!mask[i])
            output[l++] = i;
    }
    return output;
}

void Solver::updateB0(Eigen::SparseMatrix<double>& B, const vector<int>& basis) const {
    const int rows = this->m_reader.A.rows();
    Eigen::SparseMatrix<double> A = this->m_reader.A.sparseView();
    
    Eigen::VectorXi nonzero_per_col(rows);
    for(int i = 0; i < rows; i++) {
        nonzero_per_col[i] = A.col(basis[i]).nonZeros();
    }

    B.reserve(nonzero_per_col); // allocate memory for sparse matrix

    for(int i = 0; i < rows; i++) {
        B.col(i) = A.col(basis[i]); // fill basis coefficients
    }

    B.makeCompressed();
}
