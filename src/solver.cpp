#include "solver.h"

#define MAX_ITERATIONS_PER_RECALC 20
#define EPSILON 1e-6
#define SMEPSILON 1e-9

#define EIGEN_ERRDETECT(solver, err_msg) if((solver).info() != Eigen::Success) { \
    throw SolverException(ResultType::INFEASIBLE, (err_msg)); \
}

Solver::Solver(mpsReader reader) {
    this->m_reader = reader;
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

    VectorXd solution = VectorXd::Zero(cols);
    for(int i = 0; i < nonbasic_count; i++) { // we only iterate to the cols - rows because the rest is in the basis
        double lb = this->m_reader.lb[i];
        double ub = this->m_reader.ub[i];

        if(!isinf(lb))
            solution[i] = lb;
        else if(!isinf(ub))
            solution[i] = ub;
        else
            solution[i] = 0.0;
    }
    
    // calculate the basis solution
    VectorXd rhs = this->m_reader.b - this->m_reader.A.block(0, 0, rows, nonbasic_count) * solution.head(nonbasic_count);

    this->m_B = Eigen::SparseMatrix<double>(rows, rows);
    this->m_basis = start_basis;
    this->updateB0();

    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> solver(this->m_B);
    EIGEN_ERRDETECT(solver, "failed to compute first phase B matrix");

    solution.tail(rows) = solver.solve(rhs);
    EIGEN_ERRDETECT(solver, "failed to solve for initial basic solution");

    // solve subproblem
    cout << "solving first phase" << endl;
    SolutionResult result = this->solveFromBasicSolution(start_basis, solution, true);
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

    cout << "solving second phase" << endl;
    return this->solveFromBasicSolution(result.basis, result.variables, false);
}

SolutionResult Solver::solveFromBasicSolution(vector<int> start_basis, VectorXd start_solution, bool phase1) {
    static SolutionResult invalid_solution = {.type = ResultType::INFEASIBLE};
    const size_t rows = this->m_reader.A.rows();
    const size_t cols = this->m_reader.A.cols();

    if(start_solution.size() < cols)
        throw SolverException(ResultType::NONE, "start solution has not enough size");
    if(start_basis.size() < rows)
        throw SolverException(ResultType::NONE, "start basis has not enough size");

    this->m_Asv = this->m_reader.A.sparseView();

    this->m_activeCosts = this->m_reader.c;
    this->m_activeLB = this->m_reader.lb;
    this->m_activeUB = this->m_reader.ub;
    
    this->m_basis = start_basis;
    this->m_nonbasis = calculateNonBasicFromBasic(start_basis);
    this->m_solution = start_solution;
    this->m_etas.clear();

    if(phase1) {
        this->updatePhaseCosts();
    }

    this->updateBcosts();
    
    // calculate B0
    this->m_B = Eigen::SparseMatrix<double>(rows, rows);
    this->updateB0();

    // compute solvers
    this->m_nsolver.compute(this->m_B);
    EIGEN_ERRDETECT(this->m_nsolver, "failed to compute normal solver")

    this->m_tsolver.compute(this->m_B.transpose());
    EIGEN_ERRDETECT(m_tsolver, "failed to compute transpose solver")

    // STEP 1: calculate starting y with yB = cb
    this->m_y = m_tsolver.solve(this->m_bcosts);
    EIGEN_ERRDETECT(m_tsolver, "failed to calculate dual vector's linear system");

    SolutionResult result = {};
    result.type = ResultType::FEASIBLE;

    int iterations = 0;
    while(true) {
        if(phase1) {
            int count = this->updatePhaseCosts();
            if(count == 0) {
                // no variable outside of limits
                cout << "no outside variables" << endl;
                break;
            }
            
            this->updateBcosts();
            this->btran();
        }

        auto [entering_col, best_ya] = this->findEntering();
        if(entering_col == -1) {
            // no way to improve
            cout << "no improve" << endl;
            break;
        }

        this->ftran(entering_col);

        LeavingCollumnData leavingData = this->findLeaving(entering_col, best_ya);
        if(leavingData.leaving_col == -1) {
            if(isinf(leavingData.t)) {
                cout << "unbounded" << endl;
                result.type = ResultType::UNBOUNDED;
                break;
            }

            // border swap
            cout << "border swap" << endl;
            this->updateSolution(entering_col, leavingData.t);
            this->btran();
            continue;
        }

        this->updateSolution(entering_col, leavingData.t);
        this->swapCollumns(entering_col, leavingData.leaving_col);
        
        iterations++;
        if(iterations >= MAX_ITERATIONS_PER_RECALC) {
            iterations = 0;
            this->updateB0();

            this->m_nsolver.compute(this->m_B);
            EIGEN_ERRDETECT(this->m_nsolver, "failed to refactor normal B matrix");

            this->m_tsolver.compute(this->m_B.transpose());
            EIGEN_ERRDETECT(this->m_tsolver, "failed to refactor transpose B matrix");

            this->m_etas.clear();
            this->updateBcosts();
        }

        if(!phase1)
            btran();
    }

    result.variables = this->m_solution;
    result.basis = this->m_basis;
    result.objective = this->m_activeCosts.dot(result.variables);

    return result;
}

void Solver::updateBcosts() {
    const int rows = this->m_reader.A.rows();
    this->m_bcosts = VectorXd(rows);
    for(int i = 0; i < rows; i++) {
        this->m_bcosts[i] = this->m_activeCosts[m_basis[i]]; // fill with basis cost values
    }
}

pair<int, double> Solver::findEntering() const {
    int entering_col = -1;
    double best_ya = -1;
    for(int colid : this->m_nonbasis) {
        double cost = this->m_activeCosts[colid];
        double ub = this->m_activeUB[colid];
        double lb = this->m_activeLB[colid];

        double ya = this->m_reader.A.col(colid).dot(this->m_y);

        if((ya < cost - EPSILON && this->m_solution[colid] < ub - EPSILON) || (ya > cost + EPSILON && this->m_solution[colid] > lb + EPSILON)) {
            if(entering_col == -1 || colid < entering_col) { // always select lower index to avoid cycling
                entering_col = colid;
                best_ya = ya;
            }
        }
    }

    return {entering_col, best_ya};
}

Solver::LeavingCollumnData Solver::findLeaving(int entering_col, double best_ya) const {
    const int rows = this->m_reader.A.rows();

    double cost_reduction = this->m_activeCosts[entering_col] - best_ya;
    double multiplier = (cost_reduction > -EPSILON) ? 1 : -1;

    double entering_ub = this->m_activeUB[entering_col];
    double entering_lb = this->m_activeLB[entering_col];

    double t = multiplier > 0 ? (entering_ub - this->m_solution[entering_col]) : (this->m_solution[entering_col] - entering_lb);
    int leaving = -1;

    for(int i = 0; i < rows; i++) {
        int colid = this->m_basis[i];

        if(abs(this->m_d[i]) <= EPSILON)
            continue;

        double slope = -this->m_d[i] * multiplier;
        double maxt = -1;
        if(slope < -EPSILON) {
            maxt = -(this->m_solution[colid] - this->m_activeLB[colid]) / (slope);
        }
        else if(slope > EPSILON) {
            maxt = (this->m_activeUB[colid] - this->m_solution[colid]) / (slope);
        }

        if(maxt < -EPSILON || isinf(maxt))
            continue;

        if(maxt < SMEPSILON) {
            maxt = 0;
        }

        if(maxt < t - EPSILON) {
            t = maxt;
            leaving = colid;
        }
        else if(abs(t - maxt) < EPSILON) {
            if(leaving == -1 || colid < leaving)
                leaving = colid;
        }
    }

    return {    
        .t = t * multiplier,
        .leaving_col = leaving
    };
}

void Solver::ftran(int entering_col) {
    this->m_d = this->m_nsolver.solve(this->m_Asv.col(entering_col)); // set starting rhs value
    EIGEN_ERRDETECT(this->m_nsolver, "failed to calculate start factorization of d vector");

    // forward transformation
    for(SolverETA& eta : this->m_etas) {
        double val = this->m_d[eta.index] / eta.etavector[eta.index];
        this->m_d -= eta.etavector * val;
        this->m_d[eta.index] = val;
    }
}

void Solver::btran() {
    VectorXd rhs_val = this->m_bcosts;
    for(int i = this->m_etas.size()-1; i >= 0; i--) {
        SolverETA& eta = this->m_etas[i];

        double cost = rhs_val[eta.index];
        double val = eta.etavector[eta.index];
        rhs_val[eta.index] = (cost * (1 + val) - eta.etavector.dot(rhs_val)) / val;
    }

    this->m_y = this->m_tsolver.solve(rhs_val);
    EIGEN_ERRDETECT(this->m_tsolver, "failed to calculate dual vector's linear system");
}

void Solver::updateSolution(int entering_col, double t) {
    this->m_solution[entering_col] += t;

    for(int i = 0; i < this->m_reader.A.rows(); i++) {
        if(abs(this->m_d[i]) <= EPSILON || abs(t) <= EPSILON)
            continue;
        this->m_solution[this->m_basis[i]] -= t * this->m_d[i];
    }
}

void Solver::swapCollumns(int entering_col, int leaving_col) {   
    int basis_id = std::find(this->m_basis.begin(), this->m_basis.end(), leaving_col) - this->m_basis.begin();
    int nonbasic_id = std::find(this->m_nonbasis.begin(), this->m_nonbasis.end(), entering_col) - this->m_nonbasis.begin();

    this->m_basis[basis_id] = entering_col;
    this->m_nonbasis[nonbasic_id] = leaving_col;
    
    this->m_bcosts[basis_id] = this->m_activeCosts[entering_col]; // update costs value

    // add eta vector
    this->m_etas.push_back({
        .index = basis_id,
        .etavector = this->m_d
    });
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

void Solver::updateB0() {
    Eigen::SparseMatrix<double> A = this->m_reader.A.sparseView();
    const int rows = this->m_reader.A.rows();
    
    // Lista segura de posições e valores para matrizes esparsas
    std::vector<Eigen::Triplet<double>> triplets;
    
    for(int i = 0; i < rows; i++) {
        // Itera sobre os não-zeros da coluna original em A e mapeia para a nova coluna i em B
        for(Eigen::SparseMatrix<double>::InnerIterator it(A, this->m_basis[i]); it; ++it) {
            triplets.push_back(Eigen::Triplet<double>(it.row(), i, it.value()));
        }
    }

    // Instancia a matriz vazia e preenche de uma vez só com a lista
    this->m_B = Eigen::SparseMatrix<double>(rows, rows);
    this->m_B.setFromTriplets(triplets.begin(), triplets.end());
    this->m_B.makeCompressed();
}

int Solver::updatePhaseCosts() {
    int count = 0;

    for(int i = 0; i < this->m_reader.A.cols(); i++) {
        double ub = this->m_reader.ub[i];
        double lb = this->m_reader.lb[i];
        this->m_activeUB[i] = ub;
        this->m_activeLB[i] = lb;

        if(abs(this->m_solution[i] - this->m_reader.ub[i]) <= EPSILON) {
            this->m_solution[i] = this->m_reader.ub[i];
        }
        else if(abs(this->m_solution[i] - this->m_reader.lb[i]) <= EPSILON) {
            this->m_solution[i] = this->m_reader.lb[i];
        }

        if(this->m_solution[i] > this->m_reader.ub[i] + EPSILON) {
            this->m_activeCosts[i] = -1;
            this->m_activeUB[i] = std::numeric_limits<double>::infinity();
            this->m_activeLB[i] = ub;
            count++;
        }
        else if(this->m_solution[i] < this->m_reader.lb[i] - EPSILON) {
            this->m_activeCosts[i] = 1;
            this->m_activeUB[i] = lb;
            this->m_activeLB[i] = -std::numeric_limits<double>::infinity();
            count++;
        }
        else {
            this->m_activeCosts[i] = 0;
        }
    }
    return count;
}
