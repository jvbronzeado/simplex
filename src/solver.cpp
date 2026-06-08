#include "solver.h"
#include "eigen3/Eigen/UmfPackSupport"

#define EPSILON 1e-9
#define SMEPSILON 1e-12

Solver::Solver(mpsReader reader) {
    this->m_reader = reader;
    this->m_reader.printData();
}

bool Solver::solve() {
    const int rows = this->m_reader.A.rows();
    const int cols = this->m_reader.A.cols();

    // use the created artificial variables from reader as starting basic solution
    vector<int> start_basis(rows);
    for(int i = 0; i < rows; i++) {
        start_basis[i] = cols - i - 1;
    }

    // generate basic solution
    VectorXd solution(cols);
    solution << VectorXd::Zero(cols);
    solution.tail(rows) = this->m_reader.b.tail(rows);

    return this->solveFromBasicSolution(start_basis, solution);    
}

bool Solver::solveFromBasicSolution(vector<int> basis, VectorXd solution) {
    const int rows = this->m_reader.A.rows();

    Eigen::SparseMatrix<double> A = this->m_reader.A.sparseView();    
    Eigen::SparseMatrix<double> B(rows, rows);
    
    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> solver;
    Eigen::VectorXi nonzero_per_col(rows);

    vector<int> nonbasic = calculateNonBasicFromBasic(basis);

    // simplex loop
    while(true) {
        // TODO: Factorization
        
        // STEP 1: calculate yB = cb
        // cout << "STEP 1" << endl;
        for(int i = 0; i < rows; i++) {
            nonzero_per_col[i] = A.col(basis[i]).nonZeros();
        }

        B.reserve(nonzero_per_col); // allocate memory for sparse matrix
        
        VectorXd costs = VectorXd(rows);
        for(int i = 0; i < rows; i++) {
            costs[i] = this->m_reader.c[basis[i]]; // fill with basis cost values
            B.col(i) = A.col(basis[i]); // fill basis coefficients
        }
        
        B.makeCompressed();

        solver.compute(B.transpose());
        if(solver.info() != Eigen::Success) {
            std::cerr << "failed to compute Umfpack solver for B transpose" << std::endl;
            return false;
        }

        VectorXd y = solver.solve(costs);
        if(solver.info() != Eigen::Success) {
            std::cerr << "failed to calculate dual vector's linear system" << std::endl;
            return false;
        }

        // STEP 2: Choose entering collumn
        // cout << "STEP 2" << endl;
        int best_col = -1;
        double best_cr = -1;
        for(int colid : nonbasic) {
            double cost = this->m_reader.c[colid];
            double ub = this->m_reader.ub[colid];
            double lb = this->m_reader.lb[colid];

            double cr = A.col(colid).dot(y);

            if((cr < cost - EPSILON && solution[colid] < ub - EPSILON) || (cr > cost + EPSILON && solution[colid] > lb + EPSILON)) {
                best_col = colid;
                best_cr = cr;
                break;
            }
        }

        if(best_col == -1) {
            // no way to improve
            break;
        }

        // STEP 3: Solve Bd = a
        // cout << "STEP 3" << endl;

        solver.compute(B);
        if(solver.info() != Eigen::Success) {
            std::cerr << "failed to compute Umfpack solver for B" << std::endl;
            return false;
        }
        
        VectorXd d = solver.solve(A.col(best_col));
        if(solver.info() != Eigen::Success) {
            std::cerr << "failed to calculate d vector's linear system" << std::endl;
            return false; 
        }

        // STEP 4: Define t
        // cout << "STEP 4" << endl;
        double multiplier = (best_cr < this->m_reader.c[best_col] - SMEPSILON) ? 1 : -1;
        double entering_ub = this->m_reader.ub[best_col];
        double entering_lb = this->m_reader.lb[best_col];
        double entering_bound = multiplier > 0 ? entering_ub : entering_lb;
        double bound_clamp = abs(solution[best_col] - entering_bound);
        double t = std::numeric_limits<double>::infinity();
        int leaving_col = -1;

       
        for(int i = 0; i < rows; i++) {
            int colid = basis[i];

            double ub = this->m_reader.ub[colid];
            double lb = this->m_reader.lb[colid];

            if(abs(d[i]) < SMEPSILON) { // if d[i] is 0 then maxt will be infinite
                continue;
            }
            
            double bound = (-d[i] * multiplier) > 0 ? ub : lb;
            double maxt = abs((solution[colid] - bound) / (d[i]));
            maxt = min(maxt, bound_clamp);

            if(maxt == std::numeric_limits<double>::infinity()) {
                continue;
            }

            if((abs(t - maxt) < SMEPSILON && colid < t) || (abs(t - maxt) > SMEPSILON && maxt < t - SMEPSILON)) {
                t = maxt;
                leaving_col = colid;
            }
        }        

        if(leaving_col == -1 || abs(t) == std::numeric_limits<double>::infinity()) {
            break;
        }

        // STEP 5: Replace values
        //cout << "STEP 5" << endl;

        double old_solution_val = solution[best_col];
        solution[best_col] += multiplier * t;
        
        for(int i = 0; i < rows; i++) {
            solution[basis[i]] -= multiplier * t * d[i];
        }

        if(t >= EPSILON) { // bound switch test
            if(abs(solution[best_col] - entering_ub) < SMEPSILON || abs(solution[best_col] - entering_lb) < SMEPSILON) {
                if(abs(old_solution_val - entering_ub) < SMEPSILON || abs(old_solution_val - entering_lb) < SMEPSILON) {
                    continue;
                }
            }
        }

        // swap collumns
        int basis_id = std::find(basis.begin(), basis.end(), leaving_col) - basis.begin();
        int nonbasic_id = std::find(nonbasic.begin(), nonbasic.end(), best_col) - nonbasic.begin();

        basis[basis_id] = best_col;
        nonbasic[nonbasic_id] = leaving_col;
    }

    cout << solution << endl;
    cout << "finished simplex" << endl;
    
    return true;
}

vector<int> Solver::calculateNonBasicFromBasic(vector<int> basis) {
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
