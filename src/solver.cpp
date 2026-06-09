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

    vector<int> nonbasic = calculateNonBasicFromBasic(basis);
    vector<Eta> etas;

    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> normal_solver;
    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> transp_solver;

    // calculate B0
    Eigen::SparseMatrix<double> B(rows, rows);

    Eigen::VectorXi nonzero_per_col(rows);
    for(int i = 0; i < rows; i++) {
        nonzero_per_col[i] = A.col(basis[i]).nonZeros();
    }

    B.reserve(nonzero_per_col); // allocate memory for sparse matrix

    for(int i = 0; i < rows; i++) {
        B.col(i) = A.col(basis[i]); // fill basis coefficients
    }

    B.makeCompressed();

    // create basis costs vector
    VectorXd bcosts = VectorXd(rows);
    for(int i = 0; i < rows; i++) {
        bcosts[i] = this->m_reader.c[basis[i]]; // fill with basis cost values
    }

    // compute solvers
    normal_solver.compute(B);
    EIGEN_ERRDETECT(normal_solver, false, "failed to compute normal solver")

    transp_solver.compute(B.transpose());
    EIGEN_ERRDETECT(transp_solver, false, "failed to compute transpose solver")

    // STEP 1: calculate starting y with yB = cb
    // cout << "STEP 1" << endl;
    VectorXd y = transp_solver.solve(bcosts);
    EIGEN_ERRDETECT(transp_solver, false, "failed to calculate dual vector's linear system");

    // simplex loop
    while(true) {
        // TODO: maybe recalculate the B and clear etas after some iterations to avoid precisions errors
                
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
        
        VectorXd d = normal_solver.solve(A.col(best_col)); // set starting rhs value
        EIGEN_ERRDETECT(normal_solver, false, "failed to calculate start factorization of d vector");

        for(Eta& eta : etas) {
            double val = d[eta.index];
            d += eta.etavector * d[eta.index];
            d[eta.index] -= val;
        }

        // ^ after that loop, d is correct

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

        bcosts[basis_id] = this->m_reader.c[best_col]; // update costs value

        // get eta vector
        VectorXd inverse_eta = d;

        // compute the inverse eta, cause thats what we're going to use
        double det = d[basis_id];
        for(int i = 0; i < rows; i++) {
            if(i == basis_id)
                inverse_eta[i] = 1/det;
            else
                inverse_eta[i] = -d[i]/det;
        }
        
        etas.push_back({
            .index = basis_id,
            .etavector = inverse_eta
        });

        // update the y vector
        VectorXd rhs_val = bcosts;
        for(int i = etas.size()-1; i >= 0; i--) {
            Eta& eta = etas[i];
            rhs_val[eta.index] = rhs_val.dot(eta.etavector);
        }

        y = transp_solver.solve(rhs_val);
        EIGEN_ERRDETECT(transp_solver, false, "failed to calculate dual vector's linear system");
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
