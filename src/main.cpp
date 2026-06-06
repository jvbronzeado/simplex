#include <iostream>

#include <suitesparse/umfpack.h>

#include "mpsReader.h"

int main(const int argc, const char* argv[]) {
    mpsReader reader;
    
    std::int32_t n = 5;

    std::vector<std::int32_t> Ap = {0, 2, 5, 9, 10, 12};
    std::vector<std::int32_t> Ai = {0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4};
    std::vector<double> Ax = {2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.};
    std::vector<double> b = {8., 45., -3., 3., 19.};
    std::vector<double> x(n, 0.0);

    void* Symbolic = nullptr;
    void* Numeric = nullptr;

    int status = umfpack_di_symbolic(n, n, Ap.data(), Ai.data(), Ax.data(), &Symbolic, nullptr, nullptr);
    if (status != UMFPACK_OK) {
        std::cerr << "Symbolic factorization failed." << std::endl;
        return 1;
    }

    status = umfpack_di_numeric(Ap.data(), Ai.data(), Ax.data(), Symbolic, &Numeric, nullptr, nullptr);
    umfpack_di_free_symbolic(&Symbolic); 
    
    if (status != UMFPACK_OK) {
        std::cerr << "Numerical factorization failed." << std::endl;
        return 1;
    }

    status = umfpack_di_solve(UMFPACK_A, Ap.data(), Ai.data(), Ax.data(), x.data(), b.data(), Numeric, nullptr, nullptr);
    umfpack_di_free_numeric(&Numeric);

    if (status != UMFPACK_OK) {
        std::cerr << "Solver step failed." << std::endl;
        return 1;
    }

    std::cout << "Solution vector x:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }

    return 0;
}
