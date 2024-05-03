#ifndef IS_H
#define IS_H

#include <iostream>
#include <stdio.h>
#include <Eigen/Core>

#include "spaND.h"

namespace spaND {

using namespace Eigen;

template <typename EIGMAT>
int64_t cg(const EIGMAT& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int64_t iters, double tol, bool verb) {
    using std::sqrt;
    using std::abs;
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> VectorType;
    
    double t_matvec = 0.0;
    double t_preco  = 0.0;
    timer start     = wctime();
    
    int64_t maxIters = iters;
    
    int64_t n = mat.cols();

    timer t00 = wctime();
    VectorType residual = rhs - mat * x; // r_0 = b - A x_0
    timer t01 = wctime();
    t_matvec += elapsed(t00, t01);

    double rhsNorm2 = rhs.squaredNorm();
    if (rhsNorm2 == 0) 
    {
        x.setZero();
        iters = 0;
        return iters;
    }
    double threshold = tol*tol*rhsNorm2;
    double residualNorm2 = residual.squaredNorm();
    if (residualNorm2 < threshold)
    {
        iters = 0;
        return iters;
    }
   
    VectorType p(n);
    p = residual;
    timer t02 = wctime();
    std::cout << "Applying the preconditioner ... " << std::endl;
    precond.solve(p);      // p_0 = M^-1 r_0
    timer t03 = wctime();
    auto t_el = elapsed(t02, t03);
    std::cout << "took " << t_el << " seconds." << std::endl;
    t_preco += t_el;

    VectorType z(n), tmp(n);
    double absNew = residual.dot(p);  // the square of the absolute value of r scaled by invM
    int64_t i = 0;
    while (i < maxIters) {
        timer t0 = wctime();
        tmp.noalias() = mat * p;                    // the bottleneck of the algorithm
        timer t1 = wctime();
        t_el = elapsed(t0, t1);
        t_matvec += elapsed(t0, t1);

        double alpha = absNew / p.dot(tmp);         // the amount we travel on dir
        x += alpha * p;                             // update solution
        residual -= alpha * tmp;                    // update residual
        
        residualNorm2 = residual.squaredNorm();
        if (verb) printf("%d: |Ax-b|/|b| = %3.2e <? %3.2e\n", i, sqrt(residualNorm2 / rhsNorm2), tol);
        if (residualNorm2 < threshold) {
            if (verb) printf("Converged!\n");
            break;
        }
     
        z = residual; 
        timer t2 = wctime();
        precond.solve(z);                           // approximately solve for "A z = residual"
        timer t3 = wctime();
        t_preco += elapsed(t2, t3);
        double absOld = absNew;
        absNew = residual.dot(z);                   // update the absolute value of r
        double beta = absNew / absOld;              // calculate the Gram-Schmidt value used to create the new search direction
        p = z + beta * p;                           // update search direction
        i++;
    }
    iters = i+1;
    if (verb) {
        timer stop = wctime();
        printf("# of iter:  %d\n", iters);
        printf("Total time: %3.2e s.\n", elapsed(start, stop));
        printf("  Matvec:   %3.2e s.\n", t_matvec);
        printf("  Precond:  %3.2e s.\n", t_preco);
    }
    return iters;
};

int64_t gmres(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int64_t iters, int64_t restart, double tol_error, bool verb);
int64_t ir(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int64_t iters, double tol, bool verb);

}

#endif
