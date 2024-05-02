#ifndef IS_H
#define IS_H

#include <iostream>
#include <stdio.h>
#include <Eigen/Core>

#include "spaND.h"

namespace spaND {

int64_t cg(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int64_t iters, double tol, bool verb);
int64_t gmres(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int64_t iters, int64_t restart, double tol_error, bool verb);
int64_t ir(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int64_t iters, double tol, bool verb);

}

#endif
