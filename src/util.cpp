#include "spaND.h"

using namespace Eigen;
using namespace std;

namespace spaND {

bool are_connected(VectorXi &a, VectorXi &b, SpMat &A) {
    int64_t  bsize = b.size();
    auto b_begin = b.data();
    auto b_end   = b.data() + b.size();
    for(int64_t ia = 0; ia < a.size(); ia++) {
        // Look at all the neighbors of ia
        int64_t node = a[ia];
        for(SpMat::InnerIterator it(A,node); it; ++it) {
            auto neigh = it.row();
            auto id = lower_bound(b_begin, b_end, neigh);
            int64_t pos = id - b_begin;
            if(pos < bsize && b[pos] == neigh) // Found one in b! They are connected.
                return true;
        }
    }
    return false;
}

// lvl=0=leaf
// assumes a ND binary tree
bool should_be_disconnected(int64_t lvl1, int64_t lvl2, int64_t sep1, int64_t sep2) {
    while (lvl2 > lvl1) {
        lvl1 += 1;
        sep1 /= 2;
    }
    while (lvl1 > lvl2) {
        lvl2 += 1;
        sep2 /= 2;
    }
    if (sep1 != sep2) {
        return true;
    } else {
        return false;
    }
}

/** 
 * Given A, returns |A|+|A^T|+I
 */
SpMat symmetric_graph(SpMat& A) {
    assert(A.rows() == A.cols());
    int64_t n = A.rows();
    vector<Triplet<double>> vals(2 * A.nonZeros() + n);
    int64_t l = 0;
    for (int64_t k=0; k < A.outerSize(); ++k) {
        vals[l++] = Triplet<double>(k, k, 1.0);
        for (SpMat::InnerIterator it(A,k); it; ++it) {
            vals[l++] = Triplet<double>(it.col(), it.row(), abs(it.value()));
            vals[l++] = Triplet<double>(it.row(), it.col(), abs(it.value()));
        }
    }
    assert(l == vals.size());
    SpMat AAT(n,n);
    AAT.setFromTriplets(vals.begin(), vals.end());
    return AAT;
}

double elapsed(timeval& start, timeval& end) {
    return (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);
}

timer wctime() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time;
}

// All are base-0
void swap2perm(Eigen::VectorXi* swap, Eigen::VectorXi* perm) {
    int64_t n = perm->size();
    assert(swap->size() == n);
    for(int64_t i = 0; i < n; i++) {
        (*perm)[i] = i;
    }
    for(int64_t i = 0; i < n; i++) {
        int64_t ipiv = (*swap)[i];
        int64_t tmp = (*perm)[ipiv];
        (*perm)[ipiv] = (*perm)[i];
        (*perm)[i] = tmp;
    }
}

bool isperm(const Eigen::VectorXi* perm) {
    int64_t n = perm->size();
    VectorXi count = VectorXi::Zero(n);
    for(int64_t i = 0;i < n; i++) {
        int64_t pi = (*perm)[i];
        if(pi < 0 || pi >= n) { return false; }
        count[pi] += 1;
    }
    return (count.cwiseEqual(1)).all();
}

Eigen::VectorXi invperm(const Eigen::VectorXi& perm) {
    assert(isperm(&perm));
    Eigen::VectorXi invperm(perm.size());
    for(int64_t i = 0; i < perm.size(); i++) {
        invperm[perm[i]] = i;
    }
    assert(isperm(&invperm));
    return invperm;    
}

size_t hashv(vector<size_t> vals) {
    size_t seed = 0;
    for (size_t i = 0; i < vals.size(); ++i) {
      seed ^= vals[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

/**
 * C = alpha A^(/T) * B^(/T) + beta C
 */
void gemm_spand(MatrixXd* A, MatrixXd* B, MatrixXd* C, Op tA, Op tB, double alpha, double beta) {
    int64_t m = C->rows();
    int64_t n = C->cols();    
    int64_t lda = A->rows();
    int64_t ldb = B->rows();
    int64_t ldc = C->rows();
    int64_t k  = (tA == Op::NoTrans ? A->cols() : A->rows());
    int64_t k2 = (tB == Op::NoTrans ? B->rows() : B->cols());        
    int64_t m2 = (tA == Op::NoTrans ? A->rows() : A->cols());
    int64_t n2 = (tB == Op::NoTrans ? B->cols() : B->rows());
    assert(k == k2);
    assert(m == m2); 
    assert(n == n2);
    if(m == 0 || n == 0 || k == 0)
        return;
    blas::gemm(Layout::ColMajor, tA, tB, m, n, k, alpha, A->data(), lda, B->data(), ldb, beta, C->data(), ldc);
}

MatrixXd* gemm_new(Eigen::MatrixXd* A, Eigen::MatrixXd* B, Op tA, Op tB, double alpha) {
    int64_t m = (tA == Op::NoTrans ? A->rows() : A->cols());
    int64_t n = (tB == Op::NoTrans ? B->cols() : B->rows());
    MatrixXd* C = new MatrixXd(m, n);
    gemm_spand(A, B, C, tA, tB, alpha, 0.0);
    return C;
}

void syrk(MatrixXd* A, MatrixXd* C, Op tA, double alpha, double beta) {
    int64_t n = C->rows();
    int64_t k = (tA == Op::NoTrans ? A->cols() : A->rows());
    assert(C->cols() == n);
    if (n == 0 || k == 0)
        return;
    int64_t lda = A->rows();
    blas::syrk(Layout::ColMajor, Uplo::Lower, tA, n, k, alpha, A->data(), lda, beta, C->data(), n);
}

int64_t potf(MatrixXd* A) {
    int64_t n = A->rows();
    assert(A->cols() == n);
    if (n == 0)
        return 0;
    int64_t info = lapack::potrf(Uplo::Lower n, A->data(), n);
    return info;
}

int64_t ldlt(Eigen::MatrixXd* A, Eigen::MatrixXd* L, Eigen::VectorXd* d, Eigen::VectorXi* p, double* rcond) {
    if(A->rows() == 0) return 0;
    LDLT<Ref<MatrixXd>, Lower> ldlt(*A);
    if(ldlt.info() != ComputationInfo::Success) {
        return 1;
    }
    (*rcond) = ldlt.rcond();
    VectorXd halfD = ldlt.vectorD().cwiseAbs().cwiseSqrt();
    *d = ldlt.vectorD().cwiseSign();
    (*L) = ldlt.matrixL();
    (*L) = (*L) * halfD.asDiagonal();
    VectorXi swap = ldlt.transpositionsP().indices();
    swap2perm(&swap, p); 
    return 0;
}

int64_t getf(Eigen::MatrixXd* A, Eigen::VectorXi* p) {
    int64_t n = A->rows();
    assert(A->cols() == n);
    assert(p->size() == n);
    VectorXi swap(p->size());
    if(n == 0)
        return 0;
    int64_t info = lapack::getrf(n, n, A->data(), n, swap.data());
    if(info != 0) {
        return info;
    } else {
        for(int64_t i = 0; i < n; i++) {
            (swap)[i] -= 1;
        }
        swap2perm(&swap, p);
        return info;
    }
}

void fpgetf(Eigen::MatrixXd* A, Eigen::VectorXi* p, Eigen::VectorXi* q) {
    assert(A->cols() == A->rows());
    assert(p->size() == A->cols());
    assert(q->size() == A->cols());
    if(A->rows() == 0) return;
    Eigen::FullPivLU<Eigen::Ref<Eigen::MatrixXd>> pluq(*A);
    *p = invperm(pluq.permutationP().indices());
    *q = invperm(pluq.permutationQ().indices());
}

// A = L * (D + U') = L * D (I + D^-1 U') = { L * |D|^1/2 } { |D|^1/2 sign(D) (I + D^-1 U') }
void split_LU(Eigen::MatrixXd* A, Eigen::MatrixXd* L, Eigen::MatrixXd* U) {
    assert(A->rows() == A->cols());
    assert(L->rows() == L->cols());
    assert(U->rows() == U->cols());
    assert(A->rows() == L->rows());
    assert(A->rows() == U->rows());
    VectorXd d = A->diagonal();
    *L = A->triangularView<StrictlyLower>();            
    *U = A->triangularView<StrictlyUpper>();
    *U = d.cwiseInverse().asDiagonal() * (*U);
    L->diagonal() = VectorXd::Ones(A->rows());
    U->diagonal() = VectorXd::Ones(A->rows());
    *L = (*L) * d.cwiseAbs().cwiseSqrt().asDiagonal();
    *U = (d.cwiseSign().cwiseProduct(d.cwiseAbs().cwiseSqrt())).asDiagonal() * (*U); 
}

double rcond_1_getf(Eigen::MatrixXd* A_LU, double A_1_norm) {
    int64_t n = A_LU->rows();
    assert(A_LU->cols() == n);
    double rcond = 10.0;
    int64_t info = lapack::gecon(Norm::One, n, A_LU->data(), n, A_1_norm, &rcond);
    assert(info == 0);
    return rcond;
}

double rcond_1_potf(Eigen::MatrixXd* A_LLT, double A_1_norm) {
    int64_t n = A_LLT->rows();
    assert(A_LLT->cols() == n);
    double rcond = 10.0;
    int64_t info = lapack::pocon(Uplo::Lower, n, A_LLT->data(), n, A_1_norm, &rcond);
    assert(info == 0);
    return rcond;
}

double rcond_1_trcon(Eigen::MatrixXd* LU, Uplo uplo, Diag diag) {
    assert(LU->rows() == LU->cols());
    if(LU->rows() == 0) return 1.0;
    double rcond;
    int64_t info = lapack::trcon(Norm::One, uplo, diag, LU->rows(), LU->data(), LU->rows(), &rcond);
    assert(info == 0);
    return rcond;
}

void trsm_right(MatrixXd* L, MatrixXd* B, Uplo uplo, Op trans, Diag diag) {
    int64_t m = B->rows();
    int64_t n = B->cols();
    assert(L->rows() == n);
    assert(L->cols() == n);
    if (m == 0 || n == 0)
        return;
    blas::trsm(Layout::ColMajor, Side::Right, uplo, trans, diag, m, n, 1.0, L->data(), n, B->data(), m);
}

void trsm_left(MatrixXd* L, MatrixXd* B, Uplo uplo, Op trans, Diag diag) {
    int64_t m = B->rows();
    int64_t n = B->cols();
    assert(L->rows() == m);
    assert(L->cols() == m);
    if (m == 0 || n == 0)
        return;
    blas::trsm(Layout::ColMajor, Side::Left, uplo, trans, diag, m, n, 1.0, L->data(), m, B->data(), m);
}

void trsv(MatrixXd* LU, Segment* x, Uplo uplo, Op trans, Diag diag) {
    int64_t n = LU->rows();
    assert(LU->cols() == n);
    assert(x->size() == n);
    if (n == 0)
        return;
    blas::trsv(Layout::ColMajor, uplo, trans, diag, n, LU->data(), n, x->data(), 1);
}

void trmv_trans(MatrixXd* L, Segment* x) {
    int64_t n = L->rows();
    int64_t m = L->cols();
    assert(x->size() == n);
    assert(n == m);
    if (n == 0)
        return;
    blas::trmv(Layout::ColMajor, Uplo::Lower, Op::Trans, Diag::NonUnit, L->rows(), L->data(), L->rows(), x->data(), 1);
}

// A <- L^T * A
void trmm_trans(MatrixXd* L, MatrixXd* A) {
    int64_t m = A->rows();
    int64_t n = A->cols();
    assert(L->rows() == m);
    assert(L->cols() == m);
    if (m == 0 || n == 0)
        return;
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Lower, Op::Trans, Diag::NonUnit, m, n, 1.0, L->data(), m, A->data(), m);
}

// x2 -= A21 * x1
void gemv_notrans(MatrixXd* A21, Segment* x1, Segment* x2) {
    int64_t m = A21->rows();
    int64_t n = A21->cols();
    assert(x1->size() == n);
    assert(x2->size() == m);
    if (n == 0 || m == 0)
        return;
    blas::gemv(Layout::ColMajor, Op::NoTrans, m, n, -1.0, A21->data(), m, x1->data(), 1, 1.0, x2->data(), 1);
}

// x2 -= A12^T x1
void gemv_trans(MatrixXd* A12, Segment* x1, Segment* x2) {
    int64_t m = A12->rows();
    int64_t n = A12->cols();
    assert(x1->size() == m);
    assert(x2->size() == n);
    if (n == 0 || m == 0)
        return;
    blas::gemv(Layout::ColMajor, Op::Trans, m, n, -1.0, A12->data(), m, x1->data(), 1, 1.0, x2->data(), 1);
}

// x <- Q * x
void ormqr_notrans(MatrixXd* v, VectorXd* h, Segment* x) {
    int64_t m = v->rows();
    int64_t n = v->cols();
    assert(h->size() == n);
    assert(x->size() == m);
    if (m == 0) 
        return;
    int64_t info = lapack::ormqr(Side::Left, Op::NoTrans, m, 1, n, v->data(), m, h->data(), x->data(), m); 
    assert(info == 0);
}

// x <- Q^T * x
void ormqr_trans(MatrixXd* v, VectorXd* h, Segment* x) {
    int64_t m = x->size();
    // n = 1
    int64_t k = v->cols();
    assert(h->size() == k);
    assert(v->rows() == m);
    if (m == 0 || k == 0) 
        return;
    int64_t info = lapack::ormqr(Side::Left, Op::Trans, m, 1, k, v->data(), m, h->data(), x->data(), m); 
    assert(info == 0);
}

// A <- (Q^/T) * A * (Q^/T)
void ormqr(MatrixXd* v, VectorXd* h, MatrixXd* A, Side side, Op trans) {
    int64_t m = A->rows();
    int64_t n = A->cols();
    int64_t k = v->cols(); // number of reflectors
    assert(h->size() == k);
    if (m == 0 || n == 0)
        return;
    if(side == Side::Left) // Q * A or Q^T * A
        assert(k <= m);
    if(side == Side::Right) // A * Q or A * Q^T
        assert(k <= n);
    if (k == 0)
        return; // No reflectors, so nothing to do
    int64_t info = lapack::ormqr(side, trans, m, n, k, v->data(), v->rows(), h->data(), A->data(), m);
    assert(info == 0);
}

// Create the thin Q in v
void orgqr(Eigen::MatrixXd* v, Eigen::VectorXd* h) {
    int64_t m = v->rows();
    int64_t k = v->cols();
    assert(h->size() == k);
    if(m == 0 || k == 0)
        return;
    int64_t info = lapack::orgqr(m, k, k, v->data(), m, h->data());
    assert(info == 0);
}

// RRQR
void geqp3(MatrixXd* A, VectorXi* jpvt, VectorXd* tau) {
    int64_t m = A->rows();
    int64_t n = A->cols();
    if (m == 0 || n == 0)
        return;
    assert(jpvt->size() == n);
    assert(tau->size() == min(m,n));
    int64_t info = lapack::geqp3(m, n, A->data(), m, jpvt->data(), tau->data());
    assert(info == 0);
    for (int64_t i = 0; i < jpvt->size(); i++)
        (*jpvt)[i] --;
}

// Full SVD
void gesvd(Eigen::MatrixXd* A, Eigen::MatrixXd* U, Eigen::VectorXd* S, Eigen::MatrixXd* VT) {
    int64_t m = A->rows();
    int64_t n = A->cols();
    int64_t k = min(m,n);
    assert(U->rows() == m && U->cols() == m);
    assert(VT->rows() == n && VT->cols() == n);
    assert(S->size() == k);
    if(k == 0)
        return;
    VectorXd superb(k-1);
    int64_t info = lapack::gesvd(Job::AllVec, Job::AllVec, m, n, A->data(), m, S->data(), U->data(), m, VT->data(), n, superb.data());
    assert(info == 0);
}

// Full Symmetric EVD
void syev(Eigen::MatrixXd* A, Eigen::VectorXd* S) {
    int64_t m = A->rows();
    int64_t n = A->cols();
    assert(m == n);
    assert(S->size() == m);
    if(m == 0)
        return;
    int64_t info = lapack::syev(Job::Vec, Uplo::Lower, m, A->data(), m, S->data());
    assert(info == 0);
}

// QR
void geqrf(MatrixXd* A, VectorXd* tau) {
    int64_t m = A->rows();
    int64_t n = A->cols();
    if (m == 0 || n == 0)
        return;
    assert(tau->size() == min(m,n));
    int64_t info = lapack::geqrf(m, n, A->data(), m, tau->data());
    assert(info == 0);
}

int64_t choose_rank(VectorXd& s, double tol) {
    if (tol == 0) {
        return s.size();
    } else if (tol >= 1.0) {
        return 0;
    } else {
        if (s.size() <= 1) {
            return s.size();
        } else {
            double sref = abs(s[0]);
            int64_t rank = 1;
            while(rank < s.size() && abs(s[rank]) / sref >= tol) {
                rank++;
            }
            assert(rank <= s.size());
            return rank;
        }
    }
}

void block2dense(VectorXi &rowval, VectorXi &colptr, VectorXd &nnzval, int64_t i, int64_t j, int64_t li, int64_t lj, MatrixXd *dst, bool transpose) {
    if(transpose) {
        assert(dst->rows() == lj && dst->cols() == li);
    } else {
        assert(dst->rows() == li && dst->cols() == lj);
    }
    for(int64_t col = 0; col < lj; col++) {
        // All elements in column c
        int64_t start_c = colptr[j + col];
        int64_t end_c = colptr[j + col + 1];
        int64_t size = end_c - start_c;
        auto start = rowval.data() + start_c;
        auto end = rowval.data() + end_c;
        // Find i
        auto found = lower_bound(start, end, i);
        int64_t id = distance(start, found);
        // While we are between i and i+i...
        while(id < size) {
            int64_t row = rowval[start_c + id];
            if(row >= i + li) {
                break;
            }
            row = row - i;
            double val = nnzval[start_c + id];
            if(transpose) {
                (*dst)(col,row) = val;
            } else {
                (*dst)(row,col) = val;
            }
            id ++;
        }
    }
}

MatrixXd linspace_nd(int64_t n, int64_t dim) {
    MatrixXd X = MatrixXd::Zero(dim, pow(n, dim));
    if (dim == 1) {
        for(int64_t i = 0; i < n; i++) {
            X(0,i) = double(i);
        }
    } else if (dim == 2) {
        int64_t id = 0;
        for(int64_t i = 0; i < n; i++) {
            for(int64_t j = 0; j < n; j++) {
                X(0,id) = i;
                X(1,id) = j;
                id ++;
            }
        }
    } else if (dim == 3) {
        int64_t id = 0;
        for(int64_t i = 0; i < n; i++) {
            for(int64_t j = 0; j < n; j++) {
                for(int64_t k = 0; k < n; k++) {
                    X(0,id) = i;
                    X(1,id) = j;
                    X(2,id) = k;
                    id ++;
                }
            }
        }
    }
    return X;
}

// Compute A[p,p]
SpMat symm_perm(SpMat &A, VectorXi &p) {
    // Create inverse permutation
    VectorXi pinv(p.size());
    for(int64_t i = 0; i < p.size(); i++)
        pinv[p[i]] = i;
    // Initialize A[p,p]
    int64_t n = A.rows();
    int64_t nnz = A.nonZeros();
    assert(n == A.cols()); 
    SpMat App(n, n);
    App.reserve(nnz);
    // Create permuted (I, J, V) values
    vector<Triplet<double>> vals(nnz);
    int64_t l = 0;
    for (int64_t k = 0; k < A.outerSize(); k++){
        for (SpMat::InnerIterator it(A, k); it; ++it){
            int64_t i = it.row();
            int64_t j = it.col();
            double v = it.value();
            vals[l] = Triplet<double>(pinv[i],pinv[j],v);
            l ++;
        }
    }
    // Create App
    App.setFromTriplets(vals.begin(), vals.end());
    return App;
}

// Random [-1,1]
VectorXd random(int64_t size, int64_t seed) {
    mt19937 rng;
    rng.seed(seed);
    uniform_real_distribution<double> dist(-1.0,1.0);
    VectorXd x(size);
    for(int64_t i = 0;i < size; i++) {
        x[i] = dist(rng);
    }
    return x;
}

MatrixXd random(int64_t rows, int64_t cols, int64_t seed) {
    mt19937 rng;
    rng.seed(seed);
    uniform_real_distribution<double> dist(-1.0,1.0);
    MatrixXd A(rows, cols);
    for(int64_t i = 0; i < rows; i++) {
        for(int64_t j = 0; j < cols; j++) {
            A(i,j) = dist(rng);
        }
    }
    return A;
}


void setZero(Eigen::MatrixXd* A) {
#if 0
    A->setZero();
#else
    // Probably not super portable. Should be on all systems implementing IEC 60559 (or IEEE 754-1985)
    memset(A->data(), 0, A->size() * sizeof(double));   // Is ~2x faster than Eigen's setZero()
    if(A->size() > 0) assert((*A)(A->size()-1) == 0.0); // Fails if double 0 are not sizeof(double) '0' bytes
#endif
}

}
