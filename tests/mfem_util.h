Eigen::VectorXd mfem2eigen(mfem::Vector &b) {
    int64_t N = b.Size();
    Eigen::VectorXd x(N);
    for(int64_t i = 0; i < N; i++) {
        x[i] = b(i);
    }
    return x;
}

void eigen2mfem(Eigen::VectorXd &a, mfem::Vector &b) {
    int64_t N = b.Size();
    assert(a.size() == N);
    for(int64_t i = 0; i < N; i++) {
        b(i) = a[i];
    }
}

spaND::SpMat mfem2eigen(mfem::SparseMatrix &A) {
    int64_t N = A.Height();
    int64_t M = A.Width();
    int64_t nnz = A.NumNonZeroElems();
    std::vector<Eigen::Triplet<double>> triplets(nnz);
    int64_t l = 0;
    for(int64_t i = 0; i < A.Height(); i++) {
       for(int64_t k = A.GetI()[i]; k < A.GetI()[i+1]; k++) {
           assert(l < nnz);
           int64_t j = A.GetJ()[k];
           double v = A.GetData()[k];
           assert(i >= 0 && i < N && j >= 0 && j < M);
           triplets[l] = Eigen::Triplet<double>(i,j,v);
           l += 1;
       }
    }
    assert(l == nnz);
    spaND::SpMat B(N,M);
    B.setFromTriplets(triplets.begin(), triplets.end());
    return B;
}
