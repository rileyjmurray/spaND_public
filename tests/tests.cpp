#include <gtest/gtest.h>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <random>
#include "spaND.h"
#include "mmio.hpp"
#include "cxxopts.hpp"
#include <Eigen/SparseCholesky>

using namespace std;
using namespace Eigen;
using namespace spaND;

bool VERB = false;
int64_t  N_THREADS = 4;
int64_t  RUN_MANY = 4;

SymmKind symm2syk(int64_t symm) {
    switch(symm) {
        case 0: return SymmKind::SPD;
        case 1: return SymmKind::SYM;
        case 2: return SymmKind::GEN;
        default: assert(false);
    };
    return SymmKind::SPD;
}

PartKind pki2pk(int64_t pki) {
    switch(pki) {
        case 0: return PartKind::MND;
        case 1: return PartKind::RB;
        default: assert(false);
    };
    return PartKind::MND;
}

ScalingKind ski2sk(int64_t ski) {
    switch(ski) {
        case 0: return ScalingKind::LLT;
        case 1: return ScalingKind::EVD;
        case 2: return ScalingKind::SVD;
        case 3: return ScalingKind::PLU;
        case 4: return ScalingKind::PLUQ;
        case 5: return ScalingKind::LDLT;
        default: assert(false);
    };
    return ScalingKind::LLT;
}

bool is_valid(SymmKind syk, ScalingKind sk, bool preserve) {    
    if (syk == SymmKind::SPD) {
        if (sk != ScalingKind::LLT) return false;
    }
    if (syk == SymmKind::SYM) {
        if (sk != ScalingKind::LDLT) return false;
    }
    if (syk == SymmKind::GEN) {
        if (sk != ScalingKind::PLU && sk != ScalingKind::PLUQ) return false;
    }
    if (preserve) return false;
    return true;
}

struct params {
    SymmKind syk;
    PartKind pk;
    ScalingKind sk;
    bool preserve;
};

vector<params> get_params() {
    vector<params> configs;
    for (int64_t symm = 0; symm < 3; symm++) {
        for (int64_t pki = 0; pki < 2; pki++) {
            for (int64_t ski = 0; ski < 6; ski++) {
                for (int64_t pres = 0; pres < 2; pres++) {
                    PartKind pk = pki2pk(pki);
                    ScalingKind sk = ski2sk(ski); 
                    SymmKind syk = symm2syk(symm);
                    if (! is_valid(syk, sk, pres)) continue;
                    configs.push_back({syk, pk, sk, pres == 1});
                }
            }
        }
    }
    return configs;
};

SpMat neglapl(int64_t n, int64_t d) {
    stringstream s;
    s << "../mats/neglapl_" << d << "_" << n << ".mm";
    string file = s.str();
    SpMat A = mmio::sp_mmread<double,int64_t>(file);
    return A;
}

SpMat neglapl_unsym(int64_t n, int64_t d, int64_t seed) {
    SpMat A = neglapl(n, d);
    default_random_engine gen;
    gen.seed(seed);
    std::uniform_real_distribution<double> rand(-0.1, 0.1);
    for (int64_t k = 0; k < A.outerSize(); ++k) {
        for (SpMat::InnerIterator it(A, k); it; ++it) {
            A.coeffRef(it.row(), it.col()) += rand(gen);
        }
    }
    return A;
}

SpMat make_indef(SpMat& A, int64_t seed) {
    Eigen::SimplicialLLT<SpMat, Eigen::Lower> sllt(A);
    VectorXd random_diagonal = random(A.rows(), seed);        
    for (int64_t i = 0; i < A.rows(); i++) {
        if (random_diagonal[i] <= 0.9) {
            random_diagonal[i] = -1;
        } else {
            random_diagonal[i] = 1;
        }
    }
    SpMat L = sllt.matrixL();
    return L * (random_diagonal.asDiagonal() * L.transpose());
}

SpMat random_SpMat(int64_t n, double p, int64_t seed) {
    default_random_engine gen;
    gen.seed(seed);
    std::uniform_real_distribution<double> dist(0.0,1.0);
    vector<Triplet<double>> triplets;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            auto v_ij = dist(gen);
            if (v_ij < p) {
                triplets.push_back(Triplet<double>(i,j,v_ij));
            }
        }
    }
    SpMat A(n,n);
    A.setFromTriplets(triplets.begin(), triplets.end()); 
    return A;
}

SpMat identity_SpMat(int64_t n) {
    vector<Triplet<double>> triplets;
    for (int64_t i = 0; i < n; ++i) {
        triplets.push_back(Triplet<double>(i,i,1.0));
    }
    SpMat A(n,n);
    A.setFromTriplets(triplets.begin(), triplets.end()); 
    return A;
}

TEST(MatrixMarket, Sparse) {
    // 1
    SpMat A = mmio::sp_mmread<double,int64_t>("../mats/test1.mm");
    SpMat Aref(2, 3);
    Aref.insert(0, 0) = 1;
    Aref.insert(0, 1) = -2e2;
    Aref.insert(1, 1) = 3e3;
    Aref.insert(1, 2) = -4.4e4;
    EXPECT_EQ(A.nonZeros(), 4);
    EXPECT_EQ((Aref - A).norm(), 0.0);
    // 2
    A = mmio::sp_mmread<double,int64_t>("../mats/test2.mm");
    Aref = SpMat(3, 3);
    Aref.insert(0, 0) = 1.1;
    Aref.insert(1, 1) = 2e2;
    Aref.insert(2, 0) = -3.3;
    Aref.insert(0, 2) = -3.3;
    EXPECT_EQ(A.nonZeros(), 4);
    EXPECT_EQ((Aref - A).norm(), 0.0);
    // 3
    A = mmio::sp_mmread<double,int64_t>("../mats/test3.mm");
    Aref = SpMat(4, 1);
    Aref.insert(3, 0) = -1;
    EXPECT_EQ(A.nonZeros(), 1);
    EXPECT_EQ((Aref - A).norm(), 0.0);
    // 4
    A = mmio::sp_mmread<double,int64_t>("../mats/test4.mm");
    Aref = SpMat(2, 2);
    Aref.insert(1, 0) = -3.3;
    Aref.insert(0, 1) = -3.3;
    EXPECT_EQ(A.nonZeros(), 2);
    EXPECT_EQ((Aref - A).norm(), 0.0);
}

TEST(MatrixMarket, Array) {
    // 5
    MatrixXd A = mmio::dense_mmread<double>("../mats/test5.mm");
    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    MatrixXd Aref(2, 3);
    Aref << 1, 3, -5, 2, 4, 1e6; // row-wise filling in eigen
    EXPECT_EQ((Aref - A).norm(), 0.0);
    // 6
    A = mmio::dense_mmread<double>("../mats/test6.mm");
    Aref = MatrixXd(2, 2);
    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 2);
    Aref << 1, -2, -2, 3; // row-wise filling in eigen
    EXPECT_EQ((Aref - A).norm(), 0.0);
}

/** Util.cpp tests **/

TEST(Util, AreConnected) {
    // 3x3 laplacian
    SpMat A = mmio::sp_mmread<double,int64_t>("../mats/neglapl_2_3.mm");
    VectorXi64 a(2);
    VectorXi64 b(3);
    a << 0, 1;
    b << 6, 7, 8;
    EXPECT_FALSE(are_connected(a, b, A));
    a = VectorXi64(2);
    b = VectorXi64(3);
    a << 0, 1;
    b << 2, 5, 8;
    EXPECT_TRUE(are_connected(a, b, A));
    a = VectorXi64(2);
    b = VectorXi64(1);
    a << 3, 4;
    b << 5;
    EXPECT_TRUE(are_connected(a, b, A));
    a = VectorXi64(1);
    b = VectorXi64(1);
    a << 6;
    b << 6;
    EXPECT_TRUE(are_connected(a, b, A));
}

TEST(Util, ShouldBeDisconnected) {
    EXPECT_TRUE(should_be_disconnected(0, 0, 0, 2));
    EXPECT_TRUE(should_be_disconnected(0, 0, 1, 2));
    EXPECT_TRUE(should_be_disconnected(0, 0, 4, 2));
    EXPECT_TRUE(should_be_disconnected(0, 0, 5, 2));
    EXPECT_TRUE(should_be_disconnected(0, 0, 1000, 2));

    EXPECT_TRUE(should_be_disconnected(1, 2, 2, 0));
    EXPECT_TRUE(should_be_disconnected(0, 1, 1, 1));
    EXPECT_TRUE(should_be_disconnected(0, 2, 1, 1));
    EXPECT_TRUE(should_be_disconnected(2, 0, 0, 5));
    EXPECT_TRUE(should_be_disconnected(2, 2, 0, 1));

    EXPECT_FALSE(should_be_disconnected(0, 1, 0, 0));
    EXPECT_FALSE(should_be_disconnected(0, 2, 0, 0));
    EXPECT_FALSE(should_be_disconnected(0, 10, 0, 0));

    EXPECT_FALSE(should_be_disconnected(2, 0, 1, 5));
    EXPECT_FALSE(should_be_disconnected(2, 0, 1, 6));
    EXPECT_FALSE(should_be_disconnected(2, 1, 1, 2));
    EXPECT_FALSE(should_be_disconnected(2, 1, 1, 3));
}

TEST(Util, ChooseRank) {
    VectorXd errs = VectorXd(5);
    errs << 1.0, -0.1, 0.01, -0.001, 1e-4;
    EXPECT_EQ(choose_rank(errs, 1e-1), 2);
    EXPECT_EQ(choose_rank(errs, 1e-2), 3);
    EXPECT_EQ(choose_rank(errs, 1.0), 0);
    EXPECT_EQ(choose_rank(errs, 0), 5);
    EXPECT_EQ(choose_rank(errs, 1e-16), 5);
}

TEST(Util, Block2Dense) {
    // Usual case
    {
        SpMat A(5, 5);
        A.insert(0, 0) = 1.0;
        A.insert(2, 2) = -2.0;
        A.insert(1, 3) = 3.0;
        A.makeCompressed();
        VectorXi64 rowval = Map<VectorXi64>(A.innerIndexPtr(), A.nonZeros());
        VectorXi64 colptr = Map<VectorXi64>(A.outerIndexPtr(), 6);
        VectorXd nnzval = Map<VectorXd>(A.valuePtr(), A.nonZeros());
        MatrixXd Ad = MatrixXd::Zero(3, 3);
        block2dense(rowval, colptr, nnzval, 1, 1, 3, 3, &Ad, false);
        MatrixXd Adref = MatrixXd::Zero(3, 3);
        Adref << 0, 0, 3, 0, -2, 0, 0, 0, 0;
        EXPECT_EQ((Adref - Ad).norm(), 0);
    }
    // Transpose
    {
        SpMat A(3, 4);
        A.insert(0, 1) = 1.0;
        A.insert(1, 0) = 2.0;
        A.insert(1, 2) = 3.0;
        A.insert(2, 1) = 4.0;
        A.insert(0, 3) = 5.0;
        A.makeCompressed();
        VectorXi64 rowval = Map<VectorXi64>(A.innerIndexPtr(), A.nonZeros());
        VectorXi64 colptr = Map<VectorXi64>(A.outerIndexPtr(), 5);
        VectorXd nnzval = Map<VectorXd>(A.valuePtr(), A.nonZeros());
        MatrixXd Ad = MatrixXd::Zero(3, 2);
        block2dense(rowval, colptr, nnzval, 0, 0, 2, 3, &Ad, true);
        MatrixXd Adref = MatrixXd::Zero(3, 2);
        Adref << 0, 2, 1, 0, 0, 3;
        EXPECT_EQ((Adref - Ad).norm(), 0);
    }
}

TEST(Util, LinspaceNd) {
    MatrixXd X2 = linspace_nd(3, 2);
    MatrixXd X2ref(2, 9);
    X2ref << 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2;
    EXPECT_EQ((X2ref - X2).norm(), 0);
    MatrixXd X3 = linspace_nd(2, 3);
    MatrixXd X3ref(3, 8);
    X3ref << 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1;
    EXPECT_EQ((X3ref - X3).norm(), 0);
}

TEST(Util, SymmPerm) {
    vector<int64_t> dims  = {2, 2,  2,  3, 3,  3,  3 };
    vector<int64_t> sizes = {5, 10, 20, 5, 15, 25, 30};
    for (int64_t test = 0; test < dims.size(); test++) {
        int64_t s = sizes[test];
        int64_t d = dims[test];
        stringstream ss;
        ss << "../mats/neglapl_" << d << "_" << s << ".mm";
        SpMat A = mmio::sp_mmread<double,int64_t>(ss.str());
        // Create random perm
        int64_t N = A.rows();
        VectorXi64 p = VectorXi64::LinSpaced(N, 0, N-1);
        random_device rd;
        std::mt19937 g(rd());
        shuffle(p.data(), p.data() + N, g);
        // Compare
        SpMat pAp = symm_perm(A, p);
        SpMat pApref = p.asPermutation().inverse() * A * p.asPermutation();
        EXPECT_EQ((pAp - pApref).norm(), 0.0);
    }
}

TEST(Util, isperm) {
    VectorXi64 perm1(10);
    VectorXi64 perm2(10);
    VectorXi64 noperm1(10);
    VectorXi64 noperm2(5);
    perm1 << 0, 9, 8, 1, 4, 2, 3, 7, 5, 6;
    perm2 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    noperm1 << 0, 9, 8, 1, 4, 2, 3, 5, 5, 6;
    noperm2 << 0, 9, 8, 1, 4;
    EXPECT_TRUE(isperm(&perm1));
    EXPECT_TRUE(isperm(&perm2));
    EXPECT_FALSE(isperm(&noperm1));
    EXPECT_FALSE(isperm(&noperm2));
}

TEST(Util, swap2perm) {
    VectorXi64 swap(6);
    VectorXi64 perm(6);
    VectorXi64 permRef(6);
    swap << 3, 3, 2, 5, 4, 5;
    permRef << 3, 0, 2, 5, 4, 1;
    swap2perm(&swap, &perm); // perm.asPermutation().transpose() * x <=> x[perm]
    EXPECT_EQ((perm - permRef).norm(), 0.0);
}

SpMat symmetric_graph_ref(SpMat A) {
    SpMat ATabs = A.cwiseAbs().transpose();
    return A.cwiseAbs() + ATabs + identity_SpMat(A.rows());
}

TEST(Util, symmetric_graph) {
    for (int64_t i = 1; i < 100; i++) {
        SpMat A = random_SpMat(i, 0.2, i);
        SpMat AAT = symmetric_graph(A);
        EXPECT_LT( (AAT - symmetric_graph_ref(A)).norm(), 1e-12);
    }
}


/** Partitioning tests **/

/**
 * Check the partitioning of a square laplacian 5x5
 */
TEST(PartitionTest, Square) {
    SpMat A = mmio::sp_mmread<double,int64_t>("../mats/neglapl_2_5.mm");
    MatrixXd X = linspace_nd(5, 2);
    Tree t(3);
    t.set_verb(VERB);
    t.set_use_geo(true);
    t.set_Xcoo(&X);
    auto part = t.partition(A);
    vector<SepID> sepidref { 
        SepID(0,0), SepID(0,0), SepID(1,0), SepID(0,1), SepID(0,1),
        SepID(0,0), SepID(0,0), SepID(1,0), SepID(0,1), SepID(0,1),
        SepID(2,0), SepID(2,0), SepID(2,0), SepID(2,0), SepID(2,0),
        SepID(0,2), SepID(0,2), SepID(1,1), SepID(0,3), SepID(0,3),
        SepID(0,2), SepID(0,2), SepID(1,1), SepID(0,3), SepID(0,3), 
    } ;
    vector<SepID> leftref { 
        SepID(0,0), SepID(0,0), SepID(0,0), SepID(0,1), SepID(0,1),
        SepID(0,0), SepID(0,0), SepID(0,0), SepID(0,1), SepID(0,1),
        SepID(0,0), SepID(0,0), SepID(1,0), SepID(0,1), SepID(0,1),
        SepID(0,2), SepID(0,2), SepID(0,2), SepID(0,3), SepID(0,3),
        SepID(0,2), SepID(0,2), SepID(0,2), SepID(0,3), SepID(0,3), 
    } ;
    vector<SepID> rightref { 
        SepID(0,0), SepID(0,0), SepID(0,1), SepID(0,1), SepID(0,1),
        SepID(0,0), SepID(0,0), SepID(0,1), SepID(0,1), SepID(0,1),
        SepID(0,2), SepID(0,2), SepID(1,1), SepID(0,3), SepID(0,3),
        SepID(0,2), SepID(0,2), SepID(0,3), SepID(0,3), SepID(0,3),
        SepID(0,2), SepID(0,2), SepID(0,3), SepID(0,3), SepID(0,3), 
    } ;
    for (int64_t i = 0; i < part.size(); i++) {
        ASSERT_TRUE(part[i].self  == sepidref[i]);
        ASSERT_TRUE(part[i].l     == leftref[i]);
        ASSERT_TRUE(part[i].r     == rightref[i]);
    }
}

/**
 * Check consistency of the partitioning
 */
TEST(PartitionTest, Consistency) {
    vector<int64_t> dims  = {2, 2,  2,   3, 3,  3 };
    vector<int64_t> sizes = {5, 20, 100, 5, 15, 25};
    for (int64_t test = 0; test < dims.size(); test++) {
        int64_t s = sizes[test];
        int64_t d = dims[test];
        stringstream ss;
        ss << "../mats/neglapl_" << d << "_" << s << ".mm";
        int64_t n = pow(s, d);
        string file = ss.str();
        for (int64_t nlevels = 1; nlevels < 8; nlevels++) {
            for (int64_t geoi = 0; geoi < 2; geoi++) {
                for (int64_t pki = 0; pki < 2; pki++) {
                    bool geo = (geoi == 0);
                    PartKind pk = pki == 0 ? PartKind::MND : PartKind::RB;
                    // Partition tree
                    MatrixXd X = linspace_nd(s, d);
                    Tree t(nlevels);
                    t.set_verb(VERB);
                    SpMat A = mmio::sp_mmread<double,int64_t>(file);
                    t.set_use_geo(geo);
                    t.set_Xcoo(&X);
                    t.set_part_kind(pk);
                    auto part = t.partition(A);                    
                    // (1) Lengths
                    ASSERT_EQ(part.size(), n);
                    // (2) Check ordering integrity
                    for (int64_t i = 0; i < n; i++) {
                        auto pi = part[i].self;
                        for (SpMat::InnerIterator it(A,i); it; ++it) {
                            int64_t j = it.row();
                            auto pj = part[j].self;  
                            ASSERT_FALSE(should_be_disconnected(pi.lvl, pj.lvl, pi.sep, pj.sep));
                        }
                    }
                    // (3) Check left/right integrity          
                    for (int64_t i = 0; i < n; i++) {
                        auto pi = part[i].self;
                        auto li = part[i].l;
                        auto ri = part[i].r;
                        if (pi.lvl == 0) {
                            ASSERT_TRUE(pi == li);
                            ASSERT_TRUE(pi == ri);
                        } else {
                            ASSERT_TRUE(pi.lvl > li.lvl);
                            ASSERT_TRUE(pi.lvl > ri.lvl);
                            while (li.lvl < pi.lvl - 1) {
                                li.lvl += 1;
                                li.sep /= 2;
                            }
                            while (ri.lvl < pi.lvl - 1) {
                                ri.lvl += 1;
                                ri.sep /= 2;
                            }
                            ASSERT_TRUE(li.lvl == pi.lvl - 1);
                            ASSERT_TRUE(ri.lvl == pi.lvl - 1);
                            ASSERT_TRUE(li.sep == 2 * pi.sep);
                            ASSERT_TRUE(ri.sep == 2 * pi.sep + 1);
                        }
                    }
                }
            }
        }
    }
}

/** Assembly tests **/

/** 
 * Check assembly
 */
TEST(Assembly, Consistency) {
    vector<int64_t> dims  = {2, 2,  2,  3, 3,  3};
    vector<int64_t> sizes = {5, 10, 20, 5, 10, 15};
    for (int64_t spandlorasp = 0; spandlorasp < 2; spandlorasp++) {
        for (int64_t test = 0; test < dims.size(); test++) { 
            for (int64_t pki = 0; pki < 2; pki++) {
                PartKind pk = pki == 0 ? PartKind::MND : PartKind::RB;       
                int64_t s = sizes[test];
                int64_t d = dims[test];
                SpMat Aref = neglapl(s, d);
                SpMat Arefunsym = neglapl_unsym(s, d, test);
                for (int64_t nlevels = 2; nlevels < 5 ; nlevels++) {
                    /**
                     * Symmetric case
                     */
                    {
                        // Partition and assemble
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.set_use_geo(false);
                        t.set_part_kind(pk);
                        if (spandlorasp == 0) t.partition(Aref);
                        else                 t.partition_lorasp(Aref);
                        t.assemble(Aref);
                        // Get permutation
                        VectorXi64 p = t.get_assembly_perm();
                        // Check it's indeed a permutation
                        ASSERT_TRUE(isperm(&p));
                        auto P = p.asPermutation();
                        // Check get_mat()
                        SpMat A2 = t.get_trailing_mat();
                        EXPECT_EQ((P.inverse() * Aref * P - A2).norm(), 0.0);
                    }
                    /**
                     * Unsymmetric case
                     */
                    {
                        // Partition and assemble
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.set_symm_kind(SymmKind::GEN);
                        t.set_use_geo(false);
                        t.set_part_kind(pk);                        
                        if (spandlorasp == 0) t.partition(Arefunsym);
                        else                 t.partition_lorasp(Arefunsym);
                        t.assemble(Arefunsym);                
                        // Get permutation
                        VectorXi64 p = t.get_assembly_perm();
                        // Check it's indeed a permutation
                        ASSERT_TRUE(isperm(&p));
                        auto P = p.asPermutation();
                        // Check get_mat()
                        SpMat A2 = t.get_trailing_mat();
                        EXPECT_EQ((P.inverse() * Arefunsym * P - A2).norm(), 0.0);
                    }
                }
            }
        }
    }
}

/** Factorization tests **/

TEST(ApproxTest, PrintConfigs) {
    vector<params> configs = get_params();
    cout << "Preserve ? PartKind ? ScalingKind ? SymmKind ?" << endl;
    for (auto c: configs) {
        cout << c.preserve << " " << part2str(c.pk) << " " << scaling2str(c.sk) << " " << symm2str(c.syk) << endl;
    }
}

/**
 * Test that with eps=0, we get exact solutions
 */
TEST(ApproxTest, Exact) {
    vector<int64_t> dims  = {2, 2,  2,  3, 3};
    vector<int64_t> sizes = {5, 10, 20, 5, 15};
    vector<double> tols = {1e-14, 1e-14, 0.0};
    vector<int64_t> skips   = {0,     4,     1000};
    vector<params> configs = get_params();
    for (int64_t test = 0; test < dims.size(); test++) {
        cout << "Test " << test << "... ";
        int64_t count = 0;
        int64_t s = sizes[test];
        int64_t d = dims[test];
        int64_t n = pow(s, d);
        int64_t nlevelsmin = n < 1000 ? 1 : 8;
        SpMat Aref = neglapl(s, d);
        SpMat Arefunsym = neglapl_unsym(s, d, test);
        SpMat Arefsym = make_indef(Aref, 2019+test);
        for (int64_t nlevels = nlevelsmin; nlevels < nlevelsmin+5 ; nlevels++) {
            for (auto c: configs) {
                SpMat A = (c.syk == SymmKind::SPD ? Aref : (c.syk == SymmKind::SYM ? Arefsym : Arefunsym));
                assert(! c.preserve);
                MatrixXd phi = random(Aref.rows(), 3, test+nlevels+2019);
                for (int64_t it = 0; it < tols.size(); it++) {
                    double tol = tols[it];
                    double skip = skips[it];
                    Tree t(nlevels);
                    t.set_verb(VERB);
                    t.set_part_kind(c.pk);
                    t.set_scaling_kind(c.sk);
                    t.set_symm_kind(c.syk);                                
                    t.partition(A);
                    t.assemble(A);
                    t.set_tol(tol);
                    t.set_skip(skip);
                    t.set_preserve(c.preserve);
                    if (c.preserve) t.set_phi(&phi);
                    t.factorize();
                    VectorXd b = random(n, test+nlevels+2019+1);
                    auto x = b;
                    t.solve(x);
                    double err = (A*x-b).norm() / b.norm();
                    EXPECT_LE(err, 1e-10) << err;
                    count++;
                }
            }
        }
        cout << count << " tested.\n";
    }
}

/**
 * Test SPD on A (laplacian) and SYM+LDLT on -A (- laplacian) give the same, with or without compression
 */
TEST(ApproxTest, SPD_vs_LDLT) {
    vector<int64_t> dims  = {2,  3, 3};
    vector<int64_t> sizes = {128, 5, 15};
    vector<double> tols = {0,   1e-4, 1e-14};
    vector<int64_t> skips   = {100, 1,    0};
    vector<params> configs = get_params();
    for (int64_t test = 0; test < dims.size(); test++) {
        cout << "Test " << test << "... ";
        int64_t count = 0;
        int64_t s = sizes[test];
        int64_t d = dims[test];
        int64_t n = pow(s, d);
        int64_t nlevelsmin = n < 1000 ? 1 : 8;
        SpMat A = neglapl(s, d);
        SpMat Aneg = -A;
        for (int64_t nlevels = nlevelsmin; nlevels < nlevelsmin+5 ; nlevels++) {
            for (auto c: configs) {
                if (c.sk != ScalingKind::LLT && !c.preserve) continue;
                for (int64_t it = 0; it < tols.size(); it++) {
                    double tol = tols[it];
                    double skip = skips[it];
                    VectorXd b = random(n, test+nlevels+2019+1);
                    // Use LLT on A
                    Tree t_llt(nlevels);
                    t_llt.set_verb(VERB);
                    t_llt.set_part_kind(c.pk);
                    t_llt.set_scaling_kind(ScalingKind::LLT);
                    t_llt.set_symm_kind(SymmKind::SPD);                                
                    t_llt.partition(A);
                    t_llt.assemble(A);
                    t_llt.set_tol(tol);
                    t_llt.set_skip(skip);
                    t_llt.set_preserve(false);
                    t_llt.factorize();                    
                    VectorXd x_llt = b;
                    t_llt.solve(x_llt);
                    // Use LDLT on -A
                    Tree t_ldlt(nlevels);
                    t_ldlt.set_verb(VERB);
                    t_ldlt.set_part_kind(c.pk);
                    t_ldlt.set_scaling_kind(ScalingKind::LDLT);
                    t_ldlt.set_symm_kind(SymmKind::SYM);                                
                    t_ldlt.partition(Aneg);
                    t_ldlt.assemble(Aneg);
                    t_ldlt.set_tol(tol);
                    t_ldlt.set_skip(skip);
                    t_ldlt.set_preserve(false);
                    t_ldlt.factorize();                    
                    VectorXd x_ldlt = - b;
                    t_ldlt.solve(x_ldlt);
                    // Compare                    
                    double err_llt = (A*x_llt-b).norm() / b.norm();
                    double err_ldlt = (A*x_ldlt-b).norm() / b.norm();
                    double diff = (x_llt - x_ldlt).norm() / x_llt.norm();
                    if (tol == 0.0) {
                        EXPECT_LE(err_llt, 1e-12);
                        EXPECT_LE(err_ldlt, 1e-12);  
                        EXPECT_LE(diff, 1e-12);                      
                    } else {
                        EXPECT_LE(err_llt, tol * 1e2);
                        EXPECT_LE(err_ldlt, tol * 1e2);
                        EXPECT_LE(diff, tol * 1e2);
                    }                    
                    count++;
                }
            }
        }
        cout << count << " tested.\n";
    }
}

/** 
 * Test conservation is correct
 */
TEST(ApproxTest, Preservation) {
    vector<int64_t> dims  = {2, 2,  2,  3, 3,  3};
    vector<int64_t> sizes = {5, 10, 20, 5, 10, 25};
    for (int64_t test = 0; test < dims.size(); test++) {
        cout << "Test " << test;
        int64_t s = sizes[test];
        int64_t d = dims[test];
        stringstream ss;
        ss << "../mats/neglapl_" << d << "_" << s << ".mm";
        int64_t n = pow(s, d);
        string file = ss.str();
        SpMat A_spd = mmio::sp_mmread<double,int64_t>(file);        
        SpMat A_sym = - A_spd;
        int64_t nlevelsmin = n < 1000 ? 1 : 8;
        vector<double> tols = {10, 1e-2, 1e-3, 1e-4, 1e-6, 0.0};
        for (int64_t nlevels = nlevelsmin; nlevels < nlevelsmin + 5; nlevels++) {
            for (int64_t it = 0; it < tols.size(); it++) {
                for (int64_t skip = 0; skip < 3; skip++) {
                    for (int64_t symm = 0; symm < 2; symm++) {
                        printf("."); fflush(stdout);
                        SpMat A;
                        if (symm == 0) A = A_spd;
                        else          A = A_sym;
                        // Check a 1 is preserved
                        {
                            Tree t(nlevels);
                            if (symm == 0) {
                                t.set_scaling_kind(ScalingKind::LLT);
                                t.set_symm_kind(SymmKind::SPD);
                            } else {
                                t.set_scaling_kind(ScalingKind::LDLT);
                                t.set_symm_kind(SymmKind::SYM);
                            }
                            t.set_verb(VERB);
                            t.partition(A);
                            t.assemble(A);
                            MatrixXd phi = MatrixXd::Ones(n, 1);
                            t.set_tol(tols[it]);
                            t.set_skip(skip);
                            t.set_preserve(true);
                            t.set_phi(&phi);                            
                            t.factorize();
                            for (int64_t c = 0; c < phi.cols(); c++) {
                                VectorXd b = A * phi.col(c);
                                VectorXd x = b;
                                t.solve(x);
                                double err1 = (A*x-b).norm() / b.norm();
                                double err2 = (x-phi.col(c)).norm() / phi.col(c).norm();
                                EXPECT_TRUE(err1 < 1e-12) << "err1 = " << err1 << " | " << skip << " " << it << " " << nlevels << " " << test << endl;
                                EXPECT_TRUE(err2 < 1e-12) << "err2 = " << err2 << " | " << skip << " " << it << " " << nlevels << " " << test << endl;
                            }
                            VectorXd b = random(n, nlevels+it+skip+2019);
                            auto x = b;
                            t.solve(x);
                            double err = (A*x-b).norm() / b.norm();
                            if (tols[it] == 0.0) {
                                EXPECT_TRUE(err < 1e-12);
                            } else {
                                EXPECT_TRUE(err < tols[it] * 1e2);
                            }
                        }
                        // Check that a multiple random b are preserved
                        {
                            Tree t(nlevels);
                            if (symm == 0) {
                                t.set_scaling_kind(ScalingKind::LLT);
                                t.set_symm_kind(SymmKind::SPD);
                            } else {
                                t.set_scaling_kind(ScalingKind::LDLT);
                                t.set_symm_kind(SymmKind::SYM);
                            }
                            t.set_verb(VERB);
                            t.partition(A);
                            t.assemble(A);
                            MatrixXd phi = random(n, 5, nlevels+it+skip+2019);
                            t.set_tol(tols[it]);
                            t.set_skip(skip);
                            t.set_preserve(true);
                            t.set_phi(&phi);                            
                            t.factorize();
                            for (int64_t c = 0; c < phi.cols(); c++) {
                                VectorXd b = A * phi.col(c);
                                VectorXd x = b;
                                t.solve(x);
                                double err1 = (A*x-b).norm() / b.norm();
                                double err2 = (x-phi.col(c)).norm() / phi.col(c).norm();
                                EXPECT_TRUE(err1 < 1e-12) << "err1 = " << err1 << " | " << skip << " " << it << " " << nlevels << " " << test << endl;
                                EXPECT_TRUE(err2 < 1e-12) << "err2 = " << err2 << " | " << skip << " " << it << " " << nlevels << " " << test << endl;
                            }
                            VectorXd b = random(n, nlevels+it+skip+2019);
                            auto x = b;
                            t.solve(x);
                            double err = (A*x-b).norm() / b.norm();
                            if (tols[it] == 0.0) {
                                EXPECT_TRUE(err < 1e-12);
                            } else {
                                EXPECT_TRUE(err < tols[it] * 1e2);
                            }
                        }
                    }
                }
            }
        }
        printf("\n");
    }
}

/**
 * Test that the approximations are reasonnable accurate 
 * with and without preservation
 */
TEST(ApproxTest, Approx) {
    vector<int64_t> dims  = {2, 2,  2,  2,   3, 3};
    vector<int64_t> sizes = {5, 10, 20, 128, 5, 15};
    vector<double> tols = {0.0, 1e-10, 1e-6, 1e-2, 10};
    matrix_hash<VectorXd> hash;
    vector<params> configs = get_params();
    for (int64_t test = 0; test < dims.size(); test++) {
        vector<size_t> allhashes;
        int64_t count = 0;
        cout << "Test " << test << "... ";
        int64_t s = sizes[test];
        int64_t d = dims[test];
        SpMat Aref = neglapl(s, d);
        SpMat Arefunsym = neglapl_unsym(s, d, test+2019);
        SpMat Arefsym = - Aref;
        int64_t n = pow(s, d);
        int64_t nlevelsmin = n < 1000 ? 1 : 8;
        for (int64_t nlevels = nlevelsmin; nlevels < nlevelsmin + 5; nlevels++) {
            for (int64_t it = 0; it < tols.size(); it++) {
                for (int64_t skip = 0; skip < 3; skip++) {
                    for (auto c: configs) {
                        SpMat A = (c.syk == SymmKind::SPD ? Aref : (c.syk == SymmKind::SYM ? Arefsym : Arefunsym));
                        assert(! c.preserve);
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.set_symm_kind(c.syk);
                        t.set_part_kind(c.pk);
                        t.set_scaling_kind(c.sk);
                        t.partition(A);
                        t.assemble(A);
                        MatrixXd phi = random(A.rows(), 2, nlevels+it+skip+2019);
                        t.set_tol(tols[it]);
                        t.set_skip(skip);
                        t.set_preserve(c.preserve);
                        if (c.preserve) t.set_phi(&phi);
                        t.factorize();
                        VectorXd b = random(n, nlevels+it+skip+2019);
                        auto x = b;
                        t.solve(x);
                        double err = (A*x-b).norm() / b.norm();
                        auto hb = hash(b);
                        auto hx = hash(x);
                        allhashes.push_back(hb);
                        allhashes.push_back(hx);
                        if (tols[it] == 0.0) {
                            EXPECT_LE(err, 5e-12);
                        } else {
                            EXPECT_LE(err, tols[it] * 2e2);
                        }
                        count++;
                    }
                }
            }
        }
        size_t h = hashv(allhashes);
        cout << count << " tested. Overall hash(x,b) = " << h << endl;
    }
}

TEST(ApproxTest, ApproxLoRaSp) {
    vector<int64_t> dims  = {2, 2,  2,  2,   3, 3};
    vector<int64_t> sizes = {5, 10, 20, 64,  5, 15};
    vector<double> tols = {0.0, 1e-10, 1e-6, 1e-4};
    matrix_hash<VectorXd> hash;
    vector<params> configs = get_params();
    for (int64_t test = 0; test < dims.size(); test++) {
        vector<size_t> allhashes;
        int64_t count = 0;
        cout << "Test " << test << "... ";
        int64_t s = sizes[test];
        int64_t d = dims[test];
        SpMat Aref = neglapl(s, d);
        SpMat Arefunsym = neglapl_unsym(s, d, test+2019);
        int64_t n = pow(s, d);
        int64_t nlevelsmin = n < 1000 ? 1 : 8;
        for (int64_t nlevels = nlevelsmin; nlevels < nlevelsmin + 5; nlevels++) {
            for (int64_t it = 0; it < tols.size(); it++) {
                for (auto c: configs) {
                    SpMat A = (c.syk == SymmKind::SPD ? Aref : (c.syk == SymmKind::SYM ? (-Aref) : Arefunsym));
                    Tree t(nlevels);
                    t.set_verb(VERB);
                    t.set_symm_kind(c.syk);                    
                    t.set_scaling_kind(c.sk);                                                                    
                    t.partition_lorasp(A);
                    t.assemble(A);                        
                    MatrixXd phi = random(A.rows(), 2, nlevels+it+2019);
                    t.set_tol(tols[it]);
                    try {
                        t.factorize_lorasp();
                        VectorXd b = random(n, nlevels+it+2019);
                        auto x = b;
                        t.solve(x);
                        double err = (A*x-b).norm() / b.norm();
                        auto hb = hash(b);
                        auto hx = hash(x);
                        allhashes.push_back(hb);
                        allhashes.push_back(hx);
                        if (tols[it] == 0.0) {
                            EXPECT_LE(err, 5e-12);
                        } else {
                            EXPECT_LE(err, tols[it] * 2e2);
                        } 
                    } catch (exception& ex) {
                        cout << ex.what();
                        EXPECT_TRUE(false);
                    }                    
                    count++;
                }
            }
        }
        size_t h = hashv(allhashes);
        cout << count << " tested. Overall hash(x,b) = " << h << endl;
    }
}

/** 
 * Test that the code produce reproducable results
 */
TEST(ApproxTest, Repro) {
    int64_t    dims[3]      = {2, 2, 2};
    int64_t    sizes[3]     = {20, 64, 16};
    double tols[4]      = {1e-5, 10, 1e-8, 0.1};
    double skips[4]     = {1, 2, 0, 1};
    int64_t    repeat       = 10;
    vector<params> configs = get_params();
    for (int64_t test = 0; test < 3; test++) {
        printf("Tests "); fflush(stdout);
        int64_t count = 0;
        int64_t s = sizes[test];
        int64_t d = dims[test];
        int64_t n = pow(s, d);
        SpMat Aref = neglapl(s, d);
        SpMat Arefunsym = neglapl_unsym(s, d, test);        
        for (int64_t nlevels = 5; nlevels < 7; nlevels++) {
            for (int64_t pr = 0; pr < 6; pr++) {
                for (auto c: configs) {
                    for (int64_t lrsp = 0; lrsp < 2; lrsp++) {
                        printf("."); fflush(stdout);
                        SpMat A = (c.syk == SymmKind::SPD ? Aref : (c.syk == SymmKind::SYM ? (-Aref) : Arefunsym));
                        MatrixXd phi = random(A.rows(), 3, test+nlevels+pr+2019);
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.set_symm_kind(c.syk);
                        t.set_part_kind(c.pk);
                        t.set_scaling_kind(c.sk);
                        if (lrsp == 0) {
                            t.partition(A);
                        } else {
                            t.partition_lorasp(A);
                        }
                        t.assemble(A);
                        t.set_tol(tols[pr]);
                        t.set_skip(skips[pr]);
                        t.set_preserve(c.preserve);
                        if (c.preserve) t.set_phi(&phi);
                        if (lrsp == 0) {
                            t.factorize();
                        } else {
                            t.factorize_lorasp();
                        }
                        VectorXd b = random(n, nlevels+test);
                        auto xref = b;
                        t.solve(xref);
                        count++;
                        for (int64_t i = 0; i < repeat; i++) {
                            Tree t2(nlevels);
                            t2.set_verb(VERB);                        
                            t2.set_symm_kind(c.syk);
                            t2.set_part_kind(c.pk);
                            t2.set_scaling_kind(c.sk);
                            if (lrsp == 0) {
                                t2.partition(A);
                            } else {
                                t2.partition_lorasp(A);
                            }
                            t2.assemble(A);
                            t2.set_tol(tols[pr]);
                            t2.set_skip(skips[pr]);
                            t2.set_preserve(c.preserve);
                            t2.set_phi(&phi);
                            if (lrsp == 0) {
                                t2.factorize();
                            } else {
                                t2.factorize_lorasp();
                            }
                            auto x = b;
                            t2.solve(x);
                            EXPECT_EQ((xref - x).norm(), 0.0);
                        }
                    }
                }
            }
        }
        printf(": %d tested.\n", count); fflush(stdout);
    }
}

TEST(Run, Many) {
    vector<int64_t>    dims  = {2,  2,  2,   3, 3,  3 };
    vector<int64_t>    sizes = {5,  16, 64,  5, 10, 15};
    vector<double> tols  = {0.0, 1e-2, 1.0, 10.0};
    RUN_MANY = RUN_MANY > dims.size() ? dims.size() : RUN_MANY;
    matrix_hash<VectorXd> hash;
    vector<size_t> allhashes;
    vector<params> configs = get_params();
    for (int64_t test = 0; test < RUN_MANY; test++) {
        cout << "Run " << test << "... \n";
        int64_t count = 0;
        int64_t n = sizes[test];
        int64_t d = dims[test];
        SpMat Aref = neglapl(n, d);
        SpMat Arefunsym = neglapl_unsym(n, d, test);        
        int64_t N = Aref.rows();
        int64_t nlevelsmin = N < 1000 ? 1 : 8;
        for (int64_t nlevels = nlevelsmin; nlevels < nlevelsmin+3; nlevels++) {
            for (double tol : tols) {
                for (int64_t skip = 0; skip < 3; skip++) {
                    for (int64_t geo = 0; geo < 2; geo++) {
                        for (auto c: configs) {
                            SpMat A = (c.syk == SymmKind::SPD ? Aref : (c.syk == SymmKind::SYM ? (-Aref) : Arefunsym));
                            MatrixXd phi = random(A.rows(), 3, test+nlevels+2019);
                            MatrixXd X = linspace_nd(n, d);
                            Tree t(nlevels);
                            t.set_verb(VERB);                        
                            t.set_symm_kind(c.syk);
                            t.set_part_kind(c.pk);
                            t.set_scaling_kind(c.sk);
                            t.set_use_geo(geo);
                            t.set_Xcoo(&X);
                            t.set_tol(tol);
                            t.set_skip(skip);
                            t.set_preserve(c.preserve);
                            if (c.preserve) t.set_phi(&phi);
                            t.partition(A);
                            t.assemble(A);
                            t.factorize();
                            VectorXd b = random(N, nlevels+test);
                            auto x = b;
                            t.solve(x);                            
                            double res = (A*x-b).norm() / b.norm();
                            auto h = hash(x);
                            allhashes.push_back(h);
                            printf("%6d %4d %d] %3d %3.2e %d %d %d %d %d %d %d %d | %3.2e | %lu\n", 
                                     N, n, d,   nlevels, 
                                                     tol, skip, 
                                                              geo,
                                                                c.preserve, 
                                                                   int64_t(c.syk), int64_t(c.pk), int64_t(c.sk), 1, 1,
                                                                   res, h);
                            count++;
                        }
                    }
                }
            }
        }
        cout << "Ran " << count << " tests\n";
        size_t h = hashv(allhashes);
        cout << "Overall hash so far: " << h << endl;
    }
    size_t h = hashv(allhashes);
    cout << "Overall hash: " << h << endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    cxxopts::Options options("spaND tests", "Test suite for the spaND algorithms.");
    options.add_options()
        ("help", "Print help")
        ("v,verb", "Verbose (default: false)", cxxopts::value<bool>()->default_value("false"))
        ("n_threads", "Number of threads", cxxopts::value<int64_t>()->default_value("4"))
        ("run", "How many Run.Many to run", cxxopts::value<int64_t>()->default_value("4"))
        ;
    int64_t argc64 = argc;
    auto result = options.parse(argc64, argv);

    if (result.count("help"))
    {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    VERB = result["verb"].as<bool>();
    N_THREADS = result["n_threads"].as<int64_t>();
    RUN_MANY = result["run"].as<int64_t>();
    cout << "n_threads: " << N_THREADS << endl;
    cout << "verb: " << VERB << endl;
    cout << "run: " << RUN_MANY << endl;

    return RUN_ALL_TESTS();
}
