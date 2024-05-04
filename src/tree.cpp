#include "spaND.h"
#include <map>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <iostream>
#include <algorithm>
#include <functional>


using namespace Eigen;

namespace spaND {

/**
 *  Export Tree data to text file
 */

void write_clustering(const Tree& t, const MatrixXd& X, std::string fn) {
    std::vector<int64_t> dof2ID = t.get_dof2ID();
    std::ofstream clustering_s{fn};
    clustering_s << t.get_N() << ";" << X.rows() << ";" << t.get_nlevels() << "\n";
    clustering_s << "X;id" << "\n";
    for (int64_t i = 0; i < t.get_N(); i++) {
        // Coordinates
        for (int64_t d = 0; d < X.rows(); d++) {
            clustering_s << " " << X(d,i);
        }
        // ID & lvl
        clustering_s << ";" << dof2ID[i] << "\n";
    }
    clustering_s.close();
}

void write_merging(const Tree& t, std::string fn) {
    std::ofstream merging_s(fn);
    merging_s << "child;parent" << "\n";
    for (auto n: t.get_clusters()) {
        if (n->parent() != nullptr) {
            merging_s << n->order() << ";" << n->parent()->order() << "\n";
        }
    }
    merging_s.close();
}

void write_clusters(const Tree& t, std::string fn) {
    std::ofstream clusters_s(fn);
    clusters_s << "id;lvl;mergeLvl;name" << "\n";
    for (auto n: t.get_clusters()) {
        clusters_s << n->order() << ";" << n->level() << ";" << n->merge_level() << ";" << n->get_name() << "\n";
    }
    clusters_s.close();
}

void write_stats(const Tree& t, std::string fn) {
    std::ofstream stats_s(fn);
    stats_s << "id;size;rank;Rdiag" << "\n";
    for (auto n: t.get_clusters()) {
        stats_s << n->order() << ";" << n->original_size() << ";" << n->size() << ";";
        for (auto v: n->Rdiag) { stats_s << v << " "; }
        stats_s << "\n";
    }
    stats_s.close();
}


void write_log_flops(const Tree& t, std::string fn) {         
    std::ofstream flops(fn);
    for (int64_t l = 0; l < t.tprof_flops.size(); l++) {
        for (const auto& f: t.tprof_flops[l].pivot) {
            flops << l << ";pivot;" << f.rows << ";0;0;" << f.time << "\n";
        }
        for (const auto& f: t.tprof_flops[l].panel) {
            flops << l << ";panel;" << f.rows << ";" << f.cols << ";0;" << f.time << "\n";
        }
        for (const auto& f: t.tprof_flops[l].gemm) {
            flops << l << ";gemm;" << f.rows << ";" << f.cols << ";" << f.inner << ";" << f.time << "\n";
        }
        for (const auto& f: t.tprof_flops[l].rrqr) {
            flops << l << ";rrqr;" << f.rows << ";" << f.cols << ";0;" << f.time << "\n";
        }
    }    
    flops.close();
}

/**
 * Tree
 */

/** 
 * Indicates default values for all parameters
 */
void Tree::init(int64_t nlevels) {
    assert(nlevels > 0);
    this->verb = true;
    this->geo = false;
    this->part_kind = PartKind::MND;
    this->preserve = false;
    this->ilvl = 0;
    this->nlevels = nlevels;
    this->tol = 10.0;
    this->skip = 0;
    this->stop = 0;
    this->scale_kind = ScalingKind::LLT;
    this->symm_kind = SymmKind::SPD;
    this->Xcoo = nullptr;
    this->phi = nullptr;
    this->use_want_sparsify = true;
    this->monitor_condition_pivots = false;
    this->monitor_unsymmetry = false;
    this->monitor_Rdiag = false;
    this->tprof = std::vector<Profile>(nlevels, Profile());
    this->tprof_flops = std::vector<ProfileFlops>(nlevels, ProfileFlops());
    this->monitor_flops = false;
    this->log   = std::vector<Log>(nlevels, Log());
    this->bottoms = std::vector<std::list<pCluster>>(nlevels);
    this->current_bottom = 0;
    this->max_order = 0;
}

Tree::Tree(int64_t nlevels) { this->init(nlevels); }
void Tree::set_verb(bool verb) { this->verb = verb; }
void Tree::set_Xcoo(MatrixXd* Xcoo) { this->Xcoo = Xcoo; }
void Tree::set_use_geo(bool geo) { this->geo = geo; }
void Tree::set_phi(MatrixXd* phi) { this->phi = phi; }
int64_t  Tree::nphis() const {
    return phi->cols();
}
void Tree::set_preserve(bool preserve) { this->preserve = preserve; }
void Tree::set_tol(double tol) { this->tol = tol; }
void Tree::set_skip(int64_t skip) { this->skip = skip; }
void Tree::set_stop(int64_t stop) { this->stop = stop; }
void Tree::set_scaling_kind(ScalingKind scaling_kind) { this->scale_kind = scaling_kind; }
void Tree::set_symm_kind(SymmKind symm_kind) { this->symm_kind = symm_kind; }
void Tree::set_part_kind(PartKind part_kind) { this->part_kind = part_kind; }
void Tree::set_use_sparsify(bool use) { this->use_want_sparsify = use; }
void Tree::set_monitor_condition_pivots(bool monitor) { this->monitor_condition_pivots = monitor; }
void Tree::set_monitor_flops(bool monitor) { this->monitor_flops = monitor; }
void Tree::set_monitor_unsymmetry(bool monitor) { this->monitor_unsymmetry = monitor; }
void Tree::set_monitor_Rdiag(bool monitor) { this->monitor_Rdiag = monitor; }
int64_t Tree::get_new_order() {
    int64_t o = max_order;
    this->max_order ++;
    return o;
}

int64_t Tree::nclusters_left() const {
    int64_t n = 0;
    for (auto& self : bottom_current()){
        if (! self->is_eliminated()) n++;
    }
    return n;
}

int64_t Tree::ndofs_left() const {
    int64_t n = 0;
    for (auto& self : bottom_current()){
        if (! self->is_eliminated()) n += self->size();
    }
    return n;
}

int64_t Tree::get_N() const {
    return this->perm.size();
}

int64_t Tree::get_last() const {
    int64_t last = get_N();
    for (int64_t lvl = 0; lvl < this->log.size(); lvl++) {
        if (this->log[lvl].dofs_left_elim > 0) {
            last = std::min(last, this->log[lvl].dofs_left_elim);
        }
        if (this->log[lvl].dofs_left_spars > 0) {
            last = std::min(last, this->log[lvl].dofs_left_spars);
        }
    }
    return last;
}

int64_t Tree::get_nlevels() const {
    return this->nlevels;
}

VectorXi64 Tree::get_assembly_perm() const {
    return this->perm;
}

const std::list<pCluster>& Tree::bottom_current() const {
    assert(current_bottom < bottoms.size());
    return bottoms[current_bottom];
}

const std::list<pCluster>& Tree::bottom_original() const {
    assert(bottoms.size() > 0);
    return bottoms[0];
}

bool Tree::symmetry() const {
    return this->symm_kind == SymmKind::SPD || this->symm_kind == SymmKind::SYM;
}

void Tree::assert_symmetry() {
    for (const auto& self : this->bottom_current()) {
        for (auto e: self->edgesOutNbr()) {
            assert(e->n1 == self.get());
            assert(e->n2->order() > e->n1->order());
        }
    }
}

/** 
 * Goes over all edges and collect stats on size and count
 */
void Tree::stats() const {
    Stats<int64_t> cluster_size = Stats<int64_t>();
    Stats<int64_t> edge_size = Stats<int64_t>();
    Stats<int64_t> edge_count = Stats<int64_t>();
    for (const auto& self : bottom_original()) {
        cluster_size.addData(self->size());
        edge_count.addData(self->nnbr_in_self_out());
        for (const auto edge : self->edgesOutAll()) {
            assert(edge->n1 == self.get());
            edge_size.addData(edge->n1->size() * edge->n2->size());
        }
        for (const auto edge : self->edgesInNbr()) {
            assert(edge->n2 == self.get());
            edge_size.addData(edge->n1->size() * edge->n2->size());
        }
    }
    printf("    Cluster size: %9d | %9d | %9d | %9f\n", cluster_size.getCount(), cluster_size.getMin(), cluster_size.getMax(), cluster_size.getMean());
    printf("    Edge sizes:   %9d | %9d | %9d | %9f\n", edge_size.getCount(),    edge_size.getMin(),    edge_size.getMax(),    edge_size.getMean());
    printf("    Edge count:   %9d | %9d | %9d | %9f\n", edge_count.getCount(),   edge_count.getMin(),   edge_count.getMax(),   edge_count.getMean());
}

void print_nodes_hierarchy(Cluster* n, int64_t indent) {        
    for (int64_t i = 0; i < indent; i++) std::cout << "  ";
    if (indent == 0) std::cout << "* ";
    if (indent >  0) std::cout << "|_";
    std::cout << n->order() << ": " << n->start() << ", " << n->size() << ", " << n->level() << ", " << n->get_name() << "\n";
    for (auto c: n->children()) {
        print_nodes_hierarchy(c, indent + 1);
    }
}

void Tree::print_clusters_hierarchy() const {
    // (1) Find all cluster hierarchy roots
    std::set<Cluster*> roots;
    for (int64_t l = 0; l < bottoms.size(); l++) {
        for (const auto& n: bottoms[l]) {
            if (n->parent() == nullptr) roots.insert(n.get());
        }
    }
    std::vector<Cluster*> roots_sorted(roots.begin(), roots.end());
    // (2) Print hierarchy
    for (auto r: roots_sorted) {
        print_nodes_hierarchy(r, 0);
    }
}

void Tree::print_connectivity() const {
    for (int64_t l = 0; l < bottoms.size(); l++) {
        for (const auto& n: bottoms[l]) {
            printf("Node %d (%p) size %d:\n", n->order(), n.get(), n->size());
            for (auto e: n->edgesOutAll()) {
                printf(" -> %d (%p) orig ? %d\n", e->n2->order(), e->n2, e->is_original());
            }
            for (auto e: n->edgesInNbr()) {
                printf(" <- %d (%p) orig ? %d\n", e->n1->order(), e->n1, e->is_original());
            }
        }
    }
}

std::list<const Cluster*> Tree::get_clusters() const {
    std::list<const Cluster*> all;
    for (int64_t l = 0; l < bottoms.size(); l++) {
        for (const auto& n: bottoms[l]) {
            all.push_back(n.get());
        }
    }
    return all;
}

std::vector<int64_t> Tree::get_dof2ID() const {
    std::vector<int64_t> ids(this->get_N());
    assert(bottoms.size() > 0);
    for (const auto& n: bottoms[0]) {
        for (int64_t i = 0; i < n->original_size(); i++) {
            ids[perm[n->start() + i]] = n->order();
        }
    }
    return ids;
}

// We sparsify a cluster when both his left and right have been eliminated
bool Tree::want_sparsify(Cluster* self) {
    assert(self->level() >= this->ilvl); // If we sparsify a node, it means it cannot be eliminated
    if (this->use_want_sparsify) {
        return self->should_sparsify();
    } else {
        return true;
    }
}

/**
 * Partition & Order
 * A is assumed to have a symmetric pattern. 
 * The diagonal is irrelevant.
 */

/** Takes a matrix A
 *  Creates clusters & parents suitable for assembly/factorize
 *  Returns an informative struct with the partition & info on the hierarchy
 */
std::vector<ClusterID> Tree::partition(SpMat &A) {
    assert(this->ilvl == 0);
    timer tstart = wctime();
    // Basic stuff
    int64_t N = A.rows();
    assert(A.rows() == A.cols());
    assert(nlevels > 0);
    if (this->geo) {
        assert(this->Xcoo != nullptr);
        assert(this->Xcoo->cols() == N);
    }
    std::vector<ClusterID> part;
    bool use_vertex_sep = true;
    // Print
    if (this->verb) {
        if (this->part_kind == PartKind::MND) {
            if (this->geo) std::cout << "MND geometric partitioning of matrix with " << N << " dofs with " << nlevels << " levels in " << this->Xcoo->rows() << "D" << std::endl;
            else           std::cout << "MND algebraic (with vertex sep ? " << use_vertex_sep << ") partitioning of matrix with " << N << " dofs with " << nlevels << " levels" << std::endl;
        } else if (this->part_kind == PartKind::RB) {
            if (this->geo) std::cout << "RB geometric partitioning of matrix with " << N << " dofs with " << nlevels << " levels in " << this->Xcoo->rows() << "D" << std::endl;
            else           std::cout << "RB algebraic partitioning of matrix with " << N << " dofs with " << nlevels << " levels" << std::endl;
        }
    }
    // Compute the self/left/right partitioning
    if (this->part_kind == PartKind::MND) {
        part = partition_modifiedND(A, this->nlevels, this->verb, use_vertex_sep, this->geo ? this->Xcoo : nullptr);
    } else if (this->part_kind == PartKind::RB) {
        part = partition_recursivebissect(A, this->nlevels, this->verb, this->geo ? this->Xcoo : nullptr);
    }
    // Logging
    for (int64_t i = 0; i < part.size(); i++) {
        this->log[part[i].self.lvl].dofs_nd += 1;
    }
    this->log[nlevels-1].dofs_left_nd = 0;
    for (int64_t l = nlevels-2; l >= 0; l--) {
        this->log[l].dofs_left_nd = this->log[l+1].dofs_nd + this->log[l+1].dofs_left_nd;
    }
    // Compute the ordering & associated permutation
    perm = VectorXi64::LinSpaced(N, 0, N-1);
    // Sort according to the ND ordering FIRST & the cluster merging process THEN
    std::vector<ClusterID> partmerged = part;
    auto compIJ = [&partmerged](int64_t i, int64_t j){return (partmerged[i] < partmerged[j]);};
    std::stable_sort(perm.data(), perm.data() + perm.size(), compIJ); // !!! STABLE SORT MATTERS !!!
    for (int64_t lvl = 1; lvl < nlevels; lvl++) {
        std::transform(partmerged.begin(), partmerged.end(), partmerged.begin(), [&lvl](ClusterID s){return merge_if(s, lvl);}); // lvl matters    
        std::stable_sort(perm.data(), perm.data() + perm.size(), compIJ); // !!! STABLE SORT MATTERS !!!
    }
    // Apply permutation
    std::vector<ClusterID> partpermed(N);
    std::transform(perm.data(), perm.data() + perm.size(), partpermed.begin(), [&](int64_t i){return part[i];});
    // Create the initial clusters
    std::map<Cluster*, ClusterID> clusters_ids;
    std::map<Cluster*, ClusterID> parents_ids;
    std::vector<Stats<int64_t>> clustersstats(nlevels, Stats<int64_t>());
    for (int64_t k = 0; k < N; ) {
        int64_t knext = k+1;
        ClusterID id = partpermed[k];
        while (knext < N && partpermed[knext] == id) { knext += 1; }
        int64_t size = knext - k;
        auto self = std::make_unique<Cluster>(k, size, id.self.lvl, get_new_order(), id.l.lvl == 0 && id.r.lvl == 0);
        clusters_ids[self.get()] = id;
        clustersstats[self->level()].addData(self->size());
        bottoms[0].push_back(std::move(self));
        k = knext;
    }
    if (this->verb) {
        printf("Clustering size statistics (# of leaf-clusters at each level of the ND hierarchy)\n");
        printf("Lvl     Count       Min       Max      Mean\n");
        for (int64_t lvl = 0; lvl < nlevels; lvl++) {
            printf("%3d %9d %9d %9d %9.0f\n", lvl, clustersstats[lvl].getCount(), clustersstats[lvl].getMin(), clustersstats[lvl].getMax(), clustersstats[lvl].getMean());
        }
    }
    // Create the cluster hierarchy
    if (this->verb) printf("Hierarchy numbers (# of cluster at each level of the cluster-hierarchy)\n");
    if (this->verb) printf("%3d %9lu\n", 0, bottoms[0].size());
    for (int64_t lvl = 1; lvl < nlevels; lvl++) {
        auto begin = std::find_if(bottoms[lvl-1].begin(), bottoms[lvl-1].end(), [lvl](const pCluster& s){
                        return s->level() >= lvl; // All others should have been eliminated by now
                     });
        auto end   = bottoms[lvl-1].end();        
        // Merge clusters        
        for (auto self = begin; self != end; self++) {
            assert((*self)->level() >= lvl);
            parents_ids[self->get()] = merge_if(clusters_ids.at(self->get()), lvl);            
        }
        // Figure out who gets merged together, setup children/parent, parentID
        for (auto k = begin; k != end;) {
            // Figures out who gets merged together
            auto idparent = parents_ids.at(k->get());
            std::vector<Cluster*> children;
            // Find all the guys that get merged with him
            children.push_back(k->get());
            int64_t children_start = (*k)->start();
            int64_t children_size  = (*k)->size();
            k++;
            while (k != end && idparent == parents_ids.at(k->get())) {
                children.push_back(k->get());
                children_size += (*k)->size();
                k++;                
            }
            auto parent = std::make_unique<Cluster>(children_start, children_size, idparent.self.lvl, get_new_order(), idparent.l.lvl == lvl && idparent.r.lvl == lvl);
            for (auto c: children) {
                c->set_parent(parent.get());
                parent->add_children(c);                
            }
            clusters_ids[parent.get()] = idparent;
            bottoms[lvl].push_back(std::move(parent));            
        }
        if (this->verb) printf("%3d %9lu\n", lvl, bottoms[lvl].size());
    }
    timer tend = wctime();
    if (this->verb) printf("Partitioning time : %3.2e s.\n", elapsed(tstart, tend));
    return part;
}

void Tree::partition_lorasp(SpMat& A) {
    assert(this->ilvl == 0);
    timer tstart = wctime();
    // Basic stuff
    int64_t N = A.rows();
    assert(A.rows() == A.cols());
    assert(nlevels > 0);
    if (this->geo) {
        assert(this->Xcoo != nullptr);
        assert(this->Xcoo->cols() == N);
    }
    // Print
    if (this->verb) {        
        if (this->geo) std::cout << "RB geometric partitioning of matrix with " << N << " dofs with " << nlevels << " levels in " << this->Xcoo->rows() << "D" << std::endl;
        else           std::cout << "RB algebraic partitioning of matrix with " << N << " dofs with " << nlevels << " levels" << std::endl;
    }    
    std::vector<int64_t> parts = partition_RB(A, nlevels, this->verb, this->Xcoo);
    std::vector<int64_t> partpermed = parts;
    // Compute the ordering & associated permutation
    perm = VectorXi64::LinSpaced(N, 0, N-1);
    // Sort according to parts
    auto compIJ = [&parts](int64_t i, int64_t j){return (parts[i] < parts[j]);};
    std::sort(perm.data(), perm.data() + perm.size(), compIJ);    
    // Apply permutation
    std::transform(perm.data(), perm.data() + perm.size(), partpermed.begin(), [&](int64_t i){return parts[i];});
    // Create the initial clusters
    std::map<Cluster*, int64_t> clusters_ids; // Basically the partition ID
    std::map<Cluster*, int64_t> parents_ids;
    int64_t lvltop = nlevels-1;
    for (int64_t k = 0; k < N; ) {
        int64_t knext = k+1;
        int64_t id = partpermed[k];
        while (knext < N && partpermed[knext] == id) { knext += 1; }
        int64_t size = knext - k;
        auto self = std::make_unique<Cluster>(k, size, lvltop, get_new_order(), true);
        clusters_ids[self.get()] = self->order();
        bottoms[0].push_back(std::move(self));
        k = knext;
    }    
    // Create the cluster hierarchy
    if (this->verb) printf("Hierarchy numbers (# of cluster at each level of the cluster-hierarchy)\n");
    if (this->verb) printf("%3d %9lu\n", 0, bottoms[0].size());
    for (int64_t lvl = 1; lvl < nlevels; lvl++) {
        auto begin = std::find_if(bottoms[lvl-1].begin(), bottoms[lvl-1].end(), [lvl](const pCluster& s){
                        return s->level() >= lvl; // All others should have been eliminated by now
                     });
        auto end = bottoms[lvl-1].end();
        // Merge clusters
        for (auto self = begin; self != end; self++) {
            assert((*self)->level() >= lvl);
            parents_ids[self->get()] = clusters_ids[self->get()] / 2;
        }
        // Figure out who gets merged together, setup children/parent, parentID
        for (auto k = begin; k != end;) {
            // Figures out who gets merged together
            auto idparent = parents_ids.at(k->get());
            std::vector<Cluster*> children;
            // Find all the guys that get merged with him
            children.push_back(k->get());
            int64_t children_start = (*k)->start();
            int64_t children_size  = (*k)->size();
            k++;
            while (k != end && idparent == parents_ids.at(k->get())) {
                children.push_back(k->get());
                children_size += (*k)->size();
                k++;                
            }
            auto parent = std::make_unique<Cluster>(children_start, children_size, lvltop, get_new_order(), true);
            for (auto c: children) {
                c->set_parent(parent.get());
                parent->add_children(c);                
            }
            clusters_ids[parent.get()] = idparent;
            bottoms[lvl].push_back(std::move(parent));            
        }
        if (this->verb) printf("%3d %9lu\n", lvl, bottoms[lvl].size());
    }
    timer tend = wctime();
    if (this->verb) printf("Partitioning time : %3.2e s.\n", elapsed(tstart, tend));
}

/** 
 * Assemble the matrix
 */
void Tree::assemble(SpMat& A) {
    assert(this->ilvl == 0);
    int64_t N = this->get_N();
    assert(N >= 0);
    timer tstart = wctime();        
    int64_t nlevels = this->nlevels;
    if (this->verb) std::cout << "Assembling (Size " << N << " with " << nlevels << " levels and symmetry " << this->symmetry() << ")" << std::endl;    
    // Verify the clusters hierarchy
    for (const auto& n: this->bottom_current()) {
        assert(n->depth() == n->level());
    }
    // Permute & compress the matrix for assembly
    timer t0 = wctime();
    SpMat App = symm_perm(A, perm);
    App.makeCompressed();
    timer t1 = wctime();
    // Get CSC format    
    int64_t nnz = App.nonZeros();
    VectorXi64 rowval = Eigen::Map<VectorXi64>(App.innerIndexPtr(), nnz);
    VectorXi64 colptr = Eigen::Map<VectorXi64>(App.outerIndexPtr(), N + 1);
    VectorXd nnzval = Eigen::Map<VectorXd>(App.valuePtr(), nnz);
    // Some edge stats
    std::vector<Stats<int64_t>> edgesizestats(nlevels, Stats<int64_t>());
    std::vector<Stats<int64_t>> nedgestats(nlevels, Stats<int64_t>());
    // Create all edges
    std::vector<Cluster*> cmap(N);
    for (auto& self : bottom_original()) {
        assert(self->start() >= 0);
        for (int64_t k = self->start(); k < self->start() + self->size(); k++) { cmap[k] = self.get(); }
    }
    for (auto& self : bottom_original()) {
        // Get all neighbors in column
        int64_t col  = self->start();
        int64_t size = self->size();
        std::set<Cluster*> nbrs;
        for (int64_t j = col; j < col + size; ++j) {
            // Get the column
            for (SpMat::InnerIterator it(App,j); it; ++it) {
                if (this->symmetry() && (it.row() < j)) continue; // Symmetric: skip stuff above the diagonal
                Cluster* n = cmap[it.row()];
                nbrs.insert(n);
            }
        }
        nbrs.insert(self.get());
        // Go and get the actual edges
        for (auto nbr : nbrs) {
            pMatrixXd A = std::make_unique<MatrixXd>(nbr->size(), self->size());
            setZero(A.get());
            block2dense(rowval, colptr, nnzval, nbr->start(), self->start(), nbr->size(), self->size(), A.get(), false);
            edgesizestats[self->level()].addData(nbr->size() * self->size());
            pEdge e = std::make_unique<Edge>(self.get(), nbr, std::move(A), true);
            self->add_edge(std::move(e));
        }
        nedgestats[self->level()].addData(nbrs.size());
        self->sort_edges();
    }
    if (this->verb) {
        printf("Edge size statistics (Leaf-cluster edge size at each level of the ND hierarchy)\n");
        printf("Lvl     Count       Min       Max      Mean\n");
        for (int64_t lvl = 0; lvl < nlevels; lvl++) {
            printf("%3d %9d %9d %9d %9.0f\n", lvl, edgesizestats[lvl].getCount(), edgesizestats[lvl].getMin(), edgesizestats[lvl].getMax(), edgesizestats[lvl].getMean());
        }
        printf("Edge count statistics (Leaf-cluster edge count at each level of the ND hierarchy)\n");
        printf("Lvl     Count       Min       Max      Mean\n");
        for (int64_t lvl = 0; lvl < nlevels; lvl++) {
            printf("%3d %9d %9d %9d %9.0f\n", lvl, nedgestats[lvl].getCount(), nedgestats[lvl].getMin(), nedgestats[lvl].getMax(), nedgestats[lvl].getMean());
        }
    }
    timer tend = wctime();
    if (this->verb) printf("Assembly time : %3.2e s. (%3.2e permuting A)\n", elapsed(tstart, tend), elapsed(t0, t1));
}

/**
 * Factorize
 */

// Scale pivot A = L L^T, replace A by L
void Tree::potf_cluster(Cluster* self) {
    MatrixXd* Ass  = self->pivot()->A();
    timer t_ = wctime();
    int64_t info = potf(Ass); // Ass = L L^T
    timer t__ = wctime();    
    if (info != 0) {
        std::cout << "Not SPD!" << std::endl;
        throw std::runtime_error("Error: Non-SPD Pivot\n");
    }
    this->tprof[this->ilvl].potf += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].pivot.push_back({Ass->rows(), elapsed(t_, t__)});
}

// Scale pivot Ass = P L D LT PT, returns P, L, D
void Tree::ldlt_cluster(Cluster* self, pMatrixXd* L, pVectorXd* d, pVectorXi64* p) {
    *p = std::make_unique<VectorXi64>(self->size());
    *d = std::make_unique<VectorXd>(self->size());
    *L = std::make_unique<MatrixXd>(self->size(), self->size());
    MatrixXd* Ass  = self->pivot()->A();
    double rcond = 1.0;
    timer t_ = wctime();
    int64_t info = ldlt(Ass, L->get(), d->get(), p->get(), &rcond);
    timer t__ = wctime();
    if (info != 0) {
        std::cout << "Singular Pivot in LDLT!" << std::endl;
        throw std::runtime_error("Error: Singular Pivot\n");
    }
    this->tprof[this->ilvl].ldlt += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].pivot.push_back({Ass->rows(), elapsed(t_, t__)});
}

// Scale pivot Ass = P L U Q, returns L, U, P, Q
void Tree::getf_cluster(Cluster* self, pMatrixXd* L, pMatrixXd* U, pVectorXi64* p, pVectorXi64* q) {
    *p = std::make_unique<VectorXi64>(self->size());
    *q = std::make_unique<VectorXi64>(self->size());
    *L = std::make_unique<MatrixXd>(self->size(), self->size());            
    *U = std::make_unique<MatrixXd>(self->size(), self->size());
    MatrixXd* Ass  = self->pivot()->A();
    timer t_ = wctime();
    if (this->scale_kind == ScalingKind::PLUQ) {
        fpgetf(Ass, p->get(), q->get()); // Ass = P L U Q
    } else {
        int64_t info = getf(Ass, p->get()); // Ass = P L U
        if (info != 0) {
            std::cout << "Singular Pivot!" << std::endl;
            throw std::runtime_error("Error: Singular Pivot\n");
        }
        *(*q) = VectorXi64::LinSpaced(self->size(), 0, self->size()-1);
    }
    timer t__ = wctime();    
    split_LU(Ass, L->get(), U->get());
    this->tprof[this->ilvl].getf += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].pivot.push_back({Ass->rows(), elapsed(t_, t__)});
}

// Assume A = L L^T is the pivot
// edge->n2 is self, edge->n1 is the nbr
// edge is n -> s
void Tree::trsm_potf_edgeIn(Edge* edge){
    Cluster*  self = edge->n2;
    MatrixXd* Asn  = edge->A();
    MatrixXd* L    = self->pivot()->A();
    timer t_ = wctime();
    trsm_left(L, Asn, Uplo::Lower, Op::NoTrans, Diag::NonUnit); // Lss^(-1) Asn
    timer t__ = wctime();
    this->tprof[this->ilvl].trsm += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].panel.push_back({Asn->cols(), Asn->rows(), elapsed(t_, t__)});
}

// Assumes A = L L^T is the pivot
// edge->n1 is self, edge->n2 is the nbr
// edge is s -> n
void Tree::trsm_potf_edgeOut(Edge* edge){
    Cluster*  self = edge->n1;
    MatrixXd* Ans  = edge->A();    
    MatrixXd* L    = self->pivot()->A();
    timer t_ = wctime();
    trsm_right(L, Ans, Uplo::Lower, Op::Trans, Diag::NonUnit); // Ans Lss^-T
    timer t__ = wctime();
    this->tprof[this->ilvl].trsm += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].panel.push_back({Ans->rows(), Ans->cols(), elapsed(t_, t__)});
}

// Assume A = P L U Q
// edge->n2 is self, edge->n1 is the nbr
// edge is n -> s
void Tree::trsm_getf_edgeIn(Edge* edge, MatrixXd* L, VectorXi64* p){
    MatrixXd* Asn  = edge->A();
    timer t_ = wctime();
    (*Asn) = p->asPermutation().transpose() * (*Asn);
    trsm_left(L, Asn, Uplo::Lower, Op::NoTrans, Diag::NonUnit); // Lss^-1 Pss^T Asn
    timer t__ = wctime();
    this->tprof[this->ilvl].trsm += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].panel.push_back({Asn->cols(), Asn->rows(), elapsed(t_, t__)});
}

// Assumes A = P U L Q
// edge->n1 is self, edge->n2 is the nbr
// edge is s -> n
void Tree::trsm_getf_edgeOut(Edge* edge, MatrixXd* U, VectorXi64* q){
    MatrixXd* Ans  = edge->A();
    timer t_ = wctime();
    (*Ans) = (*Ans) * q->asPermutation().transpose();
    trsm_right(U, Ans, Uplo::Upper, Op::NoTrans, Diag::NonUnit); // Ans Qss^T Uss^-1
    timer t__ = wctime();
    this->tprof[this->ilvl].trsm += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].panel.push_back({Ans->rows(), Ans->cols(), elapsed(t_, t__)});
}

// A = P L S LT PT with S a sign
// edge->n2 is self, edge->n1 is the nbr
// edge is n -> s
void Tree::trsm_ldlt_edgeIn(Edge* edge, Eigen::MatrixXd* L, Eigen::VectorXi64* p) {
    MatrixXd* Asn  = edge->A();
    timer t_ = wctime();
    (*Asn) = p->asPermutation().transpose() * (*Asn);
    trsm_left( L, Asn, Uplo::Lower, Op::NoTrans, Diag::NonUnit); // Lss^-1 Pss^T Asn
    timer t__ = wctime();
    this->tprof[this->ilvl].trsm += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].panel.push_back({Asn->cols(), Asn->rows(), elapsed(t_, t__)});
}

// A = P L S LT PT with S a sign
// edge->n1 is self, edge->n2 is the nbr
// edge is s -> n
void Tree::trsm_ldlt_edgeOut(Edge* edge, Eigen::MatrixXd* L, Eigen::VectorXi64* p) {
    MatrixXd* Ans  = edge->A();
    timer t_ = wctime();
    (*Ans) = (*Ans) * p->asPermutation();
    trsm_right(L, Ans, Uplo::Lower, Op::Trans, Diag::NonUnit); // Ans Pss Lss^-T
    timer t__ = wctime();
    this->tprof[this->ilvl].trsm += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].panel.push_back({Ans->rows(), Ans->cols(), elapsed(t_, t__)});
}

void Tree::panel_potf(Cluster* self) {
    for (auto edge : self->edgesInNbr()){
        trsm_potf_edgeIn(edge);
    }
    for (auto edge : self->edgesOutNbr()){
        trsm_potf_edgeOut(edge);
    }
}

void Tree::panel_ldlt(Cluster* self, MatrixXd* L, VectorXi64* p) {
    for (auto edge : self->edgesInNbr()) {
        trsm_ldlt_edgeIn(edge, L, p);
    }
    for (auto edge : self->edgesOutNbr()) {
        trsm_ldlt_edgeOut(edge, L, p);
    }
}

void Tree::panel_getf(Cluster* self, MatrixXd* L, MatrixXd* U, VectorXi64* p, VectorXi64* q) {
    for (auto edge : self->edgesInNbr()) {
        trsm_getf_edgeIn(edge, L, p);
    }
    for (auto edge : self->edgesOutNbr()) {
        trsm_getf_edgeOut(edge, U, q);
    }
}


// if ! transpose_edge_1, edge1 is self -> n1 (out, n1 = ROW)
// if ! transpose_edge_2, edge2 is n2 -> self (in,  n2 = COLUMN)
// Compute An1n2 -= An1s Diag^-1 Asn2
void Tree::gemm_edges(Edge* edge1, Edge* edge2, VectorXd* diag, bool transpose_edge_1, bool transpose_edge_2){
    /**
        [s ]     [n2]   edge is n2 -> s
        [n1]     {X}    edge is s  -> n1
    **/
    Cluster*  s     = transpose_edge_1 ? edge1->n2 : edge1->n1;
    Cluster*  s2    = transpose_edge_2 ? edge2->n1 : edge2->n2;
    assert(s == s2);
    Cluster*  n1    = transpose_edge_1 ? edge1->n1 : edge1->n2;
    MatrixXd* An1s  = edge1->A();
    Cluster*  n2    = transpose_edge_2 ? edge2->n2 : edge2->n1;
    MatrixXd* Asn2  = edge2->A();    
    // Look for An1n2
    auto begin = n2->edgesOutAll().begin();
    auto end   = n2->edgesOutAll().end();
    auto found = std::find_if(begin, end, [&](Edge* e){ return e->n2 == n1; } );
    if (found == end) { // An1n2 doesn't exist - create fill in
        pMatrixXd An1n2 = std::make_unique<MatrixXd>(n1->size(), n2->size());
        setZero(An1n2.get());
        pEdge e = std::make_unique<Edge>(n2, n1, std::move(An1n2), false);
        n2->add_edge(std::move(e));
        auto begin = n2->edgesOutAll().begin();
        auto end   = n2->edgesOutAll().end();
        found = std::find_if(begin, end, [&](Edge* e){ return e->n2 == n1; } );
    }
    // We have An1n2
    assert(found != n2->edgesOutAll().end());
    MatrixXd* An1n2 = (*found)->A();
    timer t_ = wctime();
    if (diag == nullptr) {
        if (n1 == n2 && this->symmetry()) {
            syrk(An1s, An1n2, transpose_edge_1 ? Op::Trans : Op::NoTrans, -1.0, 1.0);
        } else {
            gemm_spand(An1s, Asn2, An1n2, transpose_edge_1 ? Op::Trans : Op::NoTrans, transpose_edge_2 ? Op::Trans : Op::NoTrans, -1.0, 1.0);
        }
    } else {
        if     ( (! transpose_edge_1) && (! transpose_edge_2)) (*An1n2).noalias() -= (*An1s)             * (diag->cwiseInverse().asDiagonal()) * (*Asn2);
        else if ( (! transpose_edge_1) && (  transpose_edge_2)) (*An1n2).noalias() -= (*An1s)             * (diag->cwiseInverse().asDiagonal()) * (*Asn2).transpose();
        else if ( (  transpose_edge_1) && (! transpose_edge_2)) (*An1n2).noalias() -= (*An1s).transpose() * (diag->cwiseInverse().asDiagonal()) * (*Asn2);
        else if ( (  transpose_edge_1) && (  transpose_edge_2)) (*An1n2).noalias() -= (*An1s).transpose() * (diag->cwiseInverse().asDiagonal()) * (*Asn2).transpose();
        else assert(false);
    }
    timer t__ = wctime();
    this->tprof[this->ilvl].gemm += elapsed(t_, t__);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].gemm.push_back({An1n2->rows(), An1n2->cols(), s->size(), elapsed(t_, t__)});
}

// Scale a cluster to make it identity
void Tree::scale_cluster(Cluster* self) {
    timer t0, t1, t2;
    pMatrixXd NewPiv = std::make_unique<MatrixXd>();
    if (this->scale_kind == ScalingKind::LLT) {
        // Factor pivot in-place
        t0 = wctime();
        potf_cluster(self);    
        t1 = wctime();
        panel_potf(self);
        t2 = wctime();
        // Record & make diagonal identity
        if (preserve) {
            trmm_trans(self->pivot()->A(), self->phi());
        }
        this->ops.push_back(std::make_unique<ScalingLLT>(self, self->pivot()->get_A()));
        *NewPiv = MatrixXd::Identity(self->size(), self->size());        
    } else if (this->scale_kind == ScalingKind::LDLT) {
        pMatrixXd L = nullptr;
        pVectorXd s = nullptr;
        pVectorXi64 p = nullptr;
        t0 = wctime();
        ldlt_cluster(self, &L, &s, &p);
        assert(L != nullptr && s != nullptr && p != nullptr);
        // Factor panel in place
        t1 = wctime();
        panel_ldlt(self, L.get(), p.get());
        t2 = wctime();
        // Record
        if (preserve) {
            *self->phi() = p->asPermutation().transpose() * (*self->phi());
            trmm_trans(L.get(), self->phi());
        }
        VectorXd* diag = s.get();
        this->ops.push_back(std::make_unique<ScalingLDLT>(self, std::move(L), std::move(s), std::move(p))); // Fwd, Bwd and Diag
        *NewPiv = diag->asDiagonal();
    } else if (this->scale_kind == ScalingKind::PLUQ || this->scale_kind == ScalingKind::PLU) {
        // Factor pivot in-place
        pMatrixXd L = nullptr;
        pMatrixXd U = nullptr;
        pVectorXi64 p = nullptr;
        pVectorXi64 q = nullptr;
        t0 = wctime();
        getf_cluster(self, &L, &U, &p, &q);
        assert(L != nullptr && U != nullptr && p != nullptr && q != nullptr);
        // Factor panel in place
        t1 = wctime();
        panel_getf(self, L.get(), U.get(), p.get(), q.get());
        // Schur complement
        t2 = wctime();
        // Record
        assert(! preserve);
        this->ops.push_back(std::make_unique<ScalingPLUQ>(self, std::move(L), std::move(U), std::move(p), std::move(q)));
        *NewPiv = MatrixXd::Identity(self->size(), self->size());
    } else {
        std::cout << "Wrong scaling kind" << std::endl;
        assert(false);
    }
    self->pivot()->set_A(std::move(NewPiv));
    this->tprof[this->ilvl].scale_pivot += elapsed(t0, t1);
    this->tprof[this->ilvl].scale_panel += elapsed(t1, t2);
}

void Tree::schur_symmetric(Cluster* self) {
    schur_symmetric(self, nullptr);
}

void Tree::schur_symmetric(Cluster* self, VectorXd* Adiag) {
    for (auto edge1 : self->edgesOutNbr()){ // s -> n1, row
        for (auto edge2 : self->edgesOutNbr()){ // s -> n2, col
            if (edge1->n2->order() >= edge2->n2->order()) {
                gemm_edges(edge1, edge2, Adiag, false, true);
            }
        }
    }
    for (auto edge1 : self->edgesInNbr()){ // n1 -> s, row
        for (auto edge2 : self->edgesInNbr()){ // n2 -> s, col
            if (edge1->n1->order() >= edge2->n1->order()) {
                gemm_edges(edge1, edge2, Adiag, true, false);
            }
        }
    }
    for (auto edge1 : self->edgesOutNbr()){ // s -> n1, row
        for (auto edge2 : self->edgesInNbr()){ // n2 -> s, col
            assert(edge1->n2->order() > edge2->n1->order());
            gemm_edges(edge1, edge2, Adiag, false, false);
        }
    }
}

void Tree::record_schur_symmetric(Cluster* self, VectorXd* Adiag) {
    for (auto edge : self->edgesOutNbr()) {
        this->ops.push_back(std::make_unique<GemmSymmOut>(self, edge->n2, edge->get_A(), Adiag));
    }
    for (auto edge : self->edgesInNbr()) {
        this->ops.push_back(std::make_unique<GemmSymmIn>(self, edge->n1, edge->get_A(), Adiag));
    }
}

// Eliminate a cluster
void Tree::eliminate_cluster(Cluster* self){
    timer t0, t1, t2, t3;
    if (this->scale_kind == ScalingKind::LLT) {
        // Factor pivot in-place
        t0 = wctime();
        potf_cluster(self);    
        // Factor panel in place
        t1 = wctime();
        panel_potf(self);
        // Schur complement
        t2 = wctime();
        schur_symmetric(self, nullptr);
        t3 = wctime(); 
        // Record
        this->ops.push_back(std::make_unique<ScalingLLT>(self, self->pivot()->get_A())); // Fwd and Bwd
        record_schur_symmetric(self, nullptr);
    } else if (this->scale_kind == ScalingKind::LDLT) {
        pMatrixXd L = nullptr;
        pVectorXd s = nullptr;
        pVectorXi64 p = nullptr;
        t0 = wctime();
        ldlt_cluster(self, &L, &s, &p);
        assert(L != nullptr && s != nullptr && p != nullptr);
        VectorXd* diag = s.get();
        // Factor panel in place
        t1 = wctime();
        panel_ldlt(self, L.get(), p.get());
        // Schur complement
        t2 = wctime();
        schur_symmetric(self, diag);
        t3 = wctime(); 
        // Record
        this->ops.push_back(std::make_unique<ScalingLDLT>(self, std::move(L), std::move(s), std::move(p))); // Fwd, Bwd and Diag
        record_schur_symmetric(self, diag);        
    } else if (this->scale_kind == ScalingKind::PLUQ || this->scale_kind == ScalingKind::PLU) {
        // Factor pivot in-place
        pMatrixXd L = nullptr;
        pMatrixXd U = nullptr;
        pVectorXi64 p = nullptr;
        pVectorXi64 q = nullptr;
        t0 = wctime();
        getf_cluster(self, &L, &U, &p, &q);
        assert(L != nullptr && U != nullptr && p != nullptr && q != nullptr);
        // Factor panel in place
        t1 = wctime();
        panel_getf(self, L.get(), U.get(), p.get(), q.get());
        // Schur complement
        t2 = wctime();
        for (auto edge1 : self->edgesOutNbr()){ // s -> n1
            for (auto edge2 : self->edgesInNbr()){ // n2 -> s    
                gemm_edges(edge1, edge2, nullptr, false, false);
            }
        }
        t3 = wctime(); 
        // Record
        this->ops.push_back(std::make_unique<ScalingPLUQ>(self, std::move(L), std::move(U), std::move(p), std::move(q))); // Fwd and Bwd
        for (auto edge : self->edgesOutNbr()) {
            this->ops.push_back(std::make_unique<GemmOut>(self, edge->n2, edge->get_A())); // Fwd only
        }
        for (auto edge : self->edgesInNbr()) {
            this->ops.push_back(std::make_unique<GemmIn>(self, edge->n1, edge->get_A())); // Bwd only
        }
    } else {
        std::cout << "Wrong scaling choice" << std::endl;
        assert(false);
    }
    // Update data structure
    assert(self->parent() == nullptr);
    self->set_eliminated();
    this->tprof[this->ilvl].elim_pivot += elapsed(t0, t1);
    this->tprof[this->ilvl].elim_panel += elapsed(t1, t2);
    this->tprof[this->ilvl].elim_schur += elapsed(t2, t3);
}

/**
 * Split a cluster original into two, itself ("coarse", with an unchanged parent/children, and a smaller size) and a sibling ("fine", without parent/children)
 * For edges e for which pred(e) is true:
 *     Aon is Asn[:,cols]
 *     Asn doesn't exist
 * For edges e for which pred(e) is false:
 *     Aon, Asn are the original Q^T Asn, split
 * Asn corresponds to the new edges original->nbr for edges e such that 
 * 
 * sibling is sent to the end and given a last order
 */
Cluster* Tree::shrink_split_scatter_phi(Cluster* original, int64_t original_size, std::function<bool(Edge*)> pred, MatrixXd* v, VectorXd* h, MatrixXd* Asn, bool is_pivot_I, bool keep_sibling_if_pred_false) {
    assert(original->size() >= original_size);
    assert(original->size() == original->original_size());
    // Create a new sibling cluster
    int64_t sibling_size = original->size() - original_size;
    pCluster psibling = std::make_unique<Cluster>(original->start() + original_size, 
                                             original->size()  - original_size, 
                                             original->level(),
                                             get_new_order(),
                                             original->should_sparsify());
    Cluster* sibling = psibling.get();
    // Shrink original
    original->set_size(original_size);    
    if (preserve) {
        MatrixXd* phi_original = original->phi();
        // Apply Q on phi
        ormqr_spand(v, h, phi_original, Side::Left, Op::Trans); // Ass <- Q^T Ass
        // Split and shrink phi        
        sibling->set_phi(std::make_unique<MatrixXd>(sibling_size, phi_original->cols()));        
        assert(phi_original->rows() == original_size + sibling_size);
        (*sibling->phi()) = phi_original->middleRows(original_size, sibling_size);
        phi_original->conservativeResize(original_size, NoChange_t::NoChange);
    }
    // Scatter
    int64_t Asn_col = 0;
    // Edge before   
    for (auto edge : original->edgesInNbr()) {
        assert(edge->n2 == original);
        Cluster* n = edge->n1;  
        if (this->symmetry()) {
            assert(n->order() < original->order());
        }
        // From Asn for original, none for sibling
        if (pred(edge)) {
            assert(Asn != nullptr);
            int64_t cols = n->size();
            (*edge->A()) = Asn->middleCols(Asn_col, cols);
            Asn_col += cols;
        // From Q^T original for both
        } else {
            MatrixXd* Apn = edge->A();
            ormqr_spand(v, h, Apn, Side::Left, Op::Trans); // Q^T Apn
            // Allocate sibling
            if (keep_sibling_if_pred_false) {
                pMatrixXd Asn = std::make_unique<MatrixXd>(sibling_size,  n->size());
                (*Asn) = Apn->middleRows(original_size, sibling_size);
                pEdge Ens = std::make_unique<Edge>(n, sibling, std::move(Asn), edge->is_original());
                n->add_edge(std::move(Ens));
            }
            // Shrink original
            Apn->conservativeResize(original_size, NoChange_t::NoChange);
        }
    }
    // Edge after
    for (auto edge : original->edgesOutNbr()) {
        assert(edge->n1 == original);
        Cluster* n = edge->n2;
        if (this->symmetry()) {
            assert(n->order() > original->order());
        }
        // From Asn for original, none for sibling
        if (pred(edge)) {
            assert(Asn != nullptr);            
            int64_t cols = n->size();
            (*edge->A()) = Asn->middleCols(Asn_col, cols).transpose();
            Asn_col += cols;
        // From original Q for both
        } else {
            MatrixXd* Anp = edge->A();
            ormqr_spand(v, h, Anp, Side::Right, Op::NoTrans); // Anp Q
            // Allocate sibling
            if (keep_sibling_if_pred_false) {
                if (! this->symmetry()) {
                    pMatrixXd Ans = std::make_unique<MatrixXd>(n->size(), sibling_size);
                    (*Ans) = Anp->middleCols(original_size, sibling_size);
                    pEdge Esn = std::make_unique<Edge>(sibling, n, std::move(Ans), edge->is_original());
                    sibling->add_edge(std::move(Esn));
                } else {
                    pMatrixXd Asn = std::make_unique<MatrixXd>(sibling_size, n->size());
                    (*Asn) = Anp->middleCols(original_size, sibling_size).transpose();
                    pEdge Ens = std::make_unique<Edge>(n, sibling, std::move(Asn), edge->is_original());
                    n->add_edge(std::move(Ens));
                }
            }
            // Shrink original
            Anp->conservativeResize(NoChange_t::NoChange, original_size);
        }
    }
    if (Asn != nullptr) assert(Asn_col == Asn->cols());
    // Self edges
    MatrixXd* piv = original->pivot()->A();
    if (! is_pivot_I) {
        ormqr_spand(v, h, piv, Side::Right, Op::NoTrans); // Ass <- Ass Q 
        ormqr_spand(v, h, piv, Side::Left, Op::Trans); // Ass <- Q^T Ass
    }
    // Diagonal
    pMatrixXd Aoo = std::make_unique<MatrixXd>(original_size, original_size);
    pMatrixXd Ass = std::make_unique<MatrixXd>(sibling_size,  sibling_size);
    (*Aoo) = piv->block(0,             0,             original_size, original_size); // Aoo : s -> s
    (*Ass) = piv->block(original_size, original_size, sibling_size,  sibling_size);  // Ass : s -> s
    // Create s-s pivot
    pEdge Ess = std::make_unique<Edge>(sibling,   sibling, std::move(Ass), true);
    sibling->add_edge(std::move(Ess));
    // Create s-o and o-s edges, if needed
    if (! is_pivot_I) {
        // Create s-o edge
        pMatrixXd Aso = std::make_unique<MatrixXd>(sibling_size,  original_size);
        (*Aso) = piv->block(original_size, 0,             sibling_size,  original_size); // Aso : o -> s                
        pEdge Eso = std::make_unique<Edge>(original,  sibling, std::move(Aso), true);
        original->add_edge(std::move(Eso));
        // Create o-s edge
        if (! this->symmetry()) {
            pMatrixXd Aos = std::make_unique<MatrixXd>(original_size, sibling_size);
            (*Aos) = piv->block(0,             original_size, original_size, sibling_size);  // Aos : s -> o
            pEdge Eos = std::make_unique<Edge>(sibling,   original, std::move(Aos), true);
            sibling->add_edge(std::move(Eos));
        }
    }
    // Reset o-o pivot
    original->pivot()->set_A(std::move(Aoo));
    // Store new sibling somewhere, return a view on it
    this->others.push_back(std::move(psibling));
    this->ops.push_back(std::make_unique<Split>(original, sibling));
    return sibling;
}

pOperation Tree::reset_size(Cluster* snew, std::map<Cluster*, int64_t>* posparent) {
    assert(current_bottom > 0);
    assert(this->ilvl == current_bottom-1);
    // Update sizes & posparent
    int64_t size = 0;
    for (auto sold : snew->children()){
        (*posparent)[sold] = size;
        size += sold->size();
    }
    snew->reset_size(size);
    // Merge phis
    if (preserve) {
        assert(snew->children().size() > 0);
        auto kid = snew->children()[0];
        snew->set_phi(std::make_unique<MatrixXd>(snew->size(), kid->phi()->cols()));
        int64_t row = 0;
        for (auto sold : snew->children()) {
            assert(sold->phi()->rows() == sold->size());
            assert(sold->phi()->cols() == snew->phi()->cols());
            snew->phi()->middleRows(row, sold->size()) = *(sold->phi());
            row += sold->size();
        }  
        assert(row == snew->size());
    }
    return std::make_unique<Merge>(snew);
}

void Tree::update_edges(Cluster* snew, std::map<Cluster*,int64_t>* posparent) {
    assert(current_bottom > 0);
    assert(this->ilvl == current_bottom-1);
    // Figure out edges that gets together
    std::map<Cluster*,bool> edges_merged; // old neighbor => original?
    for (auto sold : snew->children()){
        for (auto eold : sold->edgesOutAll()){
            auto nold = eold->n2;
            auto nnew = nold->parent();
            // Default original is false
            if (edges_merged.count(nnew) == 0) {
                edges_merged[nnew] = false;
            }
            // This makes sure that if any edge is original, the new merged edge is original
            if (! edges_merged[nnew]) {
                edges_merged[nnew] = eold->is_original();
            }            
        }
    }
    // Allocate memory, create new edges 
    for (auto nnew_orig : edges_merged) {
        auto nnew = nnew_orig.first;
        auto original = nnew_orig.second;
        timer t0 = wctime();
        pMatrixXd A = std::make_unique<MatrixXd>(nnew->size(), snew->size());
        setZero(A.get());
        timer t1 = wctime();
        this->tprof[this->ilvl].merge_alloc += elapsed(t0, t1);
        pEdge e = std::make_unique<Edge>(snew, nnew, std::move(A), original);
        snew->add_edge(std::move(e));
    }
    snew->sort_edges(); // Make the edge ordering reproducible (set above can make it ~random)
    // Fill edges, delete previous edges
    for (auto sold : snew->children()){
        for (auto eold : sold->edgesOutAll()){
            auto nold = eold->n2;
            auto nnew = nold->parent();
            auto found = std::find_if(snew->edgesOutAll().begin(), snew->edgesOutAll().end(), [&nnew](Edge* e){return e->n2 == nnew;});
            assert(found != snew->edgesOutAll().end());                            
            timer t0 = wctime();
            /**  [x x] . [. x]
            *    [x x] . [x .]
            *     . .
            *    [. x]
            *    [x .]           **/
            (*found)->A()->block((*posparent)[nold], (*posparent)[sold], nold->size(), sold->size()) = *eold->A();            
            timer t1 = wctime();
            this->tprof[this->ilvl].merge_copy += elapsed(t0, t1);
        }
        sold->clear_edges();
    }
}

// Get [Asn] for all n != s (symmetric and unsymmetric case)
// Order is always
// [before, after]
pMatrixXd Tree::assemble_Asn(Cluster* self, std::function<bool(Edge*)> pred) {
    timer t0 = wctime();
    int64_t rows = self->size();
    // How many columns & prealloc
    int64_t cols = 0;
    for (auto edge : self->edgesInNbr()) { // n -> s, Asn
        if (pred(edge))
            cols += edge->n1->size();
    }    
    for (auto edge : self->edgesOutNbr()) { // s -> n, Ans
        if (pred(edge))
            cols += edge->n2->size();
    }
    auto Asn = std::make_unique<MatrixXd>(rows, cols);
    // Fill
    int64_t c = 0;
    for (auto edge : self->edgesInNbr()) {
        if (pred(edge)) {
            int64_t cols = edge->n1->size();
            Asn->middleCols(c, cols) = *edge->A(); // Asn
            c += cols;  
        }      
    }
    for (auto edge : self->edgesOutNbr()) {
        if (pred(edge)) {
            int64_t cols = edge->n2->size();
            Asn->middleCols(c, cols) = edge->A()->transpose(); // Ans^T
            c += cols;
        }
    }
    assert(c == cols);
    timer t1 = wctime();
    this->tprof[this->ilvl].assmb += elapsed(t0, t1);
    this->log[this->ilvl].nbrs.addData(cols);
    return Asn;
}

// Get [Asn phi, phi]
pMatrixXd Tree::assemble_Asphi(Cluster* self) {
    assert(this->symmetry());
    int64_t rows = self->size();
    // How many neighbors ?
    int64_t cols = 0;
    int64_t nphis = this->nphis();
    for (auto edge : self->edgesInNbr()) { // n -> s, Asn
        (void)edge;
        cols += nphis;
    }    
    for (auto edge : self->edgesOutAll()) { // s -> n, Ans
        (void)edge;
        cols += nphis;
    }    
    // Build matrix [phis, Asn*phin] into Q1
    auto Asnp = std::make_unique<MatrixXd>(rows, cols);
    int64_t c = 0;
    for (auto edge : self->edgesInNbr()) {
        Asnp->middleCols(c, nphis) = (*edge->A()) * (*(edge->n1->phi()));
        c += nphis;
    }
    for (auto edge : self->edgesOutAll()) { // This includes the pivot
        Asnp->middleCols(c, nphis) = edge->A()->transpose() * (*(edge->n2->phi()));
        c += nphis;
    }    
    assert(c == cols);
    return Asnp;
}

// Preserve only
void Tree::sparsify_preserve_only(Cluster* self) {
    assert(this->symmetry());
    bool is_pivot_I = (this->symm_kind != SymmKind::SYM);
    // Get edge
    timer t0 = wctime();
    auto Asnp = this->assemble_Asphi(self); // new
    int64_t rows = Asnp->rows();
    int64_t cols = Asnp->cols();
    if (cols >= rows) return;
    // Orthogonalize
    timer t1 = wctime();
    int64_t rank = std::min(rows, cols);
    pVectorXd h = std::make_unique<VectorXd>(rank); // new
    timer tgeqrf_0 = wctime();
    geqrf_spand(Asnp.get(), h.get());
    timer tgeqrf_1 = wctime();
    this->tprof[this->ilvl].geqrf += elapsed(tgeqrf_0, tgeqrf_1);
    timer t2 = wctime();
    Asnp->conservativeResize(rows, rank);
    pMatrixXd v = std::move(Asnp);    
    // Record
    MatrixXd* pv = v.get();
    VectorXd* ph = h.get();
    this->ops.push_back(std::make_unique<Orthogonal>(self, std::move(v), std::move(h)));        
    // Scatter Q, shrink and split
    Cluster* sibling = this->shrink_split_scatter_phi(self, rank, [](Edge*e){(void)e; return false;}, pv, ph, nullptr, is_pivot_I, false);    
    // Eliminate sibling
    eliminate_cluster(sibling);
    timer t3 = wctime();
    this->tprof[this->ilvl].spars_assmb += elapsed(t0, t1);
    this->tprof[this->ilvl].spars_spars += elapsed(t1, t2);
    this->tprof[this->ilvl].spars_scatt += elapsed(t2, t3);    
}

// RRQR only : interested in modifying to avoid densifying the sparse submatrix.
void Tree::sparsify_adaptive_only(Cluster* self, std::function<bool(Edge*)> pred, bool make_Asn_dense) {
    // (s,n) here  --> (p, n) in the paper
    //
    // The paper computes low-rank approx of A_pn, so we're interested in this function's Asn.
    //
    bool is_pivot_I = (this->symm_kind != SymmKind::SYM);
    MatrixXd* Ass = self->pivot()->A();
    if (is_pivot_I) {
        assert((*Ass - MatrixXd::Identity(self->size(), self->size())).norm() == 0.0);
    } else { // Then it's +- I
        assert((Ass->cwiseAbs() - MatrixXd::Identity(self->size(), self->size())).norm() == 0.0);
    }
    // Asn = [Asn_1 Asn_2 ... Asn_k]
    timer t0 = wctime();
    auto Asn = this->assemble_Asn(self, pred);
    int64_t rows = Asn->rows();
    int64_t cols = Asn->cols();
    VectorXi64 jpvt = VectorXi64::Zero(cols);
    VectorXd ht   = VectorXd(std::min(rows, cols));
    // GEQP3
    timer t1 = wctime();
    geqp3_spand(Asn.get(), &jpvt, &ht);
    timer t2 = wctime();
    this->tprof[this->ilvl].geqp3 += elapsed(t1, t2);
    if (this->monitor_flops) this->tprof_flops[this->ilvl].rrqr.push_back({Asn->rows(), Asn->cols(), elapsed(t1, t2)});    
    // Determine numerical rank
    // ----------------------------------------------------------------------------------------------
    //  Let "Rsn" denote the R-factor from QRCP of Asn.
    //  Right now we infer numerical rank by looking at the diagonal of Rsn.
    //  Specifically, do
    //      rank = max{ k : rcond(diag(Asn)[:k]) >= tol }.
    //  If we only had a partial QR decomposition (because Asn is kept sparse), then we could set
    //      rank = min{ k_max ,  max{ k : rcond(diag(Rsn)) }  } 
    //
    VectorXd diag = Asn->diagonal();
    if (this->monitor_Rdiag) { self->Rdiag = std::vector<double>(diag.data(), diag.data() + diag.size()); }
    int64_t rank = choose_rank(diag, tol);
    std::cout << ", numerical rank = " << rank << std::endl;
    if (rank >= rows) { // No point, nothing to do
        return;
    }
    // Get access to Q as a linear operator via our custom "Orthogonal" class,
    //  which wraps the Householder vectors from this QR decomposition.
    //
    // Note: Depending on the calls to the .fwd() and .bwd() methods 
    // of the resulting Orthogonal object, it might be that we get away with
    // only keeping the explicit representation of Q. If that isn't the case, then we'll
    // need a temporary copy of Q that we can make explicit in order to form R.
    // 
    timer tQ_0 = wctime();
    pMatrixXd v = std::make_unique<MatrixXd>(rows, rank);
    *v = Asn->leftCols(rank);
    pVectorXd h = std::make_unique<VectorXd>(rank);
    *h = ht.topRows(rank);
    timer tQ_1 = wctime();
    this->tprof[this->ilvl].buildq += elapsed(tQ_0, tQ_1);
    // Record Q
    MatrixXd* pv = v.get();
    VectorXd* ph = h.get();
    this->ops.push_back(std::make_unique<Orthogonal>(self, std::move(v), std::move(h)));
    // Compute shrinked edges
    timer tS_0 = wctime();
    MatrixXd AsnP = Asn->topRows(rank).triangularView<Upper>();
    AsnP = AsnP * (jpvt.asPermutation().transpose());
    timer tS_1 = wctime();
    this->tprof[this->ilvl].perma += elapsed(tS_0, tS_1);
    assert(AsnP.rows() == rank);
    // Scatter Asn, apply Q, shrink and split
    Cluster* sibling = this->shrink_split_scatter_phi(self, rank, pred, pv, ph, &AsnP, is_pivot_I, true);    
    // Eliminate sibling
    eliminate_cluster(sibling);
    timer t3 = wctime();
    this->tprof[this->ilvl].spars_assmb += elapsed(t0, t1);
    this->tprof[this->ilvl].spars_spars += elapsed(t1, t2);
    this->tprof[this->ilvl].spars_scatt += elapsed(t2, t3);    
}

// Preserve + RRQR
void Tree::sparsify_preserve_adaptive(Cluster* self) {
    bool is_pivot_I = (this->symm_kind != SymmKind::SYM);
    assert(this->symmetry());    
    int64_t rows = self->size();
    // (1) Get edge
    auto Asnphi = this->assemble_Asphi(self); // [Asn phi, phi]    
    int64_t cols1 = Asnphi->cols();
    assert(Asnphi->rows() == rows);
    assert(Asnphi->cols() == cols1);
    // QR
    int64_t rank1 = std::min(rows, cols1);
    VectorXd h1 = VectorXd(rank1);
    timer tgeqrf_0 = wctime();
    geqrf_spand(Asnphi.get(), &h1);
    timer tgeqrf_1 = wctime();
    this->tprof[this->ilvl].geqrf += elapsed(tgeqrf_0, tgeqrf_1);
    // Build Q1
    Asnphi->conservativeResize(rows, rank1);
    pMatrixXd Q1 = std::move(Asnphi);
    orgqr_spand(Q1.get(), &h1);
    // (2) Get edge
    auto Asn = this->assemble_Asn(self, [](Edge* e){(void)e; return true;}); // Asn    
    int64_t cols2 = Asn->cols();
    VectorXi64 jpvt = VectorXi64::Zero(cols2);
    VectorXd h2   = VectorXd(std::min(rows, cols2));
    // Remove Q1
    (*Asn) -= (*Q1) * (Q1->transpose() * (*Asn));
    // GEQP3
    timer tgeqp3_0 = wctime();
    geqp3_spand(Asn.get(), &jpvt, &h2);
    timer tgeqp3_1 = wctime();
    this->tprof[this->ilvl].geqp3 += elapsed(tgeqp3_0, tgeqp3_1);
    // Truncate ?
    VectorXd diag = Asn->diagonal();
    int64_t rank2 = choose_rank(diag, tol);
    int64_t rank = rank1 + rank2;
    if (rank >= rows) {
        return;
    }
    // Build Q2
    Asn->conservativeResize(rows, rank2);
    h2.conservativeResize(rank2);
    orgqr_spand(Asn.get(), &h2);
    pMatrixXd Q2 = std::move(Asn);
    // Concatenate [Q1, Q2] & orthogonalize
    pMatrixXd v = std::make_unique<MatrixXd>(rows, rank);
    pVectorXd h = std::make_unique<VectorXd>(rank);
    *v << *Q1, *Q2;
    timer tgeqrf_2 = wctime();
    geqrf_spand(v.get(), h.get());
    timer tgeqrf_3 = wctime();
    MatrixXd* pv = v.get();
    VectorXd* ph = h.get();
    this->tprof[this->ilvl].geqrf += elapsed(tgeqrf_2, tgeqrf_3);        
    this->ops.push_back(std::make_unique<Orthogonal>(self, std::move(v), std::move(h))); 
    // Scatter Q, shrink and split
    Cluster* sibling = this->shrink_split_scatter_phi(self, rank, [](Edge*e){(void)e; return false;}, pv, ph, nullptr, is_pivot_I, false);
    // Eliminate sibling
    eliminate_cluster(sibling);
}

void Tree::sparsify_cluster_farfield(Cluster* self) {
    this->log[this->ilvl].rank_before.addData(self->size());        
    this->sparsify_adaptive_only(self, [](Edge* e){return ! e->is_original();}, true);
    this->log[this->ilvl].rank_after.addData(self->size());
}

void Tree::sparsify_cluster(Cluster* self) {
    if (want_sparsify(self)) {
        this->log[this->ilvl].rank_before.addData(self->size());
        if (this->preserve) {
            if (this->tol <= 1.0) {
                this->sparsify_preserve_adaptive(self);                
            } else {
                this->sparsify_preserve_only(self);
            }            
        } else {
            this->sparsify_adaptive_only(self, [](Edge* e){(void)e; return true;}, true);
        }
        this->log[this->ilvl].rank_after.addData(self->size());
    } else {
        this->log[this->ilvl].ignored++;
    }
}

void Tree::merge_all() {
    current_bottom++;
    std::map<Cluster*,int64_t> posparent;
    for (auto& self : this->bottom_current()) {
        pOperation op = this->reset_size(self.get(), &posparent);
        this->ops.push_back(std::move(op));
    }
    for (auto& self : this->bottom_current()) {
        this->update_edges(self.get(), &posparent);
    }
}

void Tree::factorize() {

    timer tstart = wctime();
    if (symm_kind == SymmKind::SPD) {
        assert(scale_kind == ScalingKind::LLT);
    } else if (symm_kind == SymmKind::SYM) {
        assert(scale_kind == ScalingKind::LDLT);
    } else if (symm_kind == SymmKind::GEN) {
        assert(scale_kind == ScalingKind::PLU || scale_kind == ScalingKind::PLUQ);
    }
    if (preserve) {
        assert(this->symmetry());
    }
 
    if (this->verb) {
        std::cout << "spaND Factorization started" << std::endl;
        std::cout << "  N:          " << get_N()  << std::endl;
        std::cout << "  #levels:    " << nlevels  << std::endl;
        std::cout << "  verbose?:   " << verb     << std::endl;
        std::cout << "  tol?:       " << tol      << std::endl;
        std::cout << "  #skip:      " << skip     << std::endl;
        std::cout << "  #stop:      " << stop     << std::endl;
        std::cout << "  symmetrykd? " << symm2str(symm_kind) << std::endl;                                    
        std::cout << "  scalingkd?  " << scaling2str(scale_kind) << std::endl;
        std::cout << "  want_spars? " << use_want_sparsify << std::endl;
        std::cout << "  mon cond?   " << monitor_condition_pivots << std::endl;
        std::cout << "  preserving? " << preserve << std::endl;
        if (preserve)
            std::cout << "  preserving " << this->nphis() << " vectors" << std::endl;
    }
    
    // Create the phi
    if (preserve) {
        MatrixXd phi_ = this->perm.asPermutation().transpose() * (*phi);        
        // Store phi
        for (auto& self : bottom_original()) {
            self->set_phi(std::make_unique<MatrixXd>(self->size(), phi_.cols()));
            (*self->phi()) = phi_.middleRows(self->start(), self->size());
        }
    }

    // Factorization
    for (this->ilvl = 0; this->ilvl < nlevels; this->ilvl++) {
        
        if (this->verb) printf("Level %d, %d dofs left, %d clusters left\n", this->ilvl, this->ndofs_left(), this->nclusters_left());

        if (this->symmetry()) this->assert_symmetry();

        {
            // Eliminate interiors
            timer telim_0 = wctime();        
            for (auto& self : this->bottom_current()) {
                if (self->level() == this->ilvl) {
                    this->eliminate_cluster(self.get());                                        
                }
            }
            timer telim_1 = wctime();
            this->tprof[this->ilvl].elim += elapsed(telim_0, telim_1);
            this->log[this->ilvl].dofs_left_elim = this->ndofs_left();
            if (this->verb) printf("  Elim: %3.2e s., %d dofs left, %d clusters left\n", elapsed(telim_0, telim_1), this->log[this->ilvl].dofs_left_elim, this->nclusters_left());
        }

        if (this->ilvl >= skip && this->ilvl < stop) {

            // Scale
            timer tscale_0 = wctime();
            for (auto& self : this->bottom_current()) {
                if (self->level() > this->ilvl) {
                    this->scale_cluster(self.get());
                }
            }
            timer tscale_1 = wctime();
            this->tprof[this->ilvl].scale += elapsed(tscale_0, tscale_1);
            if (this->verb) printf("  Scaling: %3.2e s.\n", elapsed(tscale_0, tscale_1));

            // Sparsify        
            timer tspars_0 = wctime();
            for (auto& self : this->bottom_current()) {
                if (self->level() > this->ilvl) {
                    this->sparsify_cluster(self.get());
                }
            }
            timer tsparse_1 = wctime();
            this->tprof[this->ilvl].spars += elapsed(tspars_0, tsparse_1);
            if (this->verb) printf("  Sparsification: %3.2e s., %d dofs left, geqp3 %3.2e, geqrf %3.2e, assmb %3.2e, buildQ %3.2e, scatterQ %3.2e, permA %3.2e, scatterA %3.2e\n", 
                    elapsed(tspars_0, tsparse_1), this->ndofs_left(), this->tprof[this->ilvl].geqp3, this->tprof[this->ilvl].geqrf, 
                    this->tprof[this->ilvl].assmb, this->tprof[this->ilvl].buildq, this->tprof[this->ilvl].scattq, this->tprof[this->ilvl].perma, this->tprof[this->ilvl].scatta);

        } // ... if skip

        // Merge
        if (this->ilvl < nlevels-1) {
            timer tmerge_0 = wctime();        
            merge_all();
            timer tmerge_1 = wctime();
            this->tprof[this->ilvl].merge += elapsed(tmerge_0, tmerge_1);
            if (this->verb) printf("  Merge: %3.2e s., %d dofs left, %d clusters left\n", elapsed(tmerge_0, tmerge_1), this->ndofs_left(), this->nclusters_left());
        }        

        this->log[this->ilvl].dofs_left_spars = this->ndofs_left();
        this->log[this->ilvl].fact_nnz = this->nnz();

    } // TOP LEVEL for (int64_t this->ilvl = 0 ...)
    timer tend = wctime();
    if (this->verb) printf("Factorization: %3.2e s.\n", elapsed(tstart, tend));
}

void Tree::factorize_lorasp() {

    if (symm_kind == SymmKind::SPD) {
        assert(scale_kind == ScalingKind::LLT);
    } else if (symm_kind == SymmKind::SYM) {
        assert(scale_kind == ScalingKind::LDLT);
    } else if (symm_kind == SymmKind::GEN) {
        assert(scale_kind == ScalingKind::PLU || scale_kind == ScalingKind::PLUQ);
    }    
    assert(! preserve);

    timer tstart = wctime();    
    if (this->verb) {
        std::cout << "LoRaSp Factorization started" << std::endl;
        std::cout << "  N:          " << get_N()  << std::endl;
        std::cout << "  #levels:    " << nlevels  << std::endl;
        std::cout << "  verbose?:   " << verb     << std::endl;
        std::cout << "  tol?:       " << tol      << std::endl;
        std::cout << "  symmetrykd? " << symm2str(symm_kind) << std::endl;
        std::cout << "  scalingkd?  " << scaling2str(scale_kind) << std::endl;
        std::cout << "  mon cond?   " << monitor_condition_pivots << std::endl;
    }

    // Factorization
    for (this->ilvl = 0; this->ilvl < nlevels; this->ilvl++) {
        
        if (this->verb) printf("Level %d, %d dofs left, %d clusters left\n", this->ilvl, this->ndofs_left(), this->nclusters_left());        

        // Scale & Sparsify
        timer tspars_0 = wctime();
        for (auto& self : this->bottom_current()) {
            this->scale_cluster(self.get());
            this->sparsify_cluster_farfield(self.get());
        }
        timer tsparse_1 = wctime();
        this->tprof[this->ilvl].spars += elapsed(tspars_0, tsparse_1);
        if (this->verb) printf("  Sparsification: %3.2e s., %d dofs left, geqp3 %3.2e, geqrf %3.2e, assmb %3.2e, buildQ %3.2e, scatterQ %3.2e, permA %3.2e, scatterA %3.2e\n", 
                elapsed(tspars_0, tsparse_1), this->ndofs_left(), this->tprof[this->ilvl].geqp3, this->tprof[this->ilvl].geqrf, 
                this->tprof[this->ilvl].assmb, this->tprof[this->ilvl].buildq, this->tprof[this->ilvl].scattq, this->tprof[this->ilvl].perma, this->tprof[this->ilvl].scatta);

        // Merge
        if (this->ilvl < nlevels-1) {
            timer tmerge_0 = wctime();        
            merge_all();
            timer tmerge_1 = wctime();
            this->tprof[this->ilvl].merge += elapsed(tmerge_0, tmerge_1);
            if (this->verb) printf("  Merge: %3.2e s., %d dofs left, %d clusters left\n", elapsed(tmerge_0, tmerge_1), this->ndofs_left(), this->nclusters_left());
        }

        this->log[this->ilvl].dofs_left_spars = this->ndofs_left();
        this->log[this->ilvl].fact_nnz = this->nnz();

    } // TOP LEVEL for (int64_t this->ilvl = 0 ...)
    timer tend = wctime();
    if (this->verb) printf("Factorization: %3.2e s.\n", elapsed(tstart, tend));
}

void Tree::solve(VectorXd& x) const {
    // Permute
    VectorXd b = perm.asPermutation().transpose() * x;
    // Set solution
    for (auto& cluster : bottom_original()) {
        cluster->set_vector(b);
    }
    // Fwd
    for (auto io = ops.begin(); io != ops.end(); io++) {
        (*io)->fwd();
    }
    // Diagonal
    for (auto io = ops.begin(); io != ops.end(); io++) {
        (*io)->diag();
    }
    // Bwd
    for (auto io = ops.rbegin(); io != ops.rend(); io++) {
        (*io)->bwd();
    }
    // Extract solution
    for (auto& cluster : bottom_original()) {
        cluster->extract_vector(b);
    }
    // Permute back
    x = perm.asPermutation() * b;
}

long long Tree::nnz() const {
    long long nnz = 0;
    for (auto& op: ops) {
        nnz += op->nnz();
    }
    return nnz;
}

void Tree::print_log() const {
    // All sorts of timings
    printf("&>>& Lvl |      Elim     Scale  Sparsify     Merge\n");
    for (int64_t lvl = 0; lvl < this->nlevels; lvl++) {
        printf("&>>& %3d |   %7.1e   %7.1e   %7.1e   %7.1e\n", 
            lvl,
            this->tprof[lvl].elim,
            this->tprof[lvl].scale,
            this->tprof[lvl].spars,
            this->tprof[lvl].merge
            );
    }
    printf("&<<& Lvl |     Pivot     Panel     Schur |     Pivot     Panel |     Assmb     Spars     Scatt |     Alloc      Copy\n");
    for (int64_t lvl = 0; lvl < this->nlevels; lvl++) {
        printf("&<<& %3d |   %7.1e   %7.1e   %7.1e |   %7.1e   %7.1e |   %7.1e   %7.1e   %7.1e |   %7.1e   %7.1e\n", 
            lvl,
            this->tprof[lvl].elim_pivot,
            this->tprof[lvl].elim_panel,
            this->tprof[lvl].elim_schur,
            this->tprof[lvl].scale_pivot,
            this->tprof[lvl].scale_panel,
            this->tprof[lvl].spars_assmb,
            this->tprof[lvl].spars_spars,
            this->tprof[lvl].spars_scatt,
            this->tprof[lvl].merge_alloc,
            this->tprof[lvl].merge_copy
            );
    }
    printf("&++& Lvl |      potf      ldlt      trsm      gemm     geqp3     geqrf      syev     gesvd      getf    buildq    scattq     perma    scatta     assmb       phi\n");
    for (int64_t lvl = 0; lvl < this->nlevels; lvl++) {
        printf("&++& %3d |   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e\n", 
            lvl,            
            this->tprof[lvl].potf,
            this->tprof[lvl].ldlt,
            this->tprof[lvl].trsm,
            this->tprof[lvl].gemm,
            this->tprof[lvl].geqp3,
            this->tprof[lvl].geqrf,
            this->tprof[lvl].syev,
            this->tprof[lvl].gesvd,
            this->tprof[lvl].getf,
            this->tprof[lvl].buildq,
            this->tprof[lvl].scattq,
            this->tprof[lvl].perma,
            this->tprof[lvl].scatta,
            this->tprof[lvl].assmb,
            this->tprof[lvl].phi
            );
    }
    // Sizes and ranks
    printf("++++ Lvl |        ND    ND lft    El lft    Sp lft   Fct nnz    Rk Bfr    Rk Aft      Nbrs   Tot Bfr   Tot Aft   Cl Sped   Cl Ignd   ConEPiv     ConEL     ConEU  NorEDiag   ConSPiv     ConSL     ConSU  NorSDiag  AssymBfr  AssymAft   UppBefo   LowBefo   UppAfte   LowAfte\n");
    for (int64_t lvl = 0; lvl < this->nlevels; lvl++) {
        printf("++++ %3d | %9d %9d %9d %9d   %7.1e %9.0f %9.0f %9.0f %9d %9d %9d %9d   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e   %7.1e\n", 
            lvl,
            this->log[lvl].dofs_nd, 
            this->log[lvl].dofs_left_nd, 
            this->log[lvl].dofs_left_elim, 
            this->log[lvl].dofs_left_spars, 
            double(this->log[lvl].fact_nnz),
            this->log[lvl].rank_before.getMean(),
            this->log[lvl].rank_after.getMean(),
            this->log[lvl].nbrs.getMean(),
            this->log[lvl].rank_before.getSum(),
            this->log[lvl].rank_after.getSum(),
            this->log[lvl].rank_after.getCount(),
            this->log[lvl].ignored,
            this->log[lvl].cond_diag_elim.getMax(),
            this->log[lvl].cond_L_elim.getMax(),
            this->log[lvl].cond_U_elim.getMax(),
            this->log[lvl].norm_diag_elim.getMean(),
            this->log[lvl].cond_diag_scal.getMax(),
            this->log[lvl].cond_L_scal.getMax(),
            this->log[lvl].cond_U_scal.getMax(),
            this->log[lvl].norm_diag_scal.getMean(),
            this->log[lvl].Asym_before_scaling,
            this->log[lvl].Asym_after_scaling,
            this->log[lvl].Anorm_before_upper.getMax(),
            this->log[lvl].Anorm_before_lower.getMax(),
            this->log[lvl].Anorm_after_upper.getMax(),
            this->log[lvl].Anorm_after_lower.getMax()
            );
    }
}

/** Return the current trailing matrix **/
SpMat Tree::get_trailing_mat() const {
    // Build matrix at current stage
    std::vector<Triplet<double>> values(0);
    for (auto& s : bottom_current()) {
        int64_t s1 = s->start();
        assert(s1 >= 0);
        int64_t s2 = s->size();
        for (auto e : s->edgesOutAll()) {
            assert(e->n1 == s.get());
            int64_t n1 = e->n2->start();
            assert(n1 >= 0);
            int64_t n2 = e->n2->size();
            for (int64_t i_ = 0; i_ < n2; i_++) {
                for (int64_t j_ = 0; j_ < s2; j_++) {
                    int64_t i = n1 + i_;
                    int64_t j = s1 + j_;
                    double v = (*e->A())(i_,j_);
                    if (this->symmetry() && i > j) {
                        values.push_back(Triplet<double>(j,i,v));
                        values.push_back(Triplet<double>(i,j,v));
                    } else if (this->symmetry() && i == j) {
                        values.push_back(Triplet<double>(i,i,v));
                    } else if (! this->symmetry()) {
                        values.push_back(Triplet<double>(i,j,v));
                    }
                }
            }
        }
    }
    int64_t N = this->get_N();
    SpMat A(N,N);
    A.setFromTriplets(values.begin(), values.end());   
    return A;
}

MatrixXd Tree::get_current_x() const {
    MatrixXd X = MatrixXd::Zero(this->get_N(), this->nlevels);
    for (int64_t lvl = 0; lvl < this->nlevels; lvl++) {
        for (const auto& s : bottoms[lvl]){
            X.block(s->start(), lvl, s->get_x()->size(), 1) = *(s->get_x());
        }
    }
    return X;
}

}
