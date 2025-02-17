// This code comes from MFEM
// Published under LGPL v2.1 license
// See https://github.com/mfem/mfem/blob/master/LICENSE for detail
// https://github.com/mfem/mfem/blob/master/examples/ex9.cpp
//
//                                MFEM Example 9
//
// Compile with: make ex9
//
// Sample runs:
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex9 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//    ex9 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/star-mixed.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex9 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.

// Modified to perform a backward euler with fixed dt

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "spaND.h"
#include "mfem_util.h"
#include "mmio.hpp"

using namespace std;
using namespace mfem;
using namespace Eigen;
using namespace spaND;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int64_t problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   mfem::SparseMatrix &M, &K;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;
   
   // Mfem implicit
   mutable Vector z;   
   mfem::SparseMatrix T;
   GMRESSolver T_solver;
   DSmoother T_prec; 

   // Spand Implicit
   SpMat A;
   Tree* t;

public:
   FE_Evolution(mfem::SparseMatrix &_M, mfem::SparseMatrix &_K, const Vector &_b, double, Eigen::MatrixXd&, double, int64_t);

   virtual void Mult(const Vector &x, Vector &y) const; // Computes y = f(x, t), i.e., y = M^{-1} (K u + b)
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k); // Solves k = f(x + dt*k, t + dt), i.e., solves for k: k = M^{-1} (K (x + dt*k) + b)

   virtual ~FE_Evolution() { }
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int64_t ref_levels = 2;
   int64_t order = 3;
   int64_t ode_solver_type = 4;
   double t_final = 10.0;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int64_t vis_steps = 5;
   double tol = 1e-2;
   int64_t skip = 4;

   int64_t precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&tol, "-tol", "--tol",
                  "spaND tolerance.");
   args.AddOption(&skip, "-skip", "--skip",
                  "skip bottom.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int64_t dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 0: ode_solver = new BackwardEulerSolver; break;
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int64_t lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec);
   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

   m.Assemble();
   m.Finalize();
   int64_t skip_zeros = 0;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   b.Assemble();

   FiniteElementSpace fesCoo(mesh, &fec, dim);
   GridFunction nodes(&fesCoo);
   mesh->GetNodes(nodes);
   int64_t N = nodes.Size() / dim;
   assert(N == fes.GetVSize());
   Eigen::MatrixXd Xcoo(dim, N);
   for (int64_t i = 0; i < N ; ++i) {
      for (int64_t j = 0; j < dim; ++j) {
         Xcoo(j,i) = nodes(j * N + i);
      }
   }
   cout << "Xcoo" << endl;
   cout << Xcoo.rows() << " x " << Xcoo.cols() << endl;
   cout << Xcoo.leftCols(10) << endl;

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex9.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("ex9-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example9", mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9", mesh);
         dc->SetPrecision(precision);
      }
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int64_t  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   SpMat K = mfem2eigen(k.SpMat());
   FE_Evolution adv(m.SpMat(), k.SpMat(), b, dt, Xcoo, tol, skip);

   // VectorXd part = adv.t->get_partition();
   // std::ofstream file2("part.txt");
   // file2 << part;
   // file2.close();
   // exit(1);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int64_t ti = 0; !done; )
   {
      
      ode_solver->Step(u, t, dt);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << *mesh << u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex9.mesh -g ex9-final.gf".
   {
      ofstream osol("ex9-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete dc;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(mfem::SparseMatrix &_M, mfem::SparseMatrix &_K, const Vector &_b, double dt, Eigen::MatrixXd &Xcoo, double tol, int64_t skip)
   : TimeDependentOperator(_M.Size()), M(_M), K(_K), b(_b), z(_M.Size())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   
   auto Meigen = mfem2eigen(_M); 
   auto Keigen = mfem2eigen(_K); 
   cout << "**** M norm     : " << Meigen.norm() << endl;
   cout << "**** K norm     : " << Keigen.norm() << endl;
   cout << "**** dt * K norm: " << dt * Keigen.norm() << endl;

   T = *Add(1.0, _M, -dt, _K);
   T_solver.SetPreconditioner(T_prec);
   T_solver.SetOperator(T);

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(1e-9);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(500);
   T_solver.SetPrintLevel(1);

   /** Setup spaND tree and preconditionner **/  
   A = mfem2eigen(T); 
   SpMat AAT = symmetric_graph(A);
   int64_t N = A.rows();
   cout << "Vertices? " << N << endl;
   cout << "NNZ? " << A.nonZeros() << endl;
   int64_t lvl = (int64_t)ceil(log( double(N) / 64.0)/log(2.0)) - 1;
   cout << "Lvl " << lvl << endl;
   
#if 0
   std::ofstream Coofs("mfem9_coords.txt", std::ofstream::out);
   Coofs << Xcoo;
   Coofs.close();
   mmio::sp_mmwrite("mfem9_A.txt", A);
#endif

   t = new Tree(lvl);
   t->set_symm_kind(SymmKind::GEN);
   t->set_tol(tol);
   t->set_skip(skip);
   t->set_Xcoo(&Xcoo);
   t->set_use_geo(true);
   t->set_use_sparsify(false);
   t->set_scaling_kind(ScalingKind::PLU);
   t->partition(AAT);
   t->assemble(A);
   try {
      t->factorize();
   } catch (std::exception& ex) {
      cout << ex.what();
   }
   t->print_log();
   // Try a random solve
   VectorXd b = VectorXd::Random(A.rows());
   VectorXd x = b;
   cout << "Random RHS GMRES\n";
   int64_t iter = gmres(A, b, x, *t, 100, 100, 1e-9, true);
   cout << "GMRES: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
}

void FE_Evolution::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   // Solves k = F(x + dt*k, t + dt) = M^(-1) ( K(x+dt*k) + b), ie,
   // T k =  K x + b, with T = M - dt K

   // z = K x + b
   K.Mult(x, z);
   z += b;

   // Mfem
   // k = T^{-1} w
   // T_solver.Mult(z, k);

   // Spand
   VectorXd rhs = mfem2eigen(z);
   VectorXd sol = VectorXd::Zero(rhs.rows());
   int64_t iter = gmres(A, rhs, sol, *t, 100, 100, 1e-9, true);
   cout << "GMRES: " << iter << " |Ax-b|/|b|: " << (A*sol-rhs).norm() / rhs.norm() << endl;
   eigen2mfem(sol, k);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int64_t dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int64_t i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
double u0_function(const Vector &x)
{
   int64_t dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int64_t i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
