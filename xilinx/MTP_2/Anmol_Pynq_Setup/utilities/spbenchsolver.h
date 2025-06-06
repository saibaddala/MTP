// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>
#include <fstream>
#include <SparseCore>
#include <BenchTimer.h>
#include <cstdlib>
#include <string>
#include <Cholesky>
#include <Jacobi>
#include <Householder>
#include <IterativeLinearSolvers>
#include <IterativeSolvers>
#include <LU>
#include <SparseExtra>
#include <SparseLU>

#include "spbenchstyle.h"

#ifdef EIGEN_METIS_SUPPORT
#include <MetisSupport>
#endif

#ifdef EIGEN_CHOLMOD_SUPPORT
#include <CholmodSupport>
#endif

#ifdef EIGEN_UMFPACK_SUPPORT
#include <UmfPackSupport>
#endif

#ifdef EIGEN_KLU_SUPPORT
#include <KLUSupport>
#endif

#ifdef EIGEN_PARDISO_SUPPORT
#include <PardisoSupport>
#endif

#ifdef EIGEN_SUPERLU_SUPPORT
#include <SuperLUSupport>
#endif

#ifdef EIGEN_PASTIX_SUPPORT
#include <PaStiXSupport>
#endif

// CONSTANTS
#define EIGEN_UMFPACK  10
#define EIGEN_KLU  11
#define EIGEN_SUPERLU  20
#define EIGEN_PASTIX  30
#define EIGEN_PARDISO  40
#define EIGEN_SPARSELU_COLAMD 50
#define EIGEN_SPARSELU_METIS 51
#define EIGEN_BICGSTAB  60
#define EIGEN_BICGSTAB_ILUT  61
#define EIGEN_GMRES 70
#define EIGEN_GMRES_ILUT 71
#define EIGEN_SIMPLICIAL_LDLT  80
#define EIGEN_CHOLMOD_LDLT  90
#define EIGEN_PASTIX_LDLT  100
#define EIGEN_PARDISO_LDLT  110
#define EIGEN_SIMPLICIAL_LLT  120
#define EIGEN_CHOLMOD_SUPERNODAL_LLT  130
#define EIGEN_CHOLMOD_SIMPLICIAL_LLT  140
#define EIGEN_PASTIX_LLT  150
#define EIGEN_PARDISO_LLT  160
#define EIGEN_CG  170
#define EIGEN_CG_PRECOND  180

using namespace Eigen;
using namespace std; 


// Global variables for input parameters
int MaximumIters; // Maximum number of iterations
double RelErr; // Relative error of the computed solution
double best_time_val; // Current best time overall solvers 
int best_time_id; //  id of the best solver for the current system 

template<typename T> inline typename NumTraits<T>::Real test_precision() { return NumTraits<T>::dummy_precision(); }
template<> inline float test_precision<float>() { return 1e-3f; }                                                             
template<> inline double test_precision<double>() { return 1e-6; }                                                            
template<> inline float test_precision<std::complex<float> >() { return test_precision<float>(); }
template<> inline double test_precision<std::complex<double> >() { return test_precision<double>(); }

void printStatheader(std::ofstream& out)
{
  // Print XML header
  // NOTE It would have been much easier to write these XML documents using external libraries like tinyXML or Xerces-C++.
  
  out << "<?xml version='1.0' encoding='UTF-8'?> \n";
  out << "<?xml-stylesheet type='text/xsl' href='#stylesheet' ?> \n"; 
  out << "<!DOCTYPE BENCH  [\n<!ATTLIST xsl:stylesheet\n id\t ID  #REQUIRED>\n]>";
  out << "\n\n<!-- Generated by the Eigen library -->\n"; 
  
  out << "\n<BENCH> \n" ; //root XML element 
  // Print the xsl style section
  printBenchStyle(out); 
  // List all available solvers 
  out << " <AVAILSOLVER> \n";
#ifdef EIGEN_UMFPACK_SUPPORT
  out <<"  <SOLVER ID='" << EIGEN_UMFPACK << "'>\n"; 
  out << "   <TYPE> LU </TYPE> \n";
  out << "   <PACKAGE> UMFPACK </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
#endif
#ifdef EIGEN_KLU_SUPPORT
  out <<"  <SOLVER ID='" << EIGEN_KLU << "'>\n";
  out << "   <TYPE> LU </TYPE> \n";
  out << "   <PACKAGE> KLU </PACKAGE> \n";
  out << "  </SOLVER> \n";
#endif
#ifdef EIGEN_SUPERLU_SUPPORT
  out <<"  <SOLVER ID='" << EIGEN_SUPERLU << "'>\n"; 
  out << "   <TYPE> LU </TYPE> \n";
  out << "   <PACKAGE> SUPERLU </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
#endif
#ifdef EIGEN_CHOLMOD_SUPPORT
  out <<"  <SOLVER ID='" << EIGEN_CHOLMOD_SIMPLICIAL_LLT << "'>\n"; 
  out << "   <TYPE> LLT SP</TYPE> \n";
  out << "   <PACKAGE> CHOLMOD </PACKAGE> \n";
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_CHOLMOD_SUPERNODAL_LLT << "'>\n"; 
  out << "   <TYPE> LLT</TYPE> \n";
  out << "   <PACKAGE> CHOLMOD </PACKAGE> \n";
  out << "  </SOLVER> \n";
  
  out <<"  <SOLVER ID='" << EIGEN_CHOLMOD_LDLT << "'>\n"; 
  out << "   <TYPE> LDLT </TYPE> \n";
  out << "   <PACKAGE> CHOLMOD </PACKAGE> \n";  
  out << "  </SOLVER> \n"; 
#endif
#ifdef EIGEN_PARDISO_SUPPORT
  out <<"  <SOLVER ID='" << EIGEN_PARDISO << "'>\n"; 
  out << "   <TYPE> LU </TYPE> \n";
  out << "   <PACKAGE> PARDISO </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_PARDISO_LLT << "'>\n"; 
  out << "   <TYPE> LLT </TYPE> \n";
  out << "   <PACKAGE> PARDISO </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_PARDISO_LDLT << "'>\n"; 
  out << "   <TYPE> LDLT </TYPE> \n";
  out << "   <PACKAGE> PARDISO </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
#endif
#ifdef EIGEN_PASTIX_SUPPORT
  out <<"  <SOLVER ID='" << EIGEN_PASTIX << "'>\n"; 
  out << "   <TYPE> LU </TYPE> \n";
  out << "   <PACKAGE> PASTIX </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_PASTIX_LLT << "'>\n"; 
  out << "   <TYPE> LLT </TYPE> \n";
  out << "   <PACKAGE> PASTIX </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_PASTIX_LDLT << "'>\n"; 
  out << "   <TYPE> LDLT </TYPE> \n";
  out << "   <PACKAGE> PASTIX </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
#endif
  
  out <<"  <SOLVER ID='" << EIGEN_BICGSTAB << "'>\n"; 
  out << "   <TYPE> BICGSTAB </TYPE> \n";
  out << "   <PACKAGE> EIGEN </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_BICGSTAB_ILUT << "'>\n"; 
  out << "   <TYPE> BICGSTAB_ILUT </TYPE> \n";
  out << "   <PACKAGE> EIGEN </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_GMRES_ILUT << "'>\n"; 
  out << "   <TYPE> GMRES_ILUT </TYPE> \n";
  out << "   <PACKAGE> EIGEN </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_SIMPLICIAL_LDLT << "'>\n"; 
  out << "   <TYPE> LDLT </TYPE> \n";
  out << "   <PACKAGE> EIGEN </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_SIMPLICIAL_LLT << "'>\n"; 
  out << "   <TYPE> LLT </TYPE> \n";
  out << "   <PACKAGE> EIGEN </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_CG << "'>\n"; 
  out << "   <TYPE> CG </TYPE> \n";
  out << "   <PACKAGE> EIGEN </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
  out <<"  <SOLVER ID='" << EIGEN_SPARSELU_COLAMD << "'>\n"; 
  out << "   <TYPE> LU_COLAMD </TYPE> \n";
  out << "   <PACKAGE> EIGEN </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
  
#ifdef EIGEN_METIS_SUPPORT
  out <<"  <SOLVER ID='" << EIGEN_SPARSELU_METIS << "'>\n"; 
  out << "   <TYPE> LU_METIS </TYPE> \n";
  out << "   <PACKAGE> EIGEN </PACKAGE> \n"; 
  out << "  </SOLVER> \n"; 
#endif
  out << " </AVAILSOLVER> \n"; 
  
}


template<typename Solver, typename Scalar>
void call_solver(Solver &solver, const int solver_id, const typename Solver::MatrixType& A, const Matrix<Scalar, Dynamic, 1>& b, const Matrix<Scalar, Dynamic, 1>& refX,std::ofstream& statbuf)
{
  
  double total_time;
  double compute_time;
  double solve_time; 
  double rel_error;
  Matrix<Scalar, Dynamic, 1> x; 
  BenchTimer timer; 
  timer.reset();
  timer.start();
  solver.compute(A); 
  if (solver.info() != Success)
  {
    std::cerr << "Solver failed ... \n";
    return;
  }
  timer.stop();
  compute_time = timer.value();
  statbuf << "    <TIME>\n"; 
  statbuf << "     <COMPUTE> " << timer.value() << "</COMPUTE>\n";
  std::cout<< "COMPUTE TIME : " << timer.value() <<std::endl; 
    
  timer.reset();
  timer.start();
  x = solver.solve(b); 
  if (solver.info() == NumericalIssue)
  {
    std::cerr << "Solver failed ... \n";
    return;
  }
  timer.stop();
  solve_time = timer.value();
  statbuf << "     <SOLVE> " << timer.value() << "</SOLVE>\n"; 
  std::cout<< "SOLVE TIME : " << timer.value() <<std::endl; 
  
  total_time = solve_time + compute_time;
  statbuf << "     <TOTAL> " << total_time << "</TOTAL>\n"; 
  std::cout<< "TOTAL TIME : " << total_time <<std::endl; 
  statbuf << "    </TIME>\n"; 
  
  // Verify the relative error
  if(refX.size() != 0)
    rel_error = (refX - x).norm()/refX.norm();
  else 
  {
    // Compute the relative residual norm
    Matrix<Scalar, Dynamic, 1> temp; 
    temp = A * x; 
    rel_error = (b-temp).norm()/b.norm();
  }
  statbuf << "    <ERROR> " << rel_error << "</ERROR>\n"; 
  std::cout<< "REL. ERROR : " << rel_error << "\n\n" ;
  if ( rel_error <= RelErr )
  {
    // check the best time if convergence
    if(!best_time_val || (best_time_val > total_time))
    {
      best_time_val = total_time;
      best_time_id = solver_id;
    }
  }
}

template<typename Solver, typename Scalar>
void call_directsolver(Solver& solver, const int solver_id, const typename Solver::MatrixType& A, const Matrix<Scalar, Dynamic, 1>& b, const Matrix<Scalar, Dynamic, 1>& refX, std::string& statFile)
{
    std::ofstream statbuf(statFile.c_str(), std::ios::app);
    statbuf << "   <SOLVER_STAT ID='" << solver_id <<"'>\n"; 
    call_solver(solver, solver_id, A, b, refX,statbuf);
    statbuf << "   </SOLVER_STAT>\n";
    statbuf.close();
}

template<typename Solver, typename Scalar>
void call_itersolver(Solver &solver, const int solver_id, const typename Solver::MatrixType& A, const Matrix<Scalar, Dynamic, 1>& b, const Matrix<Scalar, Dynamic, 1>& refX, std::string& statFile)
{
  solver.setTolerance(RelErr); 
  solver.setMaxIterations(MaximumIters);
  
  std::ofstream statbuf(statFile.c_str(), std::ios::app);
  statbuf << " <SOLVER_STAT ID='" << solver_id <<"'>\n"; 
  call_solver(solver, solver_id, A, b, refX,statbuf); 
  statbuf << "   <ITER> "<< solver.iterations() << "</ITER>\n";
  statbuf << " </SOLVER_STAT>\n";
  std::cout << "ITERATIONS : " << solver.iterations() <<"\n\n\n"; 
  
}


template <typename Scalar>
void SelectSolvers(const SparseMatrix<Scalar>&A, unsigned int sym, Matrix<Scalar, Dynamic, 1>& b, const Matrix<Scalar, Dynamic, 1>& refX, std::string& statFile)
{
  typedef SparseMatrix<Scalar, ColMajor> SpMat; 
  // First, deal with Nonsymmetric and symmetric matrices
  best_time_id = 0; 
  best_time_val = 0.0;
  //UMFPACK
  #ifdef EIGEN_UMFPACK_SUPPORT
  {
    cout << "Solving with UMFPACK LU ... \n"; 
    UmfPackLU<SpMat> solver; 
    call_directsolver(solver, EIGEN_UMFPACK, A, b, refX,statFile); 
  }
  #endif
  //KLU
  #ifdef EIGEN_KLU_SUPPORT
  {
    cout << "Solving with KLU LU ... \n";
    KLU<SpMat> solver;
    call_directsolver(solver, EIGEN_KLU, A, b, refX,statFile);
  }
  #endif
    //SuperLU
  #ifdef EIGEN_SUPERLU_SUPPORT
  {
    cout << "\nSolving with SUPERLU ... \n"; 
    SuperLU<SpMat> solver;
    call_directsolver(solver, EIGEN_SUPERLU, A, b, refX,statFile); 
  }
  #endif
    
   // PaStix LU
  #ifdef EIGEN_PASTIX_SUPPORT
  {
    cout << "\nSolving with PASTIX LU ... \n"; 
    PastixLU<SpMat> solver; 
    call_directsolver(solver, EIGEN_PASTIX, A, b, refX,statFile) ;
  }
  #endif

   //PARDISO LU
  #ifdef EIGEN_PARDISO_SUPPORT
  {
    cout << "\nSolving with PARDISO LU ... \n"; 
    PardisoLU<SpMat>  solver; 
    call_directsolver(solver, EIGEN_PARDISO, A, b, refX,statFile);
  }
  #endif
  
  // Eigen SparseLU METIS
  cout << "\n Solving with Sparse LU AND COLAMD ... \n";
  SparseLU<SpMat, COLAMDOrdering<int> >   solver;
  call_directsolver(solver, EIGEN_SPARSELU_COLAMD, A, b, refX, statFile); 
  // Eigen SparseLU METIS
  #ifdef EIGEN_METIS_SUPPORT
  {
    cout << "\n Solving with Sparse LU AND METIS ... \n";
    SparseLU<SpMat, MetisOrdering<int> >   solver;
    call_directsolver(solver, EIGEN_SPARSELU_METIS, A, b, refX, statFile); 
  }
  #endif
  
  //BiCGSTAB
  {
    cout << "\nSolving with BiCGSTAB ... \n"; 
    BiCGSTAB<SpMat> solver; 
    call_itersolver(solver, EIGEN_BICGSTAB, A, b, refX,statFile);
  }
  //BiCGSTAB+ILUT
  {
    cout << "\nSolving with BiCGSTAB and ILUT ... \n"; 
    BiCGSTAB<SpMat, IncompleteLUT<Scalar> > solver; 
    call_itersolver(solver, EIGEN_BICGSTAB_ILUT, A, b, refX,statFile); 
  }
  
   
  //GMRES
//   {
//     cout << "\nSolving with GMRES ... \n"; 
//     GMRES<SpMat> solver; 
//     call_itersolver(solver, EIGEN_GMRES, A, b, refX,statFile); 
//   }
  //GMRES+ILUT
  {
    cout << "\nSolving with GMRES and ILUT ... \n"; 
    GMRES<SpMat, IncompleteLUT<Scalar> > solver; 
    call_itersolver(solver, EIGEN_GMRES_ILUT, A, b, refX,statFile);
  }
  
  // Hermitian and not necessarily positive-definites
  if (sym != NonSymmetric)
  {
    // Internal Cholesky
    {
      cout << "\nSolving with Simplicial LDLT ... \n"; 
      SimplicialLDLT<SpMat, Lower> solver;
      call_directsolver(solver, EIGEN_SIMPLICIAL_LDLT, A, b, refX,statFile); 
    }
    
    // CHOLMOD
    #ifdef EIGEN_CHOLMOD_SUPPORT
    {
      cout << "\nSolving with CHOLMOD LDLT ... \n"; 
      CholmodDecomposition<SpMat, Lower> solver;
      solver.setMode(CholmodLDLt);
       call_directsolver(solver,EIGEN_CHOLMOD_LDLT, A, b, refX,statFile);
    }
    #endif
    
    //PASTIX LLT
    #ifdef EIGEN_PASTIX_SUPPORT
    {
      cout << "\nSolving with PASTIX LDLT ... \n"; 
      PastixLDLT<SpMat, Lower> solver; 
      call_directsolver(solver,EIGEN_PASTIX_LDLT, A, b, refX,statFile); 
    }
    #endif
    
    //PARDISO LLT
    #ifdef EIGEN_PARDISO_SUPPORT
    {
      cout << "\nSolving with PARDISO LDLT ... \n"; 
      PardisoLDLT<SpMat, Lower> solver; 
      call_directsolver(solver,EIGEN_PARDISO_LDLT, A, b, refX,statFile); 
    }
    #endif
  }

   // Now, symmetric POSITIVE DEFINITE matrices
  if (sym == SPD)
  {
    
    //Internal Sparse Cholesky
    {
      cout << "\nSolving with SIMPLICIAL LLT ... \n"; 
      SimplicialLLT<SpMat, Lower> solver; 
      call_directsolver(solver,EIGEN_SIMPLICIAL_LLT, A, b, refX,statFile); 
    }
    
    // CHOLMOD
    #ifdef EIGEN_CHOLMOD_SUPPORT
    {
      // CholMOD SuperNodal LLT
      cout << "\nSolving with CHOLMOD LLT (Supernodal)... \n"; 
      CholmodDecomposition<SpMat, Lower> solver;
      solver.setMode(CholmodSupernodalLLt);
       call_directsolver(solver,EIGEN_CHOLMOD_SUPERNODAL_LLT, A, b, refX,statFile);
      // CholMod Simplicial LLT
      cout << "\nSolving with CHOLMOD LLT (Simplicial) ... \n"; 
      solver.setMode(CholmodSimplicialLLt);
      call_directsolver(solver,EIGEN_CHOLMOD_SIMPLICIAL_LLT, A, b, refX,statFile);
    }
    #endif
    
    //PASTIX LLT
    #ifdef EIGEN_PASTIX_SUPPORT
    {
      cout << "\nSolving with PASTIX LLT ... \n"; 
      PastixLLT<SpMat, Lower> solver; 
      call_directsolver(solver,EIGEN_PASTIX_LLT, A, b, refX,statFile);
    }
    #endif
    
    //PARDISO LLT
    #ifdef EIGEN_PARDISO_SUPPORT
    {
      cout << "\nSolving with PARDISO LLT ... \n"; 
      PardisoLLT<SpMat, Lower> solver; 
      call_directsolver(solver,EIGEN_PARDISO_LLT, A, b, refX,statFile); 
    }
    #endif
    
    // Internal CG
    {
      cout << "\nSolving with CG ... \n"; 
      ConjugateGradient<SpMat, Lower> solver; 
      call_itersolver(solver,EIGEN_CG, A, b, refX,statFile);
    }
    //CG+IdentityPreconditioner
//     {
//       cout << "\nSolving with CG and IdentityPreconditioner ... \n"; 
//       ConjugateGradient<SpMat, Lower, IdentityPreconditioner> solver; 
//       call_itersolver(solver,EIGEN_CG_PRECOND, A, b, refX,statFile);
//     }
  } // End SPD matrices 
}

/* Browse all the matrices available in the specified folder 
 * and solve the associated linear system.
 * The results of each solve are printed in the standard output
 * and optionally in the provided html file
 */
template <typename Scalar>
void Browse_Matrices(const string folder, bool statFileExists, std::string& statFile, int maxiters, double tol)
{
  MaximumIters = maxiters; // Maximum number of iterations, global variable 
  RelErr = tol;  //Relative residual error  as stopping criterion for iterative solvers
  MatrixMarketIterator<Scalar> it(folder);
  for ( ; it; ++it)
  {
    //print the infos for this linear system 
    if(statFileExists)
    {
      std::ofstream statbuf(statFile.c_str(), std::ios::app);
      statbuf << "<LINEARSYSTEM> \n";
      statbuf << "   <MATRIX> \n";
      statbuf << "     <NAME> " << it.matname() << " </NAME>\n"; 
      statbuf << "     <SIZE> " << it.matrix().rows() << " </SIZE>\n"; 
      statbuf << "     <ENTRIES> " << it.matrix().nonZeros() << "</ENTRIES>\n";
      if (it.sym()!=NonSymmetric)
      {
        statbuf << "     <SYMMETRY> Symmetric </SYMMETRY>\n" ; 
        if (it.sym() == SPD) 
          statbuf << "     <POSDEF> YES </POSDEF>\n"; 
        else 
          statbuf << "     <POSDEF> NO </POSDEF>\n"; 
          
      }
      else
      {
        statbuf << "     <SYMMETRY> NonSymmetric </SYMMETRY>\n" ; 
        statbuf << "     <POSDEF> NO </POSDEF>\n"; 
      }
      statbuf << "   </MATRIX> \n";
      statbuf.close();
    }
    
    cout<< "\n\n===================================================== \n";
    cout<< " ======  SOLVING WITH MATRIX " << it.matname() << " ====\n";
    cout<< " =================================================== \n\n";
    Matrix<Scalar, Dynamic, 1> refX;
    if(it.hasrefX()) refX = it.refX();
    // Call all suitable solvers for this linear system 
    SelectSolvers<Scalar>(it.matrix(), it.sym(), it.rhs(), refX, statFile);
    
    if(statFileExists)
    {
      std::ofstream statbuf(statFile.c_str(), std::ios::app);
      statbuf << "  <BEST_SOLVER ID='"<< best_time_id
              << "'></BEST_SOLVER>\n"; 
      statbuf << " </LINEARSYSTEM> \n"; 
      statbuf.close();
    }
  } 
} 

bool get_options(int argc, char **args, string option, string* value=0)
{
  int idx = 1, found=false; 
  while (idx<argc && !found){
    if (option.compare(args[idx]) == 0){
      found = true; 
      if(value) *value = args[idx+1];
    }
    idx+=2;
  }
  return found; 
}
