#include "lsbench-impl.h"
//#include <math.h>

#if defined(LSBENCH_ROCALUTION)
#include <cstdlib>
#include <iostream>
//#include "common.hpp"
#include "rocalution/base/local_matrix.hpp"
#include <rocalution/rocalution.hpp>

using namespace rocalution;

static int initialized = 0;

//struct rocalution_csr {
//  unsigned nr;
//  rocalution_sparse *A;
//  rocalution_factor *L;
//  rocalution_dense *r;
//};

int rocalution_init() { 
  if (initialized)
    return 1;

  // Initialize rocALUTION
  init_rocalution();
}

int rocalution_bench(double *x, struct csr *A, const double *r,
                     const struct lsbench *cb) {
  if (!initialized)
    return 1;

  // Print rocALUTION info
  if (cb->verbose>0)
    info_rocalution();

  int nr = A->nrows, nnz = A->offs[nr];

  // rocALUTION objects
  LocalVector<double> roc_x;
  LocalVector<double> roc_rhs;
  LocalVector<double> roc_e;
  LocalMatrix<double> roc_mat;

  // Allocate vectors
  roc_x.Allocate("roc_x", nr);
  roc_rhs.Allocate("roc_rhs", nr);
  roc_e.Allocate("roc_e", nr);

  // Read matrix from MTX file
  // Allocate a CSR matrix
  int* csr_row_ptr   = new int[nr+1];
  int* csr_col_ind   = new int[nnz];
  double* csr_val = new double[nnz];

  // Fill the CSR matrix
  for (unsigned r = 0; r < nr+1; r++) {
    csr_row_ptr[r] = A->offs[r];
  }
  for (unsigned i = 0; i < nnz; i++) {
    csr_col_ind[i] = A->cols[i] - A->base;
    csr_val[i] = A->vals[i];
  }

  // Set the CSR matrix data, csr_row_ptr, csr_col and csr_val pointers become invalid
  roc_mat.SetDataPtrCSR(&csr_row_ptr, &csr_col_ind, &csr_val, "roc_mat", nnz, nr, nr);

  for (int i = 0; i < nr; i ++) {
    roc_rhs[i] = r[i];
  }

  // Move objects to accelerator
  roc_mat.MoveToAccelerator();
  roc_x.MoveToAccelerator();
  roc_rhs.MoveToAccelerator();
  roc_e.MoveToAccelerator();

  // Linear Solver
  CG<LocalMatrix<double>, LocalVector<double>, double> ls;

  // Preconditioner
  Jacobi<LocalMatrix<double>, LocalVector<double>, double> p;

  // Initialize rhs such that A 1 = rhs
//  roc_e.Ones();
//  roc_mat.Apply(roc_e, &roc_rhs);

  // Initial zero guess
  roc_x.Zeros();

  // Set solver operator
  ls.SetOperator(roc_mat);
  // Set solver preconditioner
  ls.SetPreconditioner(p);

  // Build solver
  ls.Build();

  // Verbosity output
  ls.Verbose(1);

  // Print matrix info
  roc_mat.Info();
  roc_x.Info();
  roc_rhs.Info();

  // Start time measurement
  double tick, tack;
  tick = rocalution_time();

  // Solve A x = rhs
  ls.Solve(roc_rhs, &roc_x);

  // Stop time measurement
  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack - tick) / 1e6 << " sec" << std::endl;

  // Clear solver
  ls.Clear();

  // Compute error L2 norm
  roc_e.Zeros();
  roc_mat.Apply(roc_x, &roc_e);
  roc_e.ScaleAdd(-1.0, roc_rhs);

  double error = roc_e.Norm();
  std::cout << "norm(Ax-b) = " << error << std::endl;

  // Stop rocALUTION platform
  stop_rocalution();
  return 0;
}

int rocalution_finalize() { return 1; }

#else // LSBENCH_HYPRE
int rocalution_init() { return 1; }
int rocalution_finalize() { return 1; }
int rocalution_bench(double *x, struct csr *A, const double *r,
                     const struct lsbench *cb) {
  return 1;
}
#endif


