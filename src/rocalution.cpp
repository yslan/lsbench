#include "lsbench-impl.h"

#if defined(LSBENCH_ROCALUTION)
#include <cstdlib>
#include <iostream>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

static int initialized = 0;

int rocalution_init() {
  if (initialized)
    return 1;

  // Initialize rocALUTION
  init_rocalution();
  initialized = 1;
  return 0;
}

int rocalution_bench(double *x, struct csr *A, const double *r,
                     const struct lsbench *cb) {
  if (!initialized)
    return 1;

  // Print rocALUTION info
  if (cb->verbose > 1)
    info_rocalution();

  int nr = A->nrows, nnz = A->offs[nr];

  // rocALUTION objects
  LocalVector<double> roc_x;
  LocalVector<double> roc_r;
  LocalVector<double> roc_e;
  LocalMatrix<double> roc_mat;

  // Allocate vectors
  roc_x.Allocate("roc_x", nr);
  roc_r.Allocate("roc_r", nr);
  roc_e.Allocate("roc_e", nr);

  // Allocate a CSR matrix
  int *csr_row_ptr = new int[nr + 1];
  int *csr_col_ind = new int[nnz];
  double *csr_val = new double[nnz];

  // Fill the CSR matrix
  for (unsigned r = 0; r < nr + 1; r++) {
    csr_row_ptr[r] = A->offs[r];
  }
  for (unsigned i = 0; i < nnz; i++) {
    csr_col_ind[i] = A->cols[i] - A->base;
    csr_val[i] = A->vals[i];
  }

  // Set the CSR matrix data, csr_row_ptr, csr_col and csr_val pointers become
  // invalid
  roc_mat.SetDataPtrCSR(&csr_row_ptr, &csr_col_ind, &csr_val, "roc_mat", nnz,
                        nr, nr);

  for (int i = 0; i < nr; i++) {
    roc_r[i] = r[i];
  }

  // Move objects to accelerator
  _rocalution_sync();
  timer_log(3, 0);
  roc_mat.MoveToAccelerator();
  roc_x.MoveToAccelerator();
  roc_r.MoveToAccelerator();
  roc_e.MoveToAccelerator();
  _rocalution_sync();
  timer_log(3, 1);

  // Solver Setup
  timer_log(2, 0);
  // Linear Solver
  CG<LocalMatrix<double>, LocalVector<double>, double> ls;

  // Preconditioner
  Jacobi<LocalMatrix<double>, LocalVector<double>, double> p;

  // Set solver operator
  ls.SetOperator(roc_mat);
  ls.SetPreconditioner(p);
  ls.Build();
  timer_log(2, 1);

  // Initial zero guess
  roc_x.Zeros();

  // Print matrix info
  if (cb->verbose > 1) {
    roc_mat.Info();
    roc_x.Info();
    roc_r.Info();
  }

  // Verbosity output
  if (cb->verbose > 1) {
    ls.Verbose(1);
    ls.Solve(roc_r, &roc_x);
  }

  // Warmup
  ls.Verbose(0);
  for (unsigned i = 0; i < cb->trials; i++)
    ls.Solve(roc_r, &roc_x);

  // Time the solve
  clock_t t;
  for (unsigned i = 0; i < cb->trials; i++) {
    _rocalution_sync();
    timer_log(4, 0);

    ls.Solve(roc_r, &roc_x);

    _rocalution_sync();
    timer_log(4, 1);
  }

  if (cb->verbose > 1) {
    // Compute error L2 norm
    roc_e.Zeros();
    roc_mat.Apply(roc_x, &roc_e);
    roc_e.ScaleAdd(-1.0, roc_r);

    double error = roc_e.Norm();
    std::cout << "rocalution norm(Ax-b) = " << error << std::endl;
  }

  // copy sol back
  _rocalution_sync();
  timer_log(5, 0);
  roc_x.MoveToHost();
  _rocalution_sync();
  timer_log(5, 1);
  for (int i = 0; i < nr; i++) {
    x[i] = roc_x[i];
  }

  ls.Clear();
  roc_x.Clear();
  roc_e.Clear();
  roc_r.Clear();
  return 0;
}

int rocalution_finalize() {
  if (!initialized)
    return 1;

  stop_rocalution();
  initialized = 0;
  return 0;
}

#else // LSBENCH_ROCALUTION
int rocalution_init() { return 1; }
int rocalution_finalize() { return 1; }
int rocalution_bench(double *x, struct csr *A, const double *r,
                     const struct lsbench *cb) {
  return 1;
}
#endif
