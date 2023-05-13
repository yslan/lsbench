#include "lsbench-impl.h"

#if defined(LSBENCH_ROCALUTION)
#include <rocalution/rocalution.hpp>

using namespace rocalution;

static int initialized = 0;

#define T double

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
  if (!initialized) {
    errx(EXIT_FAILURE, "rocALUTION is not initialized !\n");
    return 1;
  }

  // Print rocALUTION info
  if (cb->verbose > 1)
    info_rocalution();

  // rocALUTION objects
  int nr = A->nrows, nnz = A->offs[nr];
  LocalVector<T> roc_x, roc_r, roc_e;
  LocalMatrix<T> roc_mat;

  // Allocate vectors
  roc_x.Allocate("roc_x", nr);
  roc_r.Allocate("roc_r", nr);
  roc_e.Allocate("roc_e", nr);

  // Allocate a CSR matrix
  int *csr_row_ptr = new int[nr + 1];
  int *csr_col_ind = new int[nnz];
  T *csr_val = new T[nnz];

  // Fill the CSR matrix
  for (unsigned r = 0; r < nr + 1; r++)
    csr_row_ptr[r] = A->offs[r];
  for (unsigned i = 0; i < nnz; i++) {
    csr_col_ind[i] = A->cols[i] - A->base;
    csr_val[i] = A->vals[i];
  }

  // Set the CSR matrix data, csr_row_ptr, csr_col and csr_val pointers become
  // invalid
  roc_mat.SetDataPtrCSR(&csr_row_ptr, &csr_col_ind, &csr_val, "roc_mat", nnz,
                        nr, nr);

  for (int i = 0; i < nr; i++)
    roc_r[i] = r[i];

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
  _rocalution_sync();
  timer_log(2, 0);

  // Linear Solver
  CG<LocalMatrix<T>, LocalVector<T>, T> ls;
  // Preconditioner
  Jacobi<LocalMatrix<T>, LocalVector<T>, T> p;
  // Set solver operator
  ls.SetOperator(roc_mat);
  ls.SetPreconditioner(p);
  ls.SetResidualNorm(2);
  ls.Build();

  const int maxit = 10000;
  const T abs_tol = 1e-8, rel_tol = 1e-6, div_tol = 1e6;
  ls.InitMaxIter(maxit);
  ls.InitTol(abs_tol, rel_tol, div_tol);

  _rocalution_sync();
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
    roc_x.Zeros();
    ls.Solve(roc_r, &roc_x);

    // Compute error L2 norm
    roc_e.Zeros();
    roc_mat.Apply(roc_x, &roc_e);
    roc_e.ScaleAdd(-1.0, roc_r);

    T error = roc_e.Norm();
    std::cout << "rocalution norm(Ax-b) = " << error << std::endl;
  }

  // Warmup
  ls.Verbose(0);
  for (unsigned i = 0; i < cb->trials; i++) {
    roc_x.Zeros();
    ls.Solve(roc_r, &roc_x);
  }

  // Time the solve
  for (unsigned i = 0; i < cb->trials; i++) {
    roc_x.Zeros();

    _rocalution_sync();
    timer_log(4, 0);

    ls.Solve(roc_r, &roc_x);

    _rocalution_sync();
    timer_log(4, 1);
  }

  // copy sol back
  _rocalution_sync();
  timer_log(5, 0);

  roc_x.MoveToHost();

  _rocalution_sync();
  timer_log(5, 1);

  for (int i = 0; i < nr; i++)
    x[i] = roc_x[i];

  ls.Clear();
  roc_x.Clear();
  roc_e.Clear();
  roc_r.Clear();

  delete csr_row_ptr, csr_col_ind, csr_val;

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
