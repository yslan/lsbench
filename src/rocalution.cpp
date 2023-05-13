#include "lsbench-impl.h"

#if defined(LSBENCH_ROCALUTION)
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

template <typename T>
static void bench_pcg_jacobi(LocalMatrix<T> &roc_mat, LocalVector<T> &roc_x,
                             LocalVector<T> &roc_r, const struct lsbench *cb) {
  // Move objects to accelerator
  _rocalution_sync();
  timer_log(3, 0);

  roc_mat.MoveToAccelerator();
  roc_x.MoveToAccelerator();
  roc_r.MoveToAccelerator();

  _rocalution_sync();
  timer_log(3, 1);

  // Linear Solver Setup.
  _rocalution_sync();
  timer_log(2, 0);

  CG<LocalMatrix<T>, LocalVector<T>, T> ls;
  ls.SetOperator(roc_mat);

  Jacobi<LocalMatrix<T>, LocalVector<T>, T> p;
  ls.SetPreconditioner(p);

  ls.SetResidualNorm(2);

  ls.Build();

  const int maxit = 10000;
  const T abs_tol = 1e-8, rel_tol = 1e-6, div_tol = 1e6;
  ls.InitMaxIter(maxit);
  ls.InitTol(abs_tol, rel_tol, div_tol);

  _rocalution_sync();
  timer_log(2, 1);

  // Print matrix info
  if (cb->verbose > 1) {
    roc_mat.Info();
    roc_x.Info();
    roc_r.Info();

    ls.Verbose(1);
    roc_x.Zeros();
    ls.Solve(roc_r, &roc_x);

    // Compute error L2 norm
    LocalVector<T> roc_e;
    roc_e.Allocate("roc_e", roc_r.GetSize());
    roc_e.MoveToAccelerator();
    roc_e.Zeros();
    roc_mat.Apply(roc_x, &roc_e);
    roc_e.ScaleAdd(-1.0, roc_r);

    T error = roc_e.Norm();
    printf("rocalution pcg+jacobi norm(b-Ax) = %14.4e \n", error);
    fflush(stdout);

    roc_e.Clear();
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

  // Free all allocated data
  ls.Clear(), p.Clear();
}

template <typename T>
static void bench_sa_amg(LocalMatrix<T> &roc_mat, LocalVector<T> &roc_x,
                         LocalVector<T> &roc_r, const struct lsbench *cb) {
  // Move objects to accelerator
  _rocalution_sync();
  timer_log(3, 0);

  roc_mat.MoveToAccelerator();
  roc_x.MoveToAccelerator();
  roc_r.MoveToAccelerator();

  _rocalution_sync();
  timer_log(3, 1);

  // Linear Solver Setup.
  _rocalution_sync();
  timer_log(2, 0);

  // Linear Solver
  SAAMG<LocalMatrix<T>, LocalVector<T>, T> ls;

  // Set solver operator
  ls.SetOperator(roc_mat);

  // Set coupling strength
  ls.SetCouplingStrength(0.001);
  // Set maximal number of unknowns on coarsest level
  ls.SetCoarsestLevel(200);
  // Set relaxation parameter for smoothed interpolation aggregation
  ls.SetInterpRelax(2. / 3.);
  // Set manual smoothers
  ls.SetManualSmoothers(true);
  // Set manual course grid solver
  ls.SetManualSolver(true);
  // Set grid transfer scaling
  ls.SetScaling(true);
  // Set coarsening strategy
  ls.SetCoarseningStrategy(CoarseningStrategy::Greedy);
  // ls.SetCoarseningStrategy(CoarseningStrategy::PMIS);

  // Coarse Grid Solver
  CG<LocalMatrix<T>, LocalVector<T>, T> cgs;
  cgs.Verbose(0);

  // Obtain number of AMG levels
  int levels = ls.GetNumLevels();

  // Smoother for each level
  IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T> **sm =
      new IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T>
          *[levels - 1];
  Preconditioner<LocalMatrix<T>, LocalVector<T>, T> **p =
      new Preconditioner<LocalMatrix<T>, LocalVector<T>, T> *[levels - 1];

  std::string preconditioner = "Jacobi";

  // Initialize smoother for each level
  for (int i = 0; i < levels - 1; ++i) {
    FixedPoint<LocalMatrix<T>, LocalVector<T>, T> *fp;
    fp = new FixedPoint<LocalMatrix<T>, LocalVector<T>, T>;
    fp->SetRelaxation(1.3);
    sm[i] = fp;

    if (preconditioner == "GS") {
      p[i] = new GS<LocalMatrix<T>, LocalVector<T>, T>;
    } else if (preconditioner == "SGS") {
      p[i] = new SGS<LocalMatrix<T>, LocalVector<T>, T>;
    } else if (preconditioner == "ILU") {
      p[i] = new ILU<LocalMatrix<T>, LocalVector<T>, T>;
    } else if (preconditioner == "IC") {
      p[i] = new IC<LocalMatrix<T>, LocalVector<T>, T>;
    } else {
      p[i] = new Jacobi<LocalMatrix<T>, LocalVector<T>, T>;
    }

    sm[i]->SetPreconditioner(*p[i]);
    sm[i]->Verbose(0);
  }

  // Pass smoother and coarse grid solver to AMG
  ls.SetSmoother(sm);
  ls.SetSolver(cgs);

  // Set number of pre and post smoothing steps
  ls.SetSmootherPreIter(1);
  ls.SetSmootherPostIter(2);

  ls.SetResidualNorm(2);

  ls.Build();

  const int maxit = 10000;
  const T abs_tol = 1e-8, rel_tol = 1e-6, div_tol = 1e6;
  ls.InitMaxIter(maxit);
  ls.InitTol(abs_tol, rel_tol, div_tol);

  _rocalution_sync();
  timer_log(2, 1);

  // Print matrix info
  if (cb->verbose > 1) {
    roc_mat.Info();
    roc_x.Info();
    roc_r.Info();

    ls.Verbose(1);
    roc_x.Zeros();
    ls.Solve(roc_r, &roc_x);

    // Compute error L2 norm
    LocalVector<T> roc_e;
    roc_e.Allocate("roc_e", roc_r.GetSize());
    roc_e.MoveToAccelerator();
    roc_e.Zeros();
    roc_mat.Apply(roc_x, &roc_e);
    roc_e.ScaleAdd(-1.0, roc_r);

    T error = roc_e.Norm();
    printf("rocalution amg norm(b-Ax) = %14.4e \n", error);
    fflush(stdout);

    roc_e.Clear();
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

  // Free all allocated data
  ls.Clear();
  for (int i = 0; i < levels - 1; ++i) {
    delete p[i];
    delete sm[i];
  }

  delete[] p;
  delete[] sm;
}

template <typename T>
static int bench_aux(double *x, struct csr *A, const double *r,
                     const struct lsbench *cb) {
  if (!initialized) {
    errx(EXIT_FAILURE, "rocALUTION is not initialized !\n");
    return 1;
  }

  // Print rocALUTION info.
  if (cb->verbose > 1)
    info_rocalution();

  // rocALUTION objects.
  _rocalution_sync();
  timer_log(1, 0);

  int nr = A->nrows, nnz = A->offs[nr];
  LocalVector<T> roc_x, roc_r;
  LocalMatrix<T> roc_mat;

  // Allocate vectors.
  roc_x.Allocate("roc_x", nr);
  roc_r.Allocate("roc_r", nr);

  // Allocate a CSR matrix.
  int *csr_row_ptr = new int[nr + 1];
  int *csr_col_ind = new int[nnz];
  T *csr_val = new T[nnz];

  // Fill the CSR matrix.
  for (unsigned r = 0; r < nr + 1; r++)
    csr_row_ptr[r] = A->offs[r];
  for (unsigned i = 0; i < nnz; i++) {
    csr_col_ind[i] = A->cols[i] - A->base;
    csr_val[i] = A->vals[i];
  }

  // Set the CSR matrix data, csr_row_ptr, csr_col and csr_val pointers become
  // invalid.
  roc_mat.SetDataPtrCSR(&csr_row_ptr, &csr_col_ind, &csr_val, "roc_mat", nnz,
                        nr, nr);
  for (int i = 0; i < nr; i++)
    roc_r[i] = r[i];

  _rocalution_sync();
  timer_log(1, 1);

  // Benchmark PCG + Jacoib.
  bench_pcg_jacobi<T>(roc_mat, roc_x, roc_r, cb);

  timer_push();

  // bench_sa_amg<T>(roc_mat, roc_x, roc_r, cb);

  for (int i = 0; i < nr; i++)
    x[i] = roc_x[i];

  roc_x.Clear(), roc_r.Clear();
  delete csr_row_ptr, csr_col_ind, csr_val;

  return 0;
}

int rocalution_bench(double *x, struct csr *A, const double *r,
                     const struct lsbench *cb) {
  size_t prec;
  int ret;
  if (cb->precision == LSBENCH_PRECISION_FP64) {
    prec = sizeof(double);
    ret = bench_aux<double>(x, A, r, cb);
  } else if (cb->precision == LSBENCH_PRECISION_FP32) {
    prec = sizeof(float);
    ret = bench_aux<float>(x, A, r, cb);
  } else {
    errx(EXIT_FAILURE, "Requsted Precisions not supported !");
    return 1;
  }

  if (cb->verbose > 0) {
    printf("Precision: %d bytes.\n", prec);
    fflush(stdout);
  }

  return ret;
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
