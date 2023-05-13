#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define TOKEN_PASTE_(a, b) a##b
#define TOKEN_PASTE(a, b) TOKEN_PASTE_(a, b)

#define gpuError_t TOKEN_PASTE(GPU, Error_t)
#define gpuSuccess TOKEN_PASTE(GPU, Success)
#define gpuGetErrorName TOKEN_PASTE(GPU, GetErrorName)
#define gpuMalloc TOKEN_PASTE(GPU, Malloc)
#define gpuMemcpy TOKEN_PASTE(GPU, Memcpy)
#define gpuFree TOKEN_PASTE(GPU, Free)
#define gpuMemcpyHostToDevice TOKEN_PASTE(GPU, MemcpyHostToDevice)
#define gpuMemcpyDeviceToHost TOKEN_PASTE(GPU, MemcpyDeviceToHost)
#define gpuDeviceSynchronize TOKEN_PASTE(GPU, DeviceSynchronize)
#define gpuGetDeviceCount TOKEN_PASTE(GPU, GetDeviceCount)
#define gpuSetDevice TOKEN_PASTE(GPU, SetDevice)
#define gpuGetDeviceProperties TOKEN_PASTE(GPU, GetDeviceProperties)
#define gpuGetErrorString TOKEN_PASTE(GPU, GetErrorString)

#define gpurtcResult TOKEN_PASTE(RUNTIME, Result)
#define gpurtcGetErrorString TOKEN_PASTE(RUNTIME, GetErrorString)
#define gpurtcProgram TOKEN_PASTE(RUNTIME, Program)
#define gpurtcCreateProgram TOKEN_PASTE(RUNTIME, CreateProgram)
#define gpurtcCompileProgram TOKEN_PASTE(RUNTIME, CompileProgram)
#define gpurtcGetProgramLogSize TOKEN_PASTE(RUNTIME, GetProgramLogSize)
#define gpurtcGetProgramLog TOKEN_PASTE(RUNTIME, GetProgramLog)
#define gpurtcDestroyProgram TOKEN_PASTE(RUNTIME, DestroyProgram)

#define chk_rt(err)                                                            \
  {                                                                            \
    gpuError_t err_ = (err);                                                   \
    if (err_ != gpuSuccess) {                                                  \
      errx(EXIT_FAILURE, "%s:%d " TOSTRING(GPU) " error: %s", __FILE__,        \
           __LINE__, gpuGetErrorString(err_));                                 \
    }                                                                          \
  }

#define csr_init TOKEN_PASTE(GPU, _csr_init)
static struct hypre_csr *csr_init(struct csr *A, const struct lsbench *cb) {

  struct hypre_csr *B = tcalloc(struct hypre_csr, 1);

  int comm = 0;
  HYPRE_BigInt lower = A->base, upper = (HYPRE_BigInt)A->nrows - 1 + A->base;
  HYPRE_IJMatrixCreate(comm, lower, upper, lower, upper, &B->A);
  HYPRE_IJMatrixSetObjectType(B->A, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(B->A);

  unsigned nr = A->nrows, nnz = A->offs[nr];
  HYPRE_BigInt *rows = tcalloc(HYPRE_BigInt, nr);
  HYPRE_BigInt *cols = tcalloc(HYPRE_BigInt, nnz);
  HYPRE_Int *ncols = tcalloc(HYPRE_Int, nr);
  HYPRE_Real *vals = tcalloc(HYPRE_Real, nnz);
  for (unsigned r = 0; r < nr; r++) {
    rows[r] = (HYPRE_BigInt)(r + A->base);
    for (unsigned j = A->offs[r], je = A->offs[r + 1]; j < je; j++) {
      cols[j] = (HYPRE_BigInt)A->cols[j];
      vals[j] = (HYPRE_Real)A->vals[j];
    }
    ncols[r] = (HYPRE_Int)(A->offs[r + 1] - A->offs[r]);
  }

  HYPRE_BigInt *d_rows, *d_cols;
  chk_rt(gpuMalloc((void **)&d_rows, nr * sizeof(HYPRE_BigInt)));
  chk_rt(gpuMalloc((void **)&d_cols, nnz * sizeof(HYPRE_BigInt)));
  chk_rt(gpuMemcpy(d_rows, rows, nr * sizeof(HYPRE_BigInt),
                   gpuMemcpyHostToDevice));
  chk_rt(gpuMemcpy(d_cols, cols, nnz * sizeof(HYPRE_BigInt),
                   gpuMemcpyHostToDevice));
  tfree(rows), tfree(cols);

  HYPRE_Int *d_ncols;
  chk_rt(gpuMalloc((void **)&d_ncols, nr * sizeof(HYPRE_Int)));
  chk_rt(
      gpuMemcpy(d_ncols, ncols, nr * sizeof(HYPRE_Int), gpuMemcpyHostToDevice));
  tfree(ncols);

  HYPRE_Real *d_vals;
  chk_rt(gpuMalloc((void **)&d_vals, nnz * sizeof(HYPRE_Real)));
  chk_rt(
      gpuMemcpy(d_vals, vals, nnz * sizeof(HYPRE_Real), gpuMemcpyHostToDevice));
  tfree(vals);

  HYPRE_IJMatrixSetValues(B->A, nr, d_ncols, d_rows, d_cols, d_vals);
  HYPRE_IJMatrixAssemble(B->A);
  // HYPRE_IJMatrixPrint(B->A, "A.dat");

  // chk_rt(gpuFree((void *)d_rows));
  // chk_rt(gpuFree((void *)d_cols));
  // chk_rt(gpuFree((void *)d_ncols));
  // chk_rt(gpuFree((void *)d_vals));

  // Create and initialize rhs and solution vectors
  HYPRE_IJVectorCreate(comm, lower, upper, &B->b);
  HYPRE_IJVectorSetObjectType(B->b, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(B->b);
  HYPRE_IJVectorAssemble(B->b);

  HYPRE_IJVectorCreate(comm, lower, upper, &B->x);
  HYPRE_IJVectorSetObjectType(B->x, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(B->x);
  HYPRE_IJVectorAssemble(B->x);

  HYPRE_ParVector par_b, par_x;
  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJVectorGetObject(B->b, (void **)&par_b);
  HYPRE_IJVectorGetObject(B->x, (void **)&par_x);
  HYPRE_IJMatrixGetObject(B->A, (void **)&par_A);

  if (cb->verbose > 1) {
    HYPRE_BoomerAMGSetPrintLevel(solver, 3);
  }
  HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);

  return B;
}

#define hypre_bench TOKEN_PASTE(GPU, _hypre_bench)
int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {

  if (!initialized)
    return 1;

  struct hypre_csr *B = csr_init(A, cb);

  unsigned nr = A->nrows, nnz = A->offs[nr];
  HYPRE_Real *d_r, *d_x;
  chk_rt(gpuMalloc((void **)&d_x, nr * sizeof(HYPRE_Real)));
  chk_rt(gpuMalloc((void **)&d_r, nr * sizeof(HYPRE_Real)));

  HYPRE_Real *tmp = tcalloc(HYPRE_Real, nr);
  for (unsigned i = 0; i < nr; i++)
    tmp[i] = r[i];

  chk_rt(gpuDeviceSynchronize());
  timer_log(3, 0);
  chk_rt(gpuMemcpy(d_r, tmp, nr * sizeof(HYPRE_Real), gpuMemcpyHostToDevice));
  chk_rt(gpuDeviceSynchronize());
  timer_log(3, 1);

  HYPRE_IJVectorUpdateValues(B->x, nr, NULL, d_x, 1);
  HYPRE_IJVectorUpdateValues(B->b, nr, NULL, d_r, 1);

  HYPRE_ParVector par_x, par_b;
  HYPRE_IJVectorGetObject(B->x, (void **)&par_x);
  HYPRE_IJVectorGetObject(B->b, (void **)&par_b);

  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJMatrixGetObject(B->A, (void **)&par_A);

  // HYPRE_IJVectorPrint(B->b, "b.dat");
  // HYPRE_IJVectorPrint(B->x, "x.dat");

  if (cb->verbose > 1) {
    HYPRE_BoomerAMGSetPrintLevel(solver, 2);
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
  }

  // Warmup
  HYPRE_BoomerAMGSetPrintLevel(solver, 0);
  for (unsigned i = 0; i < cb->trials; i++)
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);

  // Time the solve
  for (unsigned i = 0; i < cb->trials; i++) {
    chk_rt(gpuDeviceSynchronize());
    timer_log(4, 0);
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
    chk_rt(gpuDeviceSynchronize());
    timer_log(4, 1);
  }

  HYPRE_IJVectorGetValues(B->x, nr, NULL, d_x);

  chk_rt(gpuDeviceSynchronize());
  timer_log(5, 0);
  chk_rt(gpuMemcpy(tmp, d_x, nr * sizeof(HYPRE_Real), gpuMemcpyDeviceToHost));
  chk_rt(gpuDeviceSynchronize());
  timer_log(5, 1);

  chk_rt(gpuFree((void *)d_r));
  chk_rt(gpuFree((void *)d_x));

  for (unsigned i = 0; i < nr; i++)
    x[i] = tmp[i];

  csr_finalize(B), tfree(tmp);
  return 0;
}

#undef chk_rt
#undef csr_init
#undef hypre_bench
