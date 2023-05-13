#include "lsbench-impl.h"
#include <math.h>

#define NPARAM 11

#if defined(LSBENCH_HYPRE)
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <_hypre_utilities.h>

static int initialized = 0;
//static HYPRE_Solver solver = 0, precond = 0;

struct hypre_csr {
  HYPRE_IJMatrix A;
  HYPRE_IJVector b, x;
};

static void csr_finalize(struct hypre_csr *A) {
  if (A) {
    HYPRE_IJMatrixDestroy(A->A);
    HYPRE_IJVectorDestroy(A->x);
    HYPRE_IJVectorDestroy(A->b);
  }
  tfree(A);
}

int hypre_init() {
  if (initialized)
    return 1;

  HYPRE_Init();
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

  // Settings for cuda
  HYPRE_Int spgemm_use_vendor = 0;
  HYPRE_SetSpGemmUseVendor(spgemm_use_vendor);
  HYPRE_SetSpMVUseVendor(1);

  HYPRE_SetUseGpuRand(1);

  initialized = 1;
  return 0;
}

HYPRE_Solver hypre_bench_AMG_setup(struct hypre_csr* B) {

  double params[NPARAM];
  params[0] = 8;    /* coarsening */
  params[1] = 6;    /* interpolation */
  params[2] = 2;    /* number of cycles */
  params[3] = 8;    /* smoother for crs level */
  params[4] = 3;    /* sweeps */
  params[5] = 8;    /* smoother */
  params[6] = 1;    /* sweeps   */
  params[7] = 0.25; /* threshold */
  params[8] = 0.00; /* non galerkin tolerance */
  params[9] = 0;    /* agressive coarsening */
  params[10] = 2;   /* chebyRelaxOrder */

  HYPRE_Solver solver=0;
  HYPRE_BoomerAMGCreate(&solver);

  HYPRE_BoomerAMGSetCoarsenType(solver, params[0]);
  HYPRE_BoomerAMGSetInterpType(solver, params[1]);

  HYPRE_BoomerAMGSetModuleRAP2(solver, 1);
  HYPRE_BoomerAMGSetKeepTranspose(solver, 1);

  // HYPRE_BoomerAMGSetChebyOrder(solver, params[10]);
  // HYPRE_BoomerAMGSetChebyFraction(*solver, 0.2);

  if (params[5] > 0) {
    HYPRE_BoomerAMGSetCycleRelaxType(solver, params[5], 1);
    HYPRE_BoomerAMGSetCycleRelaxType(solver, params[5], 2);
  }
  HYPRE_BoomerAMGSetCycleRelaxType(solver, 9, 3);

  HYPRE_BoomerAMGSetCycleNumSweeps(solver, params[6], 1);
  HYPRE_BoomerAMGSetCycleNumSweeps(solver, params[6], 2);
  HYPRE_BoomerAMGSetCycleNumSweeps(solver, 1, 3);

  int null_space = 0;
  if (null_space) {
    HYPRE_BoomerAMGSetMinCoarseSize(solver, 2);
    HYPRE_BoomerAMGSetCycleRelaxType(solver, params[3], 3);
    HYPRE_BoomerAMGSetCycleNumSweeps(solver, params[4], 3);
  }

  HYPRE_BoomerAMGSetStrongThreshold(solver, params[7]);

// NonGalerkin not supported yet
#if 0
  if (params[8] > 1e-3) {
    HYPRE_BoomerAMGSetNonGalerkinTol(*solver,params[8]);
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(*solver,0.0 , 0);
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(*solver,0.01, 1);
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(*solver,0.05, 2);
  }
#endif

  if (params[9] > 0) {
    HYPRE_BoomerAMGSetAggNumLevels(solver, params[9]);
    HYPRE_BoomerAMGSetAggInterpType(solver, 5);
    // HYPRE_BoomerAMGSetNumPaths(solver, 10);
  }

  HYPRE_BoomerAMGSetMaxIter(solver, params[2]);
  HYPRE_BoomerAMGSetTol(solver, 0);

  HYPRE_ParVector par_b, par_x;
  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJVectorGetObject(B->b, (void **)&par_b);
  HYPRE_IJVectorGetObject(B->x, (void **)&par_x);
  HYPRE_IJMatrixGetObject(B->A, (void **)&par_A);

  HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);

  return solver;
}

HYPRE_Solver hypre_bench_PCG_setup(struct hypre_csr* B, HYPRE_Solver precond) {

  int comm = 0;
  HYPRE_Solver solver=0;
  HYPRE_ParCSRPCGCreate(comm, &solver);

  HYPRE_ParCSRPCGSetMaxIter(solver, 1000); 
  HYPRE_ParCSRPCGSetTol(solver, 1e-6); 
  HYPRE_ParCSRPCGSetAbsoluteTol(solver, 1e-8); 
  HYPRE_ParCSRPCGSetTwoNorm(solver, 1); 
  HYPRE_ParCSRPCGSetLogging(solver, 1); // This is needed to extract final residual

  HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                      (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

  HYPRE_ParVector par_b, par_x;
  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJVectorGetObject(B->b, (void **)&par_b);
  HYPRE_IJVectorGetObject(B->x, (void **)&par_x);
  HYPRE_IJMatrixGetObject(B->A, (void **)&par_A);

  HYPRE_ParCSRPCGSetup(solver, par_A, par_b, par_x);

  return solver;
}

// hypre_bench for backends
#if defined(ENABLE_CUDA) // CUDA
#include <cuda_runtime.h>
#define GPU cuda
#define RUNTIME nvrtc
#include "hypre-impl.h"
#undef GPU
#undef RUNTIME

#elif defined(ENABLE_HIP) // HIP
#include <hip/hip_runtime.h>
#define GPU hip
#define RUNTIME hiprtc
#include "hypre-impl.h"
#undef GPU
#undef RUNTIME

#elif defined(ENABLE_DPCPP) // DPCPP (not implemented)
#else                       // CPU
static struct hypre_csr *cpu_csr_init(struct csr *A, const struct lsbench *cb) {

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

  HYPRE_IJMatrixSetValues(B->A, nr, ncols, rows, cols, vals);
  HYPRE_IJMatrixAssemble(B->A);
  // HYPRE_IJMatrixPrint(B->A, "A.dat");

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

  return B;
}

void cpu_hypre_bench_run(char* str_solver, HYPRE_Solver solver, 
                         HYPRE_Int (*PtrToFcn_Solve)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector),
                         HYPRE_Int (*PtrToFcn_PrintLevel)(HYPRE_Solver, HYPRE_Int),
                         struct hypre_csr* B,
                         double *x, struct csr *A, const double *r,
                         const struct lsbench *cb) {

  HYPRE_ParVector par_b, par_x;
  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJVectorGetObject(B->b, (void **)&par_b);
  HYPRE_IJVectorGetObject(B->x, (void **)&par_x);  
  HYPRE_IJMatrixGetObject(B->A, (void **)&par_A);

  unsigned nr = A->nrows, nnz = A->offs[nr];
  HYPRE_Real *d_x = tcalloc(HYPRE_Real, nr);;
  HYPRE_Real *d_r = tcalloc(HYPRE_Real, nr);
  for (unsigned i = 0; i < nr; i++) {
    x[i] = 0.0;
    d_r[i] = (HYPRE_Real) r[i];
    d_x[i] = (HYPRE_Real) 0.0;
  }

  HYPRE_IJVectorUpdateValues(B->x, nr, NULL, d_x, 1);
  HYPRE_IJVectorUpdateValues(B->b, nr, NULL, d_r, 1);

  if (cb->verbose > 1) {
//    HYPRE_BoomerAMGSetPrintLevel(solver, 3);
//    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
    PtrToFcn_PrintLevel(solver, 3);
    HYPRE_IJVectorUpdateValues(B->x, nr, NULL, d_x, 1);
    PtrToFcn_Solve(solver, par_A, par_b, par_x);


    int comm = 0;
    HYPRE_BigInt lower = A->base, upper = (HYPRE_BigInt)A->nrows - 1 + A->base;

    HYPRE_IJVector rd;
    HYPRE_IJVectorCreate(comm, lower, upper, &rd);
    HYPRE_IJVectorSetObjectType(rd, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(rd);
    HYPRE_IJVectorAssemble(rd);

    HYPRE_ParVector par_e;
    HYPRE_IJVectorGetObject(rd, (void **)&par_e);
    HYPRE_IJVectorUpdateValues(rd, nr, NULL, d_r, 1);

    HYPRE_Real norm;
    HYPRE_ParCSRMatrixMatvec(-1.0, par_A, par_x, 1.0, par_e); 
    HYPRE_ParVectorInnerProd(par_e, par_e, &norm);
    if (norm>0) norm = sqrt(norm);
    printf("%s norm(b-Ax) = %14.4e\n", str_solver, norm);
    fflush(stdout);

    HYPRE_IJVectorDestroy(rd);
  }

  // Warmup
//  HYPRE_BoomerAMGSetPrintLevel(solver, 0);
  PtrToFcn_PrintLevel(solver, 0);
  for (unsigned i = 0; i < cb->trials; i++) {
    HYPRE_IJVectorUpdateValues(B->x, nr, NULL, d_x, 1);
//    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
    PtrToFcn_Solve(solver, par_A, par_b, par_x);
  }

  // Time the solve
  for (unsigned i = 0; i < cb->trials; i++) {
    HYPRE_IJVectorUpdateValues(B->x, nr, NULL, d_x, 1);
    timer_log(4, 0);
    PtrToFcn_Solve(solver, par_A, par_b, par_x);
    timer_log(4, 1);
  }

  HYPRE_IJVectorGetValues(B->x, nr, NULL, d_x);
  for (unsigned i = 0; i < nr; i++)
    x[i] = (double) d_x[i];

  tfree(d_r);
  tfree(d_x);
}
#endif // hypre_bench for backends

void hypre_bench_run(char* str_solver, HYPRE_Solver solver,
                     HYPRE_Int (*PtrToFcn_Solve)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector),
                     HYPRE_Int (*PtrToFcn_PrintLevel)(HYPRE_Solver, HYPRE_Int),
                     struct hypre_csr* B,
                     double *x, struct csr *A, const double *r,
                     const struct lsbench *cb) {
#if defined(ENABLE_CUDA)
  cuda_hypre_bench_run(str_solver, solver, PtrToFcn_Solve, PtrToFcn_PrintLevel, B, x, A, r, cb);
#elif defined(ENABLE_HIP)
  hip_hypre_bench_run(str_solver, solver, PtrToFcn_Solve, PtrToFcn_PrintLevel, B, x, A, r, cb);
#elif defined(ENABLE_DPCPP)
  errx(EXIT_FAILURE, "hypre_bench_run not implemented !");
#else // CPU
  cpu_hypre_bench_run(str_solver, solver, PtrToFcn_Solve, PtrToFcn_PrintLevel, B, x, A, r, cb);
#endif
  timer_push(str_solver);
}

struct hypre_csr *csr_init(struct csr *A, const struct lsbench *cb) {
#if defined(ENABLE_CUDA)
  return cuda_csr_init(A, cb);
#elif defined(ENABLE_HIP)
  return hip_csr_init(A, cb);
#elif defined(ENABLE_DPCPP)
  errx(EXIT_FAILURE, "csr_init not implemented !");
  return 1;
#else // CPU
  return cpu_csr_init(A, cb);
#endif
}


int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {
  if ((cb->precision == LSBENCH_PRECISION_FP64 && sizeof(HYPRE_Real) != 8) ||
      (cb->precision == LSBENCH_PRECISION_FP32 && sizeof(HYPRE_Real) != 4)) {
    errx(EXIT_FAILURE, "Requsted Precisions not supported !");
    return 1;
  }
  if (cb->verbose > 0) {
    printf("Precision: %d bytes.\n", sizeof(HYPRE_Real));
    fflush(stdout);
  }
  if (!initialized) {
    errx(EXIT_FAILURE, "Hypre is not initialized !\n");
    return 1;
  }

  struct hypre_csr *B = csr_init(A, cb);
  HYPRE_Solver solver=0, precond=0;

  // Benchmark AMG.
  timer_log(2,0);
  solver = hypre_bench_AMG_setup(B);
  timer_log(2,1);
  hypre_bench_run("HYPRE AMG", solver, 
                  HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetPrintLevel, B, x, A, r, cb);
  HYPRE_BoomerAMGDestroy(solver);

  // Benchmark PCG + AMG
  timer_log(2,0);
  precond = hypre_bench_AMG_setup(B);
  solver = hypre_bench_PCG_setup(B, precond);
  timer_log(2,1);
  hypre_bench_run("HYPRE PCG+AMG", solver, 
                  HYPRE_ParCSRPCGSolve, HYPRE_ParCSRPCGSetPrintLevel, B, x, A, r, cb);

  if (cb->verbose>1) { // Run info - needed logging turned on
    int num_iterations=0;
    HYPRE_Real final_res_norm=0;

    HYPRE_PCGGetNumIterations(solver, &num_iterations);
    HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
    printf("Iterations = %d\n", num_iterations);
    printf("Final Relative Residual Norm = %e\n", final_res_norm);
    fflush(stdout);
  }

  HYPRE_ParCSRPCGDestroy(solver);
  HYPRE_BoomerAMGDestroy(precond);

  csr_finalize(B);

  return 0;
}

int hypre_finalize() {
  if (!initialized)
    return 1;

  HYPRE_Finalize();
  initialized = 0;
  return 0;
}

#else  // LSBENCH_HYPRE
int hypre_init() { return 1; }
int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {
  return 1;
}
int hypre_finalize() { return 1; }
#endif // LSBENCH_HYPRE

#undef NPARAM
