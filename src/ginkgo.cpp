#include "lsbench-impl.h"

#if defined(LSBENCH_GINKGO)
#include <ginkgo/ginkgo.hpp>

//Ghttps://stackoverflow.com/questions/23266391/find-out-the-type-of-auto
std::string demangled(std::string const& sym) {
    std::unique_ptr<char, void(*)(void*)>
        name{abi::__cxa_demangle(sym.c_str(), nullptr, nullptr, nullptr), std::free};
    return {name.get()};
}

template <typename ValueType>
void print_vector(const std::string &name,
                  const gko::matrix::Dense<ValueType> *vec) {
  std::cout << name << " = [" << std::endl;
  for (int i = 0; i < vec->get_size()[0]; ++i) {
    std::cout << "    " << vec->at(i, 0) << std::endl;
  }
  std::cout << "];" << std::endl;
}

template <typename TD, typename TI>
static std::shared_ptr<gko::matrix::Csr<TD, TI>>
csr_init(struct csr *A, const struct lsbench *cb) {
  auto exec = gko::HipExecutor::create(0, gko::OmpExecutor::create());

  unsigned m = A->nrows;
  unsigned nnz = A->offs[m];
  auto ginkgo_csr_host = gko::matrix::Csr<TD, TI>::create(
      exec->get_master(), gko::dim<2>{m, m}, nnz);
  // unsigned -> int since ginkgo also likes ints.
  for (unsigned i = 0; i < m + 1; i++)
    ginkgo_csr_host->get_row_ptrs()[i] = A->offs[i];

  for (unsigned i = 0; i < nnz; i++)
    ginkgo_csr_host->get_col_idxs()[i] = A->cols[i] - A->base;

  for (unsigned i = 0; i < nnz; i++)
    ginkgo_csr_host->get_values()[i] = A->vals[i];

  auto ginkgo_csr = gko::share(ginkgo_csr_host->clone(exec));

  return ginkgo_csr;
}

template <typename TD, typename TI>
void ginkgo_bench_run(const char* str_solver, 
                      std::shared_ptr<gko::LinOpFactory> solver_gen, 
                      std::shared_ptr<gko::matrix::Csr<TD, TI>> B,
                      double *x, struct csr *A, const double *r,
                      const struct lsbench *cb) {

  // Some shortcuts
  using ValueType = TD;
  using RealValueType = gko::remove_complex<ValueType>;
  using IndexType = TI;
  using vec = gko::matrix::Dense<ValueType>;
  using real_vec = gko::matrix::Dense<RealValueType>;
  using view_arr = gko::array<ValueType>;

  unsigned m = A->nrows, nnz = A->offs[m];
  TD *t_r = tcalloc(TD, m);
  TD *t_x = tcalloc(TD, m);
  for (unsigned i = 0; i < m; i++) {
    t_x[i] = (TD) x[i];
    t_r[i] = (TD) r[i];
  }

  auto exec = B->get_executor();
  auto r_view = view_arr::const_view(exec->get_master(), m, t_r);
  auto x_view = view_arr::view(exec->get_master(), m, t_x);
  auto dense_x_host = vec::create(exec, gko::dim<2>{m, 1}, std::move(x_view), 1);
  auto dense_r_host = vec::create_const(exec, gko::dim<2>{m, 1}, std::move(r_view), 1);

  // Copy rhs and init_guess to Device
  exec->synchronize();
  timer_log(3, 0);
  auto dense_r = dense_r_host->clone(exec);
  auto dense_x_init = dense_x_host->clone(exec);
  auto dense_x = dense_x_init->clone();
  exec->synchronize();
  timer_log(3, 1);

  auto solver = solver_gen->generate(B);

  // Verbose
  if (cb->verbose>1) {
    // This adds a simple logger that only reports convergence state at the end
    // of the solver. Specifically it captures the last residual norm, the
    // final number of iterations, and the converged or not converged status.
    std::shared_ptr<gko::log::Convergence<ValueType>> convergence_logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(convergence_logger);

    auto dense_e = dense_r_host->clone(exec);

    solver->apply(dense_e, dense_x);

    // extract value from logger
//    auto res = gko::as<vec>(convergence_logger->get_residual_norm());

    // recompute res
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    B->apply(one, dense_x, neg_one, dense_e);
    dense_e->compute_norm2(res);

    // copy res back to host
    auto res_host = exec->copy_val_to_host(res->get_const_values());

    printf("%s residual sqrt(r^T r): %14.4e, iterations: %6d (converge: %s)\n",
           str_solver,
           res_host,
           convergence_logger->get_num_iterations(),
           convergence_logger->has_converged() ? "true" : "false");

    printf("===solver,matrix,n,nnz,trials,solver,ordering===\n");
    printf("%s,%s,%u,%u,%u,%u,%d\n", str_solver, cb->matrix, m, nnz, cb->trials,
           cb->solver, cb->ordering);
    fflush(stdout);

    solver->remove_logger(convergence_logger);
  }

  // Warmup
  for (unsigned i = 0; i < cb->trials; i++) {
    dense_x->copy_from(dense_x_init);
    solver->apply(dense_r, dense_x);
  }

  // Solve
  for (unsigned i = 0; i < cb->trials; i++) {
    dense_x->copy_from(dense_x_init);

    exec->synchronize();
    timer_log(4, 0);

    solver->apply(dense_r, dense_x);

    exec->synchronize();
    timer_log(4, 1);
  }

  // Copy sol back to Host
  exec->synchronize();
  timer_log(5, 0);
  dense_x_host->copy_from(dense_x);
  exec->synchronize();
  timer_log(5, 1);

  for (unsigned i = 0; i < m; i++)
    x[i] = (double) dense_x_host->get_values()[i];

  timer_push(str_solver);
}



template <typename TD, typename TI>
static int ginkgo_bench_aux(double *x, struct csr *A, const double *r,
                            const struct lsbench *cb) {
  // Some shortcuts 
  using ValueType = TD;
  using RealValueType = gko::remove_complex<ValueType>;
  using IndexType = TI;
  using vec = gko::matrix::Dense<ValueType>;
  using real_vec = gko::matrix::Dense<RealValueType>;
  using view_arr = gko::array<ValueType>;

  auto B = csr_init<TD,TI>(A, cb);
  auto exec = B->get_executor();

  // BiCGSTAB + Jacobi
  unsigned maxit = 1000;
  ValueType rel_tol=1e-6;
  exec->synchronize();
  timer_log(2, 0);
  std::shared_ptr<gko::LinOpFactory> solver_gen = 
      gko::solver::Bicgstab<ValueType>::build()
          .with_preconditioner(
              gko::preconditioner::Jacobi<ValueType>::build().on(exec))
          .with_criteria(gko::stop::Iteration::build().with_max_iters(maxit).on(exec),
                         gko::stop::ImplicitResidualNorm<ValueType>::build() 
                             .with_baseline(gko::stop::mode::initial_resnorm)
                             .with_reduction_factor(rel_tol)
                             .on(exec))
          .on(exec);
  exec->synchronize();
  timer_log(2, 1);

  ginkgo_bench_run<TD, TI>("Ginkgo BiCGSTAB+Jacobi", solver_gen, B, x, A, r, cb);


  // Direct solver
  exec->synchronize();
  timer_log(2, 0);
  std::shared_ptr<gko::LinOpFactory> direct_factory =
      gko::experimental::solver::Direct<ValueType, IndexType>::build()
          .with_factorization(
              gko::experimental::factorization::Lu<ValueType, IndexType>::build()
                  .on(exec))
          .on(exec);
  exec->synchronize();
  timer_log(2, 1);

  ginkgo_bench_run<TD, TI>("Ginkgo Direct", direct_factory, B, x, A, r, cb);


/*
  // from example ilu-preconditioned-solver/
  // GMRES + ILU FIXME: NOT CONVERGING!!
  // Generate incomplete factors using ParILU
  exec->synchronize();
  timer_log(2, 0);
  auto par_ilu_fact =
      gko::factorization::ParIlu<ValueType, IndexType>::build().on(exec);
  // Generate concrete factorization for input matrix
  auto par_ilu = gko::share(par_ilu_fact->generate(B));
  
  // Generate an ILU preconditioner factory by setting lower and upper
  // triangular solver - in this case the exact triangular solves
  auto ilu_pre_factory =
      gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
                               gko::solver::UpperTrs<ValueType, IndexType>,
                               false>::build()
          .on(exec);
  
  // Use incomplete factors to generate ILU preconditioner
  auto ilu_preconditioner = gko::share(ilu_pre_factory->generate(par_ilu));
  
  // Use preconditioner inside GMRES solver factory
  // Generating a solver factory tied to a specific preconditioner makes sense
  // if there are several very similar systems to solve, and the same
  // solver+preconditioner combination is expected to be effective.
  std::shared_ptr<gko::LinOpFactory> ilu_gmres_factory =
      gko::solver::Gmres<ValueType>::build()
          .with_criteria(
              gko::stop::Iteration::build().with_max_iters(maxit).on(exec),
              gko::stop::ResidualNorm<ValueType>::build()
                  .with_reduction_factor(rel_tol)
                  .on(exec))
          .with_generated_preconditioner(ilu_preconditioner)
          .on(exec);
  exec->synchronize();
  timer_log(2, 1);
  ginkgo_bench_run<TD, TI>("Ginkgo GMRES+ILU", ilu_gmres_factory, B, x, A, r, cb);
*/

/* 
  // from example ir-ilu-preconditioned-solver/
  // FIXME: this takes 10k iter, poor conv.
  // GMRES, precond with ILU where ILU's inverse is solved by few iterations of IR + Jacobi
  unsigned sweeps = 5;

  // Generate incomplete factors using ParILU
  exec->synchronize();
  timer_log(2, 0);
  auto par_ilu_fact =
      gko::factorization::ParIlu<ValueType, IndexType>::build().on(exec);
  // Generate concrete factorization for input matrix
  auto par_ilu = gko::share(par_ilu_fact->generate(B));

  auto bj_factory = gko::share(
      gko::preconditioner::Jacobi<ValueType, IndexType>::build()
          .on(exec));
  auto trisolve_factory =
      gko::solver::Ir<ValueType>::build()
          .with_solver(bj_factory)
          .with_criteria(
              gko::stop::Iteration::build().with_max_iters(sweeps).on(exec))
          .on(exec);
  auto ilu_pre_factory =
      gko::preconditioner::Ilu<gko::solver::Ir<ValueType>, gko::solver::Ir<ValueType>>::build()
          .with_l_solver_factory(gko::clone(trisolve_factory))
          .with_u_solver_factory(gko::clone(trisolve_factory))
          .on(exec);
  // Use incomplete factors to generate ILU preconditioner
  auto ilu_preconditioner = gko::share(ilu_pre_factory->generate(par_ilu));

  // Use preconditioner inside GMRES solver factory
  // Generating a solver factory tied to a specific preconditioner makes sense
  // if there are several very similar systems to solve, and the same
  // solver+preconditioner combination is expected to be effective.
  std::shared_ptr<gko::LinOpFactory> ilu_gmres_factory =
      gko::solver::Gmres<ValueType>::build()
          .with_criteria(
              gko::stop::Iteration::build().with_max_iters(maxit).on(exec),
              gko::stop::ResidualNorm<ValueType>::build()
                  .with_reduction_factor(rel_tol)
                  .on(exec))
          .with_generated_preconditioner(ilu_preconditioner)
          .on(exec);
  exec->synchronize();
  timer_log(2, 1);

  ginkgo_bench_run<TD, TI>("Ginkgo GMRES M=ILU, Minv=3 IR+Jacobi", ilu_gmres_factory, B, x, A, r, cb);
*/

/*
  //PCG + AMG //FIXME poor convergence, from example multigrid-preconditioned-solver/
  exec->synchronize();
  timer_log(2, 0);
  std::shared_ptr<gko::LinOpFactory> multigrid_gen =
      gko::solver::Multigrid::build()
          .with_mg_level(gko::multigrid::Pgm<ValueType,IndexType>::build()
              .with_deterministic(true).on(exec))
          .with_criteria(
              gko::stop::Iteration::build().with_max_iters(1u).on(exec))
          .on(exec);
  std::shared_ptr<gko::LinOpFactory> pcg_amg_factory =
      gko::solver::Cg<ValueType>::build()
          .with_criteria(
              gko::stop::Iteration::build().with_max_iters(maxit).on(exec),
              gko::stop::ResidualNorm<ValueType>::build()
                  .with_reduction_factor(rel_tol)
                  .on(exec))
          .with_preconditioner(multigrid_gen)
          .on(exec);
  exec->synchronize();
  timer_log(2, 1);

  ginkgo_bench_run<TD, TI>("Ginkgo PCG+AMG", pcg_amg_factory, B, x, A, r, cb);
*/

  // PCG + AMG 2, from multigrid-preconditioned-solver-customized
  exec->synchronize();
  timer_log(2, 0);
  ValueType abs_tol=1e-8;
  auto ic_gen = gko::share(
      gko::preconditioner::Ic<gko::solver::LowerTrs<ValueType>>::build()
          .with_factorization_factory(
              gko::factorization::Ic<ValueType, int>::build().on(exec))
          .on(exec)); 
  auto smoother_gen = gko::share(
      gko::solver::build_smoother(ic_gen, 2u, static_cast<ValueType>(0.9)));
  // Use Pgm as the MultigridLevel factory.
  auto mg_level_gen = gko::share(
      gko::multigrid::Pgm<ValueType, IndexType>::build()
          .with_deterministic(true)
          .on(exec));
  // Next we select a CG solver for the coarsest level. Again, since the input
  // matrix is known to be spd, and the Pgm restriction preserves this
  // characteristic, we can safely choose the CG. We reuse the Ic factory here
  // to generate an Ic preconditioner. It is important to solve until machine
  // precision here to get a good convergence rate.
  auto coarsest_gen = gko::share(
      gko::solver::Cg<ValueType>::build()
          .with_preconditioner(ic_gen)
          .with_criteria(
              gko::stop::Iteration::build().with_max_iters(10u).on(exec),
              gko::stop::ResidualNorm<ValueType>::build()
                  .with_baseline(gko::stop::mode::rhs_norm)
                  .with_reduction_factor(abs_tol)
                  .on(exec))
          .on(exec));
  // Here we put the customized options together and create the multigrid
  // factory.
  std::shared_ptr<gko::LinOpFactory> multigrid_gen =
      gko::solver::Multigrid::build()
          .with_max_levels(10u)
          .with_min_coarse_rows(32u)
          .with_pre_smoother(smoother_gen)
          .with_post_uses_pre(true)
          .with_mg_level(mg_level_gen)
          .with_coarsest_solver(coarsest_gen)
          .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
          .with_criteria(
              gko::stop::Iteration::build().with_max_iters(1u).on(exec))
          .on(exec);
  // Create solver factory
  std::shared_ptr<gko::LinOpFactory> pcg_amg2_factory =
      gko::solver::Cg<ValueType>::build()
          .with_criteria(
              gko::stop::Iteration::build().with_max_iters(maxit).on(exec),
              gko::stop::ResidualNorm<ValueType>::build()
                  .with_reduction_factor(rel_tol)
                  .on(exec))
          .with_preconditioner(multigrid_gen)
          .on(exec);
  exec->synchronize();
  timer_log(2, 1);

  ginkgo_bench_run<TD, TI>("Ginkgo PCG+AMG2", pcg_amg2_factory, B, x, A, r, cb);

  return 0;
}



int ginkgo_bench(double *x, struct csr *A, const double *r,
                     const struct lsbench *cb) {
  size_t prec, prec_int;
  int ret;
  if (cb->precision == LSBENCH_PRECISION_FP64) {
    prec = sizeof(double);
    prec_int = sizeof(int);
    ret = ginkgo_bench_aux<double,int>(x, A, r, cb);
  } else if (cb->precision == LSBENCH_PRECISION_FP32) {
    prec = sizeof(float);
    prec_int = sizeof(int);
    ret = ginkgo_bench_aux<float,int>(x, A, r, cb);
  } else {
    errx(EXIT_FAILURE, "Requsted Precisions not supported !");
    return 1;
  }
  
  if (cb->verbose > 0) {
    printf("Precision: %d bytes (matrix index: %d bytes).\n", prec,prec_int);
    fflush(stdout);
  }
    
  return ret;
}

#else
int ginkgo_bench(double *x, struct csr *A, const double *r,
                 const struct lsbench *cb) {
  return 1;
}
#endif
