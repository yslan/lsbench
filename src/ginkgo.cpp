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

static std::shared_ptr<gko::matrix::Csr<double, int>>
csr_init(struct csr *A, const struct lsbench *cb) {
  auto exec = gko::HipExecutor::create(0, gko::OmpExecutor::create());

  unsigned m = A->nrows;
  unsigned nnz = A->offs[m];
  auto ginkgo_csr_host = gko::matrix::Csr<double, int>::create(
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

int ginkgo_bench(double *x, struct csr *A, const double *r,
                 const struct lsbench *cb) {

  // Some shortcuts
  using ValueType = double;
  using RealValueType = gko::remove_complex<ValueType>;
  using IndexType = int;
  using vec = gko::matrix::Dense<ValueType>;
  using real_vec = gko::matrix::Dense<RealValueType>;
  using view_arr = gko::array<ValueType>;
  using mtx = gko::matrix::Csr<ValueType, IndexType>;
  using cg = gko::solver::Cg<ValueType>;

  unsigned m = A->nrows, nnz = A->offs[m];
  auto B = csr_init(A, cb);
  auto exec = B->get_executor();
  auto r_view = view_arr::const_view(exec->get_master(), m, r);
  auto x_view = view_arr::view(exec->get_master(), m, x);
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

  unsigned maxit = 100;
  double rel_tol=1e-6;

  auto solver =
      gko::solver::Bicgstab<ValueType>::build()
          .with_preconditioner(
              gko::preconditioner::Jacobi<ValueType>::build().on(exec))
          .with_criteria(gko::stop::Iteration::build().with_max_iters(maxit).on(exec),
                         gko::stop::ImplicitResidualNorm<ValueType>::build()
                             .with_baseline(gko::stop::mode::initial_resnorm)
                             .with_reduction_factor(rel_tol)
                             .on(exec))
          .on(exec)
          ->generate(B);


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

    printf("Ginkgo residual sqrt(r^T r): %14.4e, iterations: %d (converge: %s)\n",
           res_host,
           convergence_logger->get_num_iterations(),
           convergence_logger->has_converged() ? "true" : "false");

    printf("===matrix,n,nnz,trials,solver,ordering===\n");
    printf("%s,%u,%u,%u,%u,%d,%.15lf\n", cb->matrix, m, nnz, cb->trials,
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
    x[i] = dense_x_host->get_values()[i];

  timer_push("Ginkgo BiCGSTAB+Jacobi");

  return 0;
}

#else
int ginkgo_bench(double *x, struct csr *A, const double *r,
                 const struct lsbench *cb) {
  return 1;
}
#endif
