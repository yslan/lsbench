#include "lsbench-impl.h"
#include <ctype.h>
#include <err.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>

static void str_to_upper(char *up, const char *str) {
  size_t n = strnlen(str, BUFSIZ);
  for (unsigned i = 0; i < n; i++)
    up[i] = toupper(str[i]);
  up[n] = '\0';
}

static lsbench_solver_t str_to_solver(const char *str) {
  char up[BUFSIZ];
  str_to_upper(up, str);

  if (strcmp(up, "CUSOLVER") == 0) {
    return LSBENCH_SOLVER_CUSOLVER;
  } else if (strcmp(up, "HYPRE") == 0) {
    return LSBENCH_SOLVER_HYPRE;
  } else if (strcmp(up, "AMGX") == 0) {
    return LSBENCH_SOLVER_AMGX;
  } else if (strcmp(up, "CHOLMOD") == 0) {
    return LSBENCH_SOLVER_CHOLMOD;
  } else if (strcmp(up, "PARALMOND") == 0) {
    return LSBENCH_SOLVER_PARALMOND;
  } else if (strcmp(up, "ROCALUTION") == 0) {
    return LSBENCH_SOLVER_ROCALUTION;
  } else if (strcmp(up, "GINKGO") == 0) {
    return LSBENCH_SOLVER_GINKGO;
  } else {
    errx(EXIT_FAILURE, "Invalid solver: \"%s\".", str);
  }
  return LSBENCH_SOLVER_NONE;
}

static lsbench_ordering_t str_to_ordering(const char *str) {
  char up[BUFSIZ];
  str_to_upper(up, str);

  if (strcmp(up, "RCM") == 0) {
    return LSBENCH_ORDERING_RCM;
  } else if (strcmp(up, "AMD") == 0) {
    return LSBENCH_ORDERING_AMD;
  } else if (strcmp(up, "METIS") == 0) {
    return LSBENCH_ORDERING_METIS;
  } else {
    errx(EXIT_FAILURE, "Invalid ordering: \"%s\".", str);
    return LSBENCH_ORDERING_NONE;
  }
}

static lsbench_precision_t str_to_precision(const char *str) {
  char up[BUFSIZ];
  str_to_upper(up, str);

  if (strcmp(up, "FP64") == 0) {
    return LSBENCH_PRECISION_FP64;
  } else if (strcmp(up, "FP32") == 0) {
    return LSBENCH_PRECISION_FP32;
  } else if (strcmp(up, "FP16") == 0) {
    return LSBENCH_PRECISION_FP16;
  } else {
    errx(EXIT_FAILURE, "Invalid precision: \"%s\".");
    return LSBENCH_PRECISION_NONE;
  }
}

static void print_help(int argc, char *argv[]) {
  printf("Usage: %s [OPTIONS]\n", "./driver");
  printf("Options:\n");
  printf("  --matrix <FILE>\n");
  printf("  --solver <SOLVER>, Values: rocalution, cusolver, hypre, amgx, "
         "cholmod, ginkgo\n");
  printf("  --ordering <ORDERING>, Values: RCM, AMD, METIS\n");
  printf("  --precision <PRECISION>, Values: FP64, FP32, FP16\n");
  printf("  --verbose <VERBOSITY>, Values: 0, 1, 2, ...\n");
  printf("  --trials <TRIALS>, Values: 1, 2, ...\n");
  printf("  --help\n");
}

struct lsbench *lsbench_init(int argc, char *argv[]) {
  // Supported command line options.
  static struct option long_options[] = {
      {"matrix", required_argument, 0, 10},
      {"solver", required_argument, 0, 20},
      {"ordering", optional_argument, 0, 30},
      {"precision", optional_argument, 0, 40},
      {"verbose", optional_argument, 0, 50},
      {"trials", optional_argument, 0, 60},
      {"help", no_argument, 0, 70},
      {0, 0, 0, 0}};

  // Create the struct and set defauls.
  struct lsbench *cb = tcalloc(struct lsbench, 1);
  cb->matrix = NULL, cb->verbose = 0, cb->trials = 100;
  cb->precision = LSBENCH_PRECISION_FP32;

  // Parse the command line arguments.
  char bfr[BUFSIZ];
  for (;;) {
    int c = getopt_long(argc, argv, "", long_options, NULL);
    if (c == -1)
      break;

    switch (c) {
    case 10:
      cb->matrix = strndup(optarg, BUFSIZ);
      break;
    case 20:
      cb->solver_str = strndup(optarg, BUFSIZ);
      cb->solver = str_to_solver(cb->solver_str);
      break;
    case 30:
      strncpy(bfr, optarg, BUFSIZ);
      cb->ordering = str_to_ordering(bfr);
      break;
    case 40:
      strncpy(bfr, optarg, BUFSIZ);
      cb->precision = str_to_precision(bfr);
      break;
    case 50:
      cb->verbose = atoi(optarg);
      break;
    case 60:
      cb->trials = atoi(optarg);
      break;
    case 70:
      print_help(argc, argv);
      exit(EXIT_SUCCESS);
      break;
    default:
      print_help(argc, argv);
      exit(EXIT_FAILURE);
      break;
    }
  }

  // Sanity checks and check for things which are not yet implemented.
  if (cb->matrix == NULL)
    errx(EXIT_FAILURE, "Input matrix file not provided. Try `--help`.");

  timer_init(2);

  return cb;
}

const char *lsbench_get_matrix_name(struct lsbench *cb) {
  return (const char *)cb->matrix;
}

void lsbench_bench(struct csr *A, const struct lsbench *cb) {
  unsigned m = A->nrows;
  double *r = tcalloc(double, m);
  double *x = tcalloc(double, m);

  if (cb->verbose > 1) {
    printf("matrix size nr = %d   nnz = %d \n", m, A->offs[m]);
    fflush(stdout);
  }

  // Initialize rhs and make ||x|| = 1.
  int seed = 27; // seed of the random
  srand(seed);
  for (unsigned i = 0; i < m; i++)
    r[i] = (double)rand() / RAND_MAX;
  double tmp = l2norm(r, m);
  for (unsigned i = 0; i < m; i++)
    r[i] = r[i] / tmp;
  if (cb->verbose > 1) {
    tmp = l2norm(r, m);
    printf("rhs initialized by rand() with seed = %d, RAND_MAX = %d and norm = "
           "%lf\n",
           seed, RAND_MAX, tmp);
    fflush(stdout);
  }

  switch (cb->solver) {
  case LSBENCH_SOLVER_CUSOLVER:
    timer_log(1, 0);
    cusparse_init();
    timer_log(1, 1);
    cusparse_bench(x, A, r, cb);
    cusparse_finalize();
    break;
  case LSBENCH_SOLVER_HYPRE:
    timer_log(1, 0);
    hypre_init();
    timer_log(1, 1);
    hypre_bench(x, A, r, cb);
    hypre_finalize();
    break;
  case LSBENCH_SOLVER_AMGX:
    timer_log(1, 0);
    amgx_init();
    timer_log(1, 1);
    amgx_bench(x, A, r, cb);
    amgx_finalize();
    break;
  case LSBENCH_SOLVER_CHOLMOD:
    timer_log(1, 0);
    cholmod_init();
    timer_log(1, 1);
    cholmod_bench(x, A, r, cb);
    cholmod_finalize();
    break;
  case LSBENCH_SOLVER_PARALMOND:
    timer_log(1, 0);
    paralmond_init();
    timer_log(1, 1);
    paralmond_bench(x, A, r, cb);
    paralmond_finalize();
    break;
  case LSBENCH_SOLVER_ROCALUTION:
    timer_log(1, 0);
    rocalution_init();
    timer_log(1, 1);
    rocalution_bench(x, A, r, cb);
    rocalution_finalize();
    break;
  case LSBENCH_SOLVER_GINKGO:
    timer_log(1, 0);
    timer_log(1, 1);
    ginkgo_bench(x, A, r, cb);
    break;
  default:
    errx(EXIT_FAILURE, "Unknown solver: %d.", cb->solver);
    break;
  }

  if (cb->verbose) {
    double *rd = tcalloc(double, m);
    for (unsigned i = 0; i < m; i++)
      rd[i] = r[i];
    csr_spmv(-1.0, A, x, 1.0, rd);

    if (cb->verbose > 1) {
      printf("x   (min/max/amax)  %14.4e %14.4e %14.4e \n", glmin(x, m),
             glmax(x, m), glamax(x, m));
      printf("rhs (min/max/amax)  %14.4e %14.4e %14.4e \n", glmin(r, m),
             glmax(r, m), glamax(r, m));
      printf("res (min/max/amax)  %14.4e %14.4e %14.4e \n", glmin(rd, m),
             glmax(rd, m), glamax(rd, m));
    }
    printf("norm(b-Ax) = %14.4e norm(b) = %14.4e  norm(x) = %14.4e\n",
           l2norm(rd, m), l2norm(r, m), l2norm(x, m));

    printf("Solver: %s.\n", cb->solver_str);
    printf("Matrix: %s,  nrow = %d nnz = %d.\n", cb->matrix, A->nrows,
           A->offs[A->nrows]);
    fflush(stdout);
    tfree(rd);
  }

  tfree(x), tfree(r);
}

void lsbench_finalize(struct lsbench *cb) {
  timer_print(cb->verbose);

  if (cb) {
    tfree(cb->matrix);
    tfree(cb->solver_str);
  }
  tfree(cb);
}
