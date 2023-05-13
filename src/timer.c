#include "lsbench-impl.h"
#include <float.h>
#include <math.h>

#define NTIMER 100

struct my_timer {
  double tmin, tmax, tsum;
  clock_t tic;
  int ncalls;
};

static struct my_timer timer[NTIMER];
static char* tags[NTIMER/5];
static int stack = 0;

void timer_init() {
  stack = 0;
  for (int i = 1; i < NTIMER; i++) {
    timer[i].tmin = DBL_MAX;
    timer[i].tmax = -DBL_MIN;
    timer[i].tsum = 0.0;
    timer[i].ncalls = 0;
  }
}

void timer_push(const char *tag) {
  tags[stack] = strndup(tag, BUFSIZ);
  stack++;
}

// TODO: add sync in this routine.
void timer_log(const int id, const int mode) {
  // tic
  if (mode == 0) {
    timer[5 * stack + id].tic = clock();
    return;
  }

  // toc
  clock_t t = clock() - timer[id].tic;
  double tsec = (double)t / CLOCKS_PER_SEC;
  timer[5 * stack + id].tmin = fmin(timer[id].tmin, tsec);
  timer[5 * stack + id].tmax = fmax(timer[id].tmax, tsec);
  timer[5 * stack + id].tsum += tsec;
  timer[5 * stack + id].ncalls++;
}

void timer_print_line(const int i) {
  double tave = 0.0;
  if (timer[i].ncalls > 0)
    tave = timer[i].tsum / (double)timer[i].ncalls;

  printf("%9.2e %9.2e %9.2e   %3d %9.2e\n", timer[i].tmin, timer[i].tmax,
         timer[i].tsum, timer[i].ncalls, tave);
}

void timer_print(int verbose) {
  /*
    0: elapsed
    1: library init
    2: setup
    3: host to device
    4: solve
    5: device to host
  */
  if (verbose == 0)
    return;

  for (int i = 1; i < NTIMER; i++) {
    if (timer[i].ncalls == 0) {
      timer[i].tmin = 0.0;
      timer[i].tmax = 0.0;
      timer[i].tsum = 0.0;
    }
  }

  printf("Runtime Statistics    (min / max / sum)        ncall  tave \n");
  printf("Library Init  :");
  timer_print_line(1);
  for (int i = 0; i < stack; i++) {
    printf("%s:\n", tags[i]);
    printf("  Solver Setup:");
    timer_print_line(5 * i + 2);
    printf("  Solver Solve:");
    timer_print_line(5 * i + 4);
    printf("  HostToDevice:");
    timer_print_line(5 * i + 3);
    printf("  DeviceToHost:");
    timer_print_line(5 * i + 5);
    printf("\n");
  }
}

#undef NTIMER
