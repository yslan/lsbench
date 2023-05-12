#include "lsbench-impl.h"
#define NTIMER 10

struct my_timer {
  double tmin, tmax, tsum;
  int ncalls;
};

static struct my_timer _timer[NTIMER];
static clock_t _tic[NTIMER];

void timer_init() {
  for (int i = 1; i < NTIMER; i++) {
    _timer[i].tmin = 1e10;
    _timer[i].tmax = -1e10;
    _timer[i].tsum = 0.0;
    _timer[i].ncalls = 0;
  }
}

void timer_log(const int id, const int mode) { // TODO add sync here
  // tic
  if (mode == 0) {
    _tic[id] = clock();
    return;
  }

  // toc
  clock_t t = clock() - _tic[id];
  double tsec = (double)t / CLOCKS_PER_SEC;
  _timer[id].tmin = fmin(_timer[id].tmin, tsec);
  _timer[id].tmax = fmax(_timer[id].tmax, tsec);
  _timer[id].tsum += tsec;
  _timer[id].ncalls++;
}

void timer_print_line(const int i) {

  double tave = 0.0;
  if (_timer[i].ncalls > 0)
    tave = _timer[i].tsum / (double)_timer[i].ncalls;

  printf("%9.2e %9.2e %9.2e   %3d %9.2e\n", _timer[i].tmin, _timer[i].tmax,
         _timer[i].tsum, _timer[i].ncalls, tave);
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
    if (_timer[i].ncalls == 0) {
      _timer[i].tmin = 0.0;
      _timer[i].tmax = 0.0;
      _timer[i].tsum = 0.0;
    }
  }


  int i;
  printf("Runtime Statistics    (min / max / sum)        ncall  tave \n");
  printf("  Lib Init.      ");
  timer_print_line(1);
  printf("  Solver Setup   ");
  timer_print_line(2);
  printf("  Solver Solve   ");
  timer_print_line(4);
  printf("  HostToDevice   ");
  timer_print_line(3);
  printf("  DeviceToHost   ");
  timer_print_line(5);
  printf("\n\n");
}
