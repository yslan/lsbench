#include "lsbench-impl.h"
#define NTIMER 10

struct my_timer {
  double tmin, tmax, tsum;
  int ncalls;
};


static struct my_timer _timer[NTIMER];
static clock_t _tic[NTIMER];

void timer_init() {
  for (int i=1; i<NTIMER; i++) {
    _timer[i].tmin = 1e10;
    _timer[i].tmax =-1e10;
    _timer[i].tsum = 0.0;
    _timer[i].ncalls = 0;
  }
}

void timer_log(const int id, const int mode) { // TODO add sync here
  // tic
  if (mode==0) {
    _tic[id] = clock();
    return;
  }

  // toc
  clock_t t = clock() - _tic[id];
  double tsec = (double) t / CLOCKS_PER_SEC;
  _timer[id].tmin = fmin( _timer[id].tmin, tsec );
  _timer[id].tmax = fmax( _timer[id].tmax, tsec );
  _timer[id].tsum += tsec;
  _timer[id].ncalls++;
}

void timer_print(int verbose){
/*
  0: elapsed
  1: library init
  2: setup
  3: host to device
  4: solve
  5: device to host
*/
  if (verbose==0) return;

  if (verbose>1) {
    printf("\n\nSummary (min/max/sum/ncall) \n");
    printf("  Lib Init.      %9.2e %9.2e %9.2e  %3d\n", 
           _timer[1].tmin,_timer[1].tmax,_timer[1].tsum,_timer[1].ncalls);
    printf("  Solver Setup   %9.2e %9.2e %9.2e  %3d\n",
           _timer[2].tmin,_timer[2].tmax,_timer[2].tsum,_timer[2].ncalls);
    printf("  Solver Solve   %9.2e %9.2e %9.2e  %3d\n", 
           _timer[5].tmin,_timer[5].tmax,_timer[5].tsum,_timer[5].ncalls);
  } else {

  }
}


