#ifndef GMRESMILU_CGS_H
#define GMRESMILU_CGS_H
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "rtwtypes.h"
#include "gmresMILU_CGS_types.h"

extern void gmresMILU_CGS(const struct0_T *A, const emxArray_real_T *b, const
  emxArray_struct1_T *M, int restart, double rtol, int maxit, const
  emxArray_real_T *x0, int verbose, int nthreads, emxArray_real_T *x, int *flag,
  int *iter, emxArray_real_T *resids);
extern void gmresMILU_CGS_initialize(void);
extern void gmresMILU_CGS_terminate(void);

#endif
