#ifndef FGMRESMILU_CGS_H
#define FGMRESMILU_CGS_H
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "rtwtypes.h"
#include "fgmresMILU_CGS_types.h"

extern void fgmresMILU_CGS(const struct0_T *A, const emxArray_real_T *b, const
  struct1_T *prec, int restart, double rtol, int maxit, const emxArray_real_T
  *x0, int verbose, int nthreads, const struct1_T *param, const emxArray_real_T *
  rowscal, const emxArray_real_T *colscal, emxArray_real_T *x, int *flag, int
  *iter, emxArray_real_T *resids);
extern void fgmresMILU_CGS_initialize(void);
extern void fgmresMILU_CGS_terminate(void);

#endif
