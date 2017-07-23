#ifndef BICGSTABMILU_KERNEL_H
#define BICGSTABMILU_KERNEL_H
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "rtwtypes.h"
#include "bicgstabMILU_kernel_types.h"

extern void bicgstabMILU_kernel(const struct0_T *A, const emxArray_real_T *b,
  const emxArray_struct1_T *M, double rtol, int maxit, const emxArray_real_T *x0,
  int verbose, int nthreads, emxArray_real_T *x, int *flag, int *iter,
  emxArray_real_T *resids);
extern void bicgstabMILU_kernel_initialize(void);
extern void bicgstabMILU_kernel_terminate(void);

#endif
