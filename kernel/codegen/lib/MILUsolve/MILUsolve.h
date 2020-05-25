#ifndef MILUSOLVE_H
#define MILUSOLVE_H
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "MILUsolve_types.h"

extern void MILUsolve(const emxArray_struct0_T *M, emxArray_real_T *b,
                      emxArray_real_T *b_y1, emxArray_real_T *y2);
extern void MILUsolve_2args(const emxArray_struct0_T *M, emxArray_real_T *b);
extern void MILUsolve_initialize(void);
extern void MILUsolve_terminate(void);

#endif
