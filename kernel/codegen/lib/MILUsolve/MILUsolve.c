#include "MILUsolve.h"
#include "MILUsolve_emxutil.h"
#include "m2c.h"

static void crs_Axpy(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
                     *A_col_ind, const emxArray_real_T *A_val, int A_nrows,
                     const emxArray_real_T *x, emxArray_real_T *y);
static void crs_Axpy_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, emxArray_real_T *y, int nrows);
static void crs_solve_utril(const emxArray_int32_T *A_row_ptr, const
  emxArray_int32_T *A_col_ind, const emxArray_real_T *A_val, emxArray_real_T *b);
static void crs_solve_utriu(const emxArray_int32_T *A_row_ptr, const
  emxArray_int32_T *A_col_ind, const emxArray_real_T *A_val, emxArray_real_T *b);
static void m2c_error(void);
static void solve_milu(const emxArray_struct0_T *M, int lvl, emxArray_real_T *b,
  int offset, emxArray_real_T *b_y1, emxArray_real_T *y2);
static void crs_Axpy(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
                     *A_col_ind, const emxArray_real_T *A_val, int A_nrows,
                     const emxArray_real_T *x, emxArray_real_T *y)
{
  if (y->size[0] < A_nrows) {
    m2c_error();
  }

  crs_Axpy_kernel(A_row_ptr, A_col_ind, A_val, x, y, A_nrows);
}

static void crs_Axpy_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, emxArray_real_T *y, int nrows)
{
  int i;
  double t;
  int j;
  for (i = 0; i + 1 <= nrows; i++) {
    t = y->data[i];
    for (j = row_ptr->data[i]; j < row_ptr->data[i + 1]; j++) {
      t += val->data[j - 1] * x->data[col_ind->data[j - 1] - 1];
    }

    y->data[i] = t;
  }
}

static void crs_solve_utril(const emxArray_int32_T *A_row_ptr, const
  emxArray_int32_T *A_col_ind, const emxArray_real_T *A_val, emxArray_real_T *b)
{
  int n;
  int i;
  int k;
  n = A_row_ptr->size[0] - 1;
  for (i = 0; i + 1 <= n; i++) {
    for (k = A_row_ptr->data[i]; k < A_row_ptr->data[i + 1]; k++) {
      b->data[i] -= A_val->data[k - 1] * b->data[A_col_ind->data[k - 1] - 1];
    }
  }
}

static void crs_solve_utriu(const emxArray_int32_T *A_row_ptr, const
  emxArray_int32_T *A_col_ind, const emxArray_real_T *A_val, emxArray_real_T *b)
{
  int i;
  int cind;
  for (i = A_row_ptr->size[0] - 2; i + 1 > 0; i--) {
    for (cind = A_row_ptr->data[i]; cind < A_row_ptr->data[i + 1]; cind++) {
      b->data[i] -= A_val->data[cind - 1] * b->data[A_col_ind->data[cind - 1] -
        1];
    }
  }
}

static void m2c_error(void)
{
  const char * msgid;
  const char * fmt;
  msgid = "crs_Axpy:BufferTooSmal";
  fmt = "Buffer space for output y is too small.";
  M2C_error(msgid, fmt);
}

static void solve_milu(const emxArray_struct0_T *M, int lvl, emxArray_real_T *b,
  int offset, emxArray_real_T *b_y1, emxArray_real_T *y2)
{
  int nB;
  int n;
  int i;
  emxArray_real_T *c_y1;
  struct1_T expl_temp;
  int k;
  emxArray_int32_T *r0;
  int j;
  int i0;
  emxArray_real_T *r1;
  nB = M->data[lvl - 1].L.nrows;
  n = M->data[lvl - 1].L.nrows + M->data[lvl - 1].negE.nrows;
  for (i = 0; i + 1 <= nB; i++) {
    b_y1->data[i] = M->data[lvl - 1].rowscal->data[M->data[lvl - 1].p->data[i] -
      1] * b->data[(M->data[lvl - 1].p->data[i] + offset) - 1];
  }

  for (i = M->data[lvl - 1].L.nrows; i + 1 <= n; i++) {
    y2->data[i - nB] = M->data[lvl - 1].rowscal->data[M->data[lvl - 1].p->data[i]
      - 1] * b->data[(M->data[lvl - 1].p->data[i] + offset) - 1];
  }

  emxInit_real_T(&c_y1, 2);
  if (n > M->data[lvl - 1].L.nrows) {
    if (1 > M->data[lvl - 1].L.nrows) {
      k = 0;
    } else {
      k = M->data[lvl - 1].L.nrows;
    }

    emxInit_int32_T(&r0, 2);
    i = M->data[lvl - 1].L.nrows;
    i0 = r0->size[0] * r0->size[1];
    r0->size[0] = 1;
    r0->size[1] = i;
    emxEnsureCapacity_int32_T(r0, i0);
    for (i0 = 0; i0 < i; i0++) {
      r0->data[r0->size[0] * i0] = (i0 + offset) + 1;
    }

    i0 = c_y1->size[0] * c_y1->size[1];
    c_y1->size[0] = 1;
    c_y1->size[1] = k;
    emxEnsureCapacity_real_T(c_y1, i0);
    for (i0 = 0; i0 < k; i0++) {
      c_y1->data[c_y1->size[0] * i0] = b_y1->data[i0];
    }

    k = c_y1->size[1];
    for (i0 = 0; i0 < k; i0++) {
      b->data[r0->data[r0->size[0] * i0] - 1] = c_y1->data[c_y1->size[0] * i0];
    }

    emxFree_int32_T(&r0);
  }

  emxInitStruct_struct1_T(&expl_temp);
  if ((M->data[lvl - 1].L.val->size[0] == 0) && (M->data[lvl - 1].U.val->size[0]
       == n * n)) {
    k = 0;
    for (j = 1; j <= nB; j++) {
      k += j;
      for (i = j; i + 1 <= nB; i++) {
        b_y1->data[i] -= M->data[lvl - 1].U.val->data[k] * b_y1->data[j - 1];
        k++;
      }
    }

    k = M->data[lvl - 1].L.nrows * M->data[lvl - 1].L.nrows - 1;
    for (j = M->data[lvl - 1].L.nrows - 1; j + 1 > 0; j--) {
      b_y1->data[j] /= M->data[lvl - 1].U.val->data[k];
      for (i = j; i > 0; i--) {
        k--;
        b_y1->data[i - 1] -= M->data[lvl - 1].U.val->data[k] * b_y1->data[j];
      }

      k = ((k - nB) + j) - 1;
    }
  } else {
    emxCopyStruct_struct1_T(&expl_temp, &M->data[lvl - 1].L);
    crs_solve_utril(expl_temp.row_ptr, expl_temp.col_ind, expl_temp.val, b_y1);
    for (i = 0; i + 1 <= nB; i++) {
      b_y1->data[i] /= M->data[lvl - 1].d->data[i];
    }

    emxCopyStruct_struct1_T(&expl_temp, &M->data[lvl - 1].U);
    crs_solve_utriu(expl_temp.row_ptr, expl_temp.col_ind, expl_temp.val, b_y1);
  }

  if (n > M->data[lvl - 1].L.nrows) {
    emxCopyStruct_struct1_T(&expl_temp, &M->data[lvl - 1].negE);
    crs_Axpy(expl_temp.row_ptr, expl_temp.col_ind, expl_temp.val,
             expl_temp.nrows, b_y1, y2);
    i0 = n - M->data[lvl - 1].L.nrows;
    if (1 > i0) {
      k = 0;
    } else {
      k = i0;
    }

    i0 = (offset + M->data[lvl - 1].L.nrows) + 1;
    if (i0 > n) {
      i0 = 0;
    } else {
      i0--;
    }

    j = c_y1->size[0] * c_y1->size[1];
    c_y1->size[0] = 1;
    c_y1->size[1] = k;
    emxEnsureCapacity_real_T(c_y1, j);
    for (j = 0; j < k; j++) {
      c_y1->data[c_y1->size[0] * j] = y2->data[j];
    }

    k = c_y1->size[1];
    for (j = 0; j < k; j++) {
      b->data[i0 + j] = c_y1->data[c_y1->size[0] * j];
    }

    solve_milu(M, lvl + 1, b, offset + M->data[lvl - 1].L.nrows, b_y1, y2);
    i0 = (offset + M->data[lvl - 1].L.nrows) + 1;
    if (i0 > n) {
      i0 = 0;
      j = 0;
    } else {
      i0--;
      j = n;
    }

    i = c_y1->size[0] * c_y1->size[1];
    c_y1->size[0] = 1;
    c_y1->size[1] = j - i0;
    emxEnsureCapacity_real_T(c_y1, i);
    k = j - i0;
    for (j = 0; j < k; j++) {
      c_y1->data[c_y1->size[0] * j] = b->data[i0 + j];
    }

    k = c_y1->size[1];
    for (i0 = 0; i0 < k; i0++) {
      y2->data[i0] = c_y1->data[c_y1->size[0] * i0];
    }

    emxInit_real_T(&r1, 2);
    k = M->data[lvl - 1].L.nrows;
    i0 = r1->size[0] * r1->size[1];
    r1->size[0] = 1;
    r1->size[1] = k;
    emxEnsureCapacity_real_T(r1, i0);
    for (i0 = 0; i0 < k; i0++) {
      r1->data[r1->size[0] * i0] = b->data[i0 + offset];
    }

    k = r1->size[1];
    for (i0 = 0; i0 < k; i0++) {
      b_y1->data[i0] = r1->data[r1->size[0] * i0];
    }

    emxFree_real_T(&r1);
    emxCopyStruct_struct1_T(&expl_temp, &M->data[lvl - 1].negF);
    crs_Axpy(expl_temp.row_ptr, expl_temp.col_ind, expl_temp.val,
             expl_temp.nrows, y2, b_y1);
    emxCopyStruct_struct1_T(&expl_temp, &M->data[lvl - 1].L);
    crs_solve_utril(expl_temp.row_ptr, expl_temp.col_ind, expl_temp.val, b_y1);
    for (i = 0; i + 1 <= nB; i++) {
      b_y1->data[i] /= M->data[lvl - 1].d->data[i];
    }

    emxCopyStruct_struct1_T(&expl_temp, &M->data[lvl - 1].U);
    crs_solve_utriu(expl_temp.row_ptr, expl_temp.col_ind, expl_temp.val, b_y1);
  }

  emxFree_real_T(&c_y1);
  emxFreeStruct_struct1_T(&expl_temp);
  for (i = 0; i + 1 <= nB; i++) {
    b->data[(M->data[lvl - 1].q->data[i] + offset) - 1] = b_y1->data[i] *
      M->data[lvl - 1].colscal->data[M->data[lvl - 1].q->data[i] - 1];
  }

  for (i = M->data[lvl - 1].L.nrows; i + 1 <= n; i++) {
    b->data[(M->data[lvl - 1].q->data[i] + offset) - 1] = y2->data[i - nB] *
      M->data[lvl - 1].colscal->data[M->data[lvl - 1].q->data[i] - 1];
  }
}

void MILUsolve(const emxArray_struct0_T *M, emxArray_real_T *b, emxArray_real_T *
               b_y1, emxArray_real_T *y2)
{
  solve_milu(M, 1, b, 0, b_y1, y2);
}

void MILUsolve_2args(const emxArray_struct0_T *M, emxArray_real_T *b)
{
  int u0;
  int u1;
  emxArray_real_T *b_y1;
  int i1;
  emxArray_real_T *y2;
  u0 = M->data[0].L.nrows;
  u1 = M->data[0].negE.nrows;
  if (u0 > u1) {
    u1 = u0;
  }

  emxInit_real_T(&b_y1, 1);
  i1 = b_y1->size[0];
  b_y1->size[0] = u1;
  emxEnsureCapacity_real_T(b_y1, i1);
  for (i1 = 0; i1 < u1; i1++) {
    b_y1->data[i1] = 0.0;
  }

  emxInit_real_T(&y2, 1);
  i1 = y2->size[0];
  y2->size[0] = M->data[0].negE.nrows;
  emxEnsureCapacity_real_T(y2, i1);
  u0 = M->data[0].negE.nrows;
  for (i1 = 0; i1 < u0; i1++) {
    y2->data[i1] = 0.0;
  }

  solve_milu(M, 1, b, 0, b_y1, y2);
  emxFree_real_T(&y2);
  emxFree_real_T(&b_y1);
}

void MILUsolve_initialize(void)
{
}

void MILUsolve_terminate(void)
{
}
