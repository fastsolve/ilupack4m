#include "MILUsolve.h"
#include "m2c.h"

static void crs_Axpy_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, emxArray_real_T *y, int nrows);
static void m2c_error(void);
static void solve_milu(const emxArray_struct0_T *M, int lvl, emxArray_real_T *b,
  int offset, emxArray_real_T *b_y1, emxArray_real_T *y2);
static void crs_Axpy_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, emxArray_real_T *y, int nrows)
{
  int i;
  int b_i;
  int t_tmp;
  double t;
  int c_i;
  int j;
  for (i = 0; i < nrows; i++) {
    b_i = i + 1;
    t_tmp = b_i + -1;
    t = y->data[t_tmp];
    c_i = row_ptr->data[b_i - 1];
    b_i = row_ptr->data[b_i] - 1;
    for (j = c_i; j <= b_i; j++) {
      t += val->data[j - 1] * x->data[col_ind->data[j - 1] + -1];
    }

    y->data[t_tmp] = t;
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
  int b_n;
  int i;
  int i1;
  int k;
  int j;
  int i2;
  int i3;
  nB = M->data[lvl - 1].L.nrows - 1;
  n = M->data[lvl - 1].L.nrows + M->data[lvl - 1].negE.nrows;
  for (b_n = 0; b_n <= nB; b_n++) {
    b_y1->data[b_n] = M->data[lvl - 1].rowscal->data[M->data[lvl - 1].p->
      data[b_n] - 1] * b->data[(M->data[lvl - 1].p->data[b_n] + offset) - 1];
  }

  i = M->data[lvl - 1].L.nrows + 1;
  for (b_n = i; b_n <= n; b_n++) {
    i1 = M->data[lvl - 1].p->data[b_n - 1];
    y2->data[(b_n - nB) - 2] = M->data[lvl - 1].rowscal->data[i1 - 1] * b->data
      [(i1 + offset) - 1];
  }

  if (n > M->data[lvl - 1].L.nrows) {
    for (b_n = 0; b_n <= nB; b_n++) {
      b->data[offset + b_n] = b_y1->data[b_n];
    }
  }

  if ((M->data[lvl - 1].L.val->size[0] == 0) && (M->data[lvl - 1].U.val->size[0]
       == n * n)) {
    k = 0;
    for (j = 0; j <= nB; j++) {
      k = (k + j) + 1;
      i1 = j + 2;
      for (b_n = i1; b_n <= nB + 1; b_n++) {
        b_y1->data[b_n - 1] -= M->data[lvl - 1].U.val->data[k] * b_y1->data[j];
        k++;
      }
    }

    k = M->data[lvl - 1].L.nrows * M->data[lvl - 1].L.nrows - 1;
    for (j = nB + 1; j >= 1; j--) {
      b_y1->data[j - 1] /= M->data[lvl - 1].U.val->data[k];
      i1 = j - 1;
      for (b_n = i1; b_n >= 1; b_n--) {
        k--;
        b_y1->data[b_n - 1] -= M->data[lvl - 1].U.val->data[k] * b_y1->data[j -
          1];
      }

      k = ((k - nB) + j) - 3;
    }
  } else {
    b_n = M->data[lvl - 1].L.col_ptr->size[0];
    for (j = 0; j <= b_n - 2; j++) {
      i1 = M->data[lvl - 1].L.col_ptr->data[j];
      i2 = M->data[lvl - 1].L.col_ptr->data[j + 1] - 1;
      for (k = i1; k <= i2; k++) {
        i3 = M->data[lvl - 1].L.row_ind->data[k - 1] - 1;
        b_y1->data[i3] -= M->data[lvl - 1].L.val->data[k - 1] * b_y1->data[j];
      }
    }

    for (b_n = 0; b_n <= nB; b_n++) {
      b_y1->data[b_n] /= M->data[lvl - 1].d->data[b_n];
    }

    b_n = M->data[lvl - 1].U.col_ptr->size[0] - 1;
    for (j = b_n; j >= 1; j--) {
      i1 = M->data[lvl - 1].U.col_ptr->data[j - 1];
      i2 = M->data[lvl - 1].U.col_ptr->data[j] - 1;
      for (k = i1; k <= i2; k++) {
        i3 = M->data[lvl - 1].U.row_ind->data[k - 1] - 1;
        b_y1->data[i3] -= M->data[lvl - 1].U.val->data[k - 1] * b_y1->data[j - 1];
      }
    }
  }

  if (n > M->data[lvl - 1].L.nrows) {
    if (y2->size[0] < M->data[lvl - 1].negE.nrows) {
      m2c_error();
    }

    crs_Axpy_kernel(M->data[lvl - 1].negE.row_ptr, M->data[lvl - 1].negE.col_ind,
                    M->data[lvl - 1].negE.val, b_y1, y2, M->data[lvl - 1].
                    negE.nrows);
    i1 = n - M->data[lvl - 1].L.nrows;
    for (b_n = 0; b_n < i1; b_n++) {
      b->data[((offset + nB) + b_n) + 1] = y2->data[b_n];
    }

    solve_milu(M, lvl + 1, b, offset + M->data[lvl - 1].L.nrows, b_y1, y2);
    for (b_n = 0; b_n <= nB; b_n++) {
      b_y1->data[b_n] = b->data[offset + b_n];
    }

    for (b_n = 0; b_n < i1; b_n++) {
      y2->data[b_n] = b->data[((offset + nB) + b_n) + 1];
    }

    if (b_y1->size[0] < M->data[lvl - 1].negF.nrows) {
      m2c_error();
    }

    crs_Axpy_kernel(M->data[lvl - 1].negF.row_ptr, M->data[lvl - 1].negF.col_ind,
                    M->data[lvl - 1].negF.val, y2, b_y1, M->data[lvl - 1].
                    negF.nrows);
    b_n = M->data[lvl - 1].L.col_ptr->size[0];
    for (j = 0; j <= b_n - 2; j++) {
      i1 = M->data[lvl - 1].L.col_ptr->data[j];
      i2 = M->data[lvl - 1].L.col_ptr->data[j + 1] - 1;
      for (k = i1; k <= i2; k++) {
        i3 = M->data[lvl - 1].L.row_ind->data[k - 1] - 1;
        b_y1->data[i3] -= M->data[lvl - 1].L.val->data[k - 1] * b_y1->data[j];
      }
    }

    for (b_n = 0; b_n <= nB; b_n++) {
      b_y1->data[b_n] /= M->data[lvl - 1].d->data[b_n];
    }

    b_n = M->data[lvl - 1].U.col_ptr->size[0] - 1;
    for (j = b_n; j >= 1; j--) {
      i1 = M->data[lvl - 1].U.col_ptr->data[j - 1];
      i2 = M->data[lvl - 1].U.col_ptr->data[j] - 1;
      for (k = i1; k <= i2; k++) {
        i3 = M->data[lvl - 1].U.row_ind->data[k - 1] - 1;
        b_y1->data[i3] -= M->data[lvl - 1].U.val->data[k - 1] * b_y1->data[j - 1];
      }
    }
  }

  for (b_n = 0; b_n <= nB; b_n++) {
    b->data[(M->data[lvl - 1].q->data[b_n] + offset) - 1] = b_y1->data[b_n] *
      M->data[lvl - 1].colscal->data[M->data[lvl - 1].q->data[b_n] - 1];
  }

  for (b_n = i; b_n <= n; b_n++) {
    i1 = M->data[lvl - 1].q->data[b_n - 1];
    b->data[(i1 + offset) - 1] = y2->data[(b_n - nB) - 2] * M->data[lvl - 1].
      colscal->data[i1 - 1];
  }
}

void MILUsolve(const emxArray_struct0_T *M, emxArray_real_T *b, emxArray_real_T *
               b_y1, emxArray_real_T *y2)
{
  solve_milu(M, 1, b, 0, b_y1, y2);
}

void MILUsolve_2args(const emxArray_struct0_T *M, emxArray_real_T *b)
{
  int maxval;
  emxArray_real_T *b_y1;
  int i;
  emxArray_real_T *y2;
  if (M->data[0].L.nrows > M->data[0].negE.nrows) {
    maxval = M->data[0].L.nrows;
  } else {
    maxval = M->data[0].negE.nrows;
  }

  emxInit_real_T(&b_y1, 1);
  i = b_y1->size[0];
  b_y1->size[0] = maxval;
  emxEnsureCapacity_real_T(b_y1, i);
  for (i = 0; i < maxval; i++) {
    b_y1->data[i] = 0.0;
  }

  emxInit_real_T(&y2, 1);
  i = y2->size[0];
  y2->size[0] = M->data[0].negE.nrows;
  emxEnsureCapacity_real_T(y2, i);
  maxval = M->data[0].negE.nrows;
  for (i = 0; i < maxval; i++) {
    y2->data[i] = 0.0;
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
