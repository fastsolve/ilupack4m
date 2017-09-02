#include "gmresMILU_HO.h"
#include "m2c.h"
#include "omp.h"

static void b_m2c_error(void);
static void backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int cend);
static void crs_Axpy_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, emxArray_real_T *y, int nrows);
static void crs_prodAx(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
  *A_col_ind, const emxArray_real_T *A_val, int A_nrows, const emxArray_real_T
  *x, emxArray_real_T *b, int nthreads);
static void crs_prodAx_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, emxArray_real_T *b, int nrows, boolean_T varargin_1);
static void m2c_error(void);
static void m2c_printf(int varargin_2, double varargin_3);
static void m2c_warn(void);
static void solve_milu(const emxArray_struct1_T *M, int lvl, emxArray_real_T *b,
  int offset, emxArray_real_T *b_y1, emxArray_real_T *y2);
static void b_m2c_error(void)
{
  const char * msgid;
  const char * fmt;
  msgid = "crs_Axpy:BufferTooSmal";
  fmt = "Buffer space for output y is too small.";
  M2C_error(msgid, fmt);
}

static void backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int cend)
{
  int jj;
  int ii;
  for (jj = cend - 1; jj + 1 > 0; jj--) {
    for (ii = jj + 1; ii + 1 <= cend; ii++) {
      bs->data[jj] -= R->data[jj + R->size[0] * ii] * bs->data[ii];
    }

    bs->data[jj] /= R->data[jj + R->size[0] * jj];
  }
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

static void crs_prodAx(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
  *A_col_ind, const emxArray_real_T *A_val, int A_nrows, const emxArray_real_T
  *x, emxArray_real_T *b, int nthreads)
{
  int n;
  if (b->size[0] < A_nrows) {
    m2c_error();
  }

  n = omp_get_nested();
  if (!(n != 0)) {
    n = omp_get_num_threads();
    if ((n > 1) && (nthreads > 1)) {

#pragma omp master
      {
        m2c_warn();
      }

    }
  }

#pragma omp parallel default(shared) num_threads(nthreads)
  {
    n = omp_get_num_threads();
    crs_prodAx_kernel(A_row_ptr, A_col_ind, A_val, x, b, A_nrows, n > 1);
  }

}

static void crs_prodAx_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, emxArray_real_T *b, int nrows, boolean_T varargin_1)
{
  int istart;
  int iend;
  double t;
  int chunk;
  int b_remainder;
  if (varargin_1) {
    istart = omp_get_num_threads();
    if (istart == 1) {
      istart = 0;
      iend = nrows;
    } else {
      iend = omp_get_thread_num();
      chunk = nrows / istart;
      b_remainder = nrows - istart * chunk;
      if (b_remainder < iend) {
        istart = b_remainder;
      } else {
        istart = iend;
      }

      istart += iend * chunk;
      iend = (istart + chunk) + (iend < b_remainder);
    }
  } else {
    istart = 0;
    iend = nrows;
  }

  for (istart++; istart <= iend; istart++) {
    t = 0.0;
    for (b_remainder = row_ptr->data[istart - 1]; b_remainder < row_ptr->
         data[istart]; b_remainder++) {
      t += val->data[b_remainder - 1] * x->data[col_ind->data[b_remainder - 1] -
        1];
    }

    b->data[istart - 1] = t;
  }
}

static void m2c_error(void)
{
  const char * msgid;
  const char * fmt;
  msgid = "crs_prodAx:BufferTooSmal";
  fmt = "Buffer space for output b is too small.";
  M2C_error(msgid, fmt);
}

static void m2c_printf(int varargin_2, double varargin_3)
{
  const char * fmt;
  fmt = "At iteration %d, relative residual is %g.\n";
  M2C_printf(fmt, varargin_2, varargin_3);
}

static void m2c_warn(void)
{
  const char * msgid;
  const char * fmt;
  msgid = "crs_prodAx:NestedParallel";
  fmt =
    "You are trying to use nested parallel regions. Solution may be incorrect.";
  M2C_warn(msgid, fmt);
}

static void solve_milu(const emxArray_struct1_T *M, int lvl, emxArray_real_T *b,
  int offset, emxArray_real_T *b_y1, emxArray_real_T *y2)
{
  int nB;
  int n;
  int i;
  int b_n;
  int k;
  int j;
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

  if (n > M->data[lvl - 1].L.nrows) {
    for (i = 0; i + 1 <= nB; i++) {
      b->data[offset + i] = b_y1->data[i];
    }
  }

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
    b_n = M->data[lvl - 1].L.col_ptr->size[0] - 1;
    for (j = 1; j <= b_n; j++) {
      for (k = M->data[lvl - 1].L.col_ptr->data[j - 1] - 1; k + 1 < M->data[lvl
           - 1].L.col_ptr->data[j]; k++) {
        b_y1->data[M->data[lvl - 1].L.row_ind->data[k] - 1] -= M->data[lvl - 1].
          L.val->data[k] * b_y1->data[j - 1];
      }
    }

    for (i = 0; i + 1 <= nB; i++) {
      b_y1->data[i] /= M->data[lvl - 1].d->data[i];
    }

    for (j = M->data[lvl - 1].U.col_ptr->size[0] - 1; j > 0; j--) {
      for (k = M->data[lvl - 1].U.col_ptr->data[j - 1] - 1; k + 1 < M->data[lvl
           - 1].U.col_ptr->data[j]; k++) {
        b_y1->data[M->data[lvl - 1].U.row_ind->data[k] - 1] -= M->data[lvl - 1].
          U.val->data[k] * b_y1->data[j - 1];
      }
    }
  }

  if (n > M->data[lvl - 1].L.nrows) {
    if (y2->size[0] < M->data[lvl - 1].negE.nrows) {
      b_m2c_error();
    }

    crs_Axpy_kernel(M->data[lvl - 1].negE.row_ptr, M->data[lvl - 1].negE.col_ind,
                    M->data[lvl - 1].negE.val, b_y1, y2, M->data[lvl - 1].
                    negE.nrows);
    b_n = n - M->data[lvl - 1].L.nrows;
    for (i = 0; i + 1 <= b_n; i++) {
      b->data[(offset + nB) + i] = y2->data[i];
    }

    solve_milu(M, lvl + 1, b, offset + M->data[lvl - 1].L.nrows, b_y1, y2);
    for (i = 0; i + 1 <= nB; i++) {
      b_y1->data[i] = b->data[offset + i];
    }

    b_n = n - M->data[lvl - 1].L.nrows;
    for (i = 0; i + 1 <= b_n; i++) {
      y2->data[i] = b->data[(offset + nB) + i];
    }

    if (b_y1->size[0] < M->data[lvl - 1].negF.nrows) {
      b_m2c_error();
    }

    crs_Axpy_kernel(M->data[lvl - 1].negF.row_ptr, M->data[lvl - 1].negF.col_ind,
                    M->data[lvl - 1].negF.val, y2, b_y1, M->data[lvl - 1].
                    negF.nrows);
    b_n = M->data[lvl - 1].L.col_ptr->size[0] - 1;
    for (j = 1; j <= b_n; j++) {
      for (k = M->data[lvl - 1].L.col_ptr->data[j - 1] - 1; k + 1 < M->data[lvl
           - 1].L.col_ptr->data[j]; k++) {
        b_y1->data[M->data[lvl - 1].L.row_ind->data[k] - 1] -= M->data[lvl - 1].
          L.val->data[k] * b_y1->data[j - 1];
      }
    }

    for (i = 0; i + 1 <= nB; i++) {
      b_y1->data[i] /= M->data[lvl - 1].d->data[i];
    }

    for (j = M->data[lvl - 1].U.col_ptr->size[0] - 1; j > 0; j--) {
      for (k = M->data[lvl - 1].U.col_ptr->data[j - 1] - 1; k + 1 < M->data[lvl
           - 1].U.col_ptr->data[j]; k++) {
        b_y1->data[M->data[lvl - 1].U.row_ind->data[k] - 1] -= M->data[lvl - 1].
          U.val->data[k] * b_y1->data[j - 1];
      }
    }
  }

  for (i = 0; i + 1 <= nB; i++) {
    b->data[(M->data[lvl - 1].q->data[i] + offset) - 1] = b_y1->data[i] *
      M->data[lvl - 1].colscal->data[M->data[lvl - 1].q->data[i] - 1];
  }

  for (i = M->data[lvl - 1].L.nrows; i + 1 <= n; i++) {
    b->data[(M->data[lvl - 1].q->data[i] + offset) - 1] = y2->data[i - nB] *
      M->data[lvl - 1].colscal->data[M->data[lvl - 1].q->data[i] - 1];
  }
}

void gmresMILU_HO(const struct0_T *A, const emxArray_real_T *b, const
                  emxArray_struct1_T *M, int restart, double rtol, int maxit,
                  const emxArray_real_T *x0, int verbose, int nthreads,
                  emxArray_real_T *x, int *flag, int *iter, emxArray_real_T
                  *resids)
{
  int n;
  double beta2;
  int ii;
  double beta0;
  int i0;
  int max_outer_iters;
  emxArray_real_T *V;
  emxArray_real_T *R;
  emxArray_real_T *y;
  emxArray_real_T *Z;
  emxArray_real_T *J;
  emxArray_real_T *w;
  emxArray_real_T *y2;
  double resid;
  int it_outer;
  emxArray_real_T *u;
  emxArray_real_T *v;
  emxArray_int32_T *r0;
  emxArray_real_T *b_Z;
  boolean_T exitg1;
  boolean_T guard1 = false;
  double beta;
  int j;
  int exitg2;
  int i;
  n = b->size[0];
  beta2 = 0.0;
  for (ii = 0; ii + 1 <= b->size[0]; ii++) {
    beta2 += b->data[ii] * b->data[ii];
  }

  beta0 = sqrt(beta2);
  if (beta0 == 0.0) {
    i0 = x->size[0];
    x->size[0] = b->size[0];
    emxEnsureCapacity((emxArray__common *)x, i0, sizeof(double));
    ii = b->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      x->data[i0] = 0.0;
    }

    *flag = 0;
    *iter = 0;
    i0 = resids->size[0];
    resids->size[0] = 1;
    emxEnsureCapacity((emxArray__common *)resids, i0, sizeof(double));
    resids->data[0] = 0.0;
  } else {
    if (restart > b->size[0]) {
      restart = b->size[0];
    } else {
      if (restart <= 0) {
        restart = 1;
      }
    }

    max_outer_iters = (int)ceil((double)maxit / (double)restart);
    if (x0->size[0] == 0) {
      i0 = x->size[0];
      x->size[0] = b->size[0];
      emxEnsureCapacity((emxArray__common *)x, i0, sizeof(double));
      ii = b->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        x->data[i0] = 0.0;
      }
    } else {
      i0 = x->size[0];
      x->size[0] = x0->size[0];
      emxEnsureCapacity((emxArray__common *)x, i0, sizeof(double));
      ii = x0->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        x->data[i0] = x0->data[i0];
      }
    }

    emxInit_real_T(&V, 2);
    i0 = V->size[0] * V->size[1];
    V->size[0] = b->size[0];
    V->size[1] = restart;
    emxEnsureCapacity((emxArray__common *)V, i0, sizeof(double));
    ii = b->size[0] * restart;
    for (i0 = 0; i0 < ii; i0++) {
      V->data[i0] = 0.0;
    }

    emxInit_real_T(&R, 2);
    i0 = R->size[0] * R->size[1];
    R->size[0] = restart;
    R->size[1] = restart;
    emxEnsureCapacity((emxArray__common *)R, i0, sizeof(double));
    ii = restart * restart;
    for (i0 = 0; i0 < ii; i0++) {
      R->data[i0] = 0.0;
    }

    emxInit_real_T(&y, 1);
    i0 = y->size[0];
    y->size[0] = restart + 1;
    emxEnsureCapacity((emxArray__common *)y, i0, sizeof(double));
    for (i0 = 0; i0 <= restart; i0++) {
      y->data[i0] = 0.0;
    }

    emxInit_real_T(&Z, 2);
    i0 = Z->size[0] * Z->size[1];
    Z->size[0] = b->size[0];
    Z->size[1] = restart;
    emxEnsureCapacity((emxArray__common *)Z, i0, sizeof(double));
    ii = b->size[0] * restart;
    for (i0 = 0; i0 < ii; i0++) {
      Z->data[i0] = 0.0;
    }

    emxInit_real_T(&J, 2);
    i0 = J->size[0] * J->size[1];
    J->size[0] = 2;
    J->size[1] = restart;
    emxEnsureCapacity((emxArray__common *)J, i0, sizeof(double));
    ii = restart << 1;
    for (i0 = 0; i0 < ii; i0++) {
      J->data[i0] = 0.0;
    }

    i0 = resids->size[0];
    resids->size[0] = maxit;
    emxEnsureCapacity((emxArray__common *)resids, i0, sizeof(double));
    for (i0 = 0; i0 < maxit; i0++) {
      resids->data[i0] = 0.0;
    }

    emxInit_real_T(&w, 1);
    i0 = w->size[0];
    w->size[0] = b->size[0];
    emxEnsureCapacity((emxArray__common *)w, i0, sizeof(double));
    ii = b->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      w->data[i0] = 0.0;
    }

    emxInit_real_T(&y2, 1);
    i0 = y2->size[0];
    y2->size[0] = M->data[0].negE.nrows;
    emxEnsureCapacity((emxArray__common *)y2, i0, sizeof(double));
    ii = M->data[0].negE.nrows;
    for (i0 = 0; i0 < ii; i0++) {
      y2->data[i0] = 0.0;
    }

    *flag = 0;
    *iter = 0;
    resid = 1.0;
    it_outer = 1;
    emxInit_real_T(&u, 1);
    emxInit_real_T(&v, 1);
    emxInit_int32_T(&r0, 2);
    emxInit_real_T(&b_Z, 1);
    exitg1 = false;
    while ((!exitg1) && (it_outer <= max_outer_iters)) {
      guard1 = false;
      if (it_outer > 1) {
        guard1 = true;
      } else {
        beta2 = 0.0;
        for (ii = 0; ii + 1 <= x->size[0]; ii++) {
          beta2 += x->data[ii] * x->data[ii];
        }

        if (beta2 > 0.0) {
          guard1 = true;
        } else {
          i0 = u->size[0];
          u->size[0] = b->size[0];
          emxEnsureCapacity((emxArray__common *)u, i0, sizeof(double));
          ii = b->size[0];
          for (i0 = 0; i0 < ii; i0++) {
            u->data[i0] = b->data[i0];
          }
        }
      }

      if (guard1) {
        crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, x, w, nthreads);
        i0 = u->size[0];
        u->size[0] = b->size[0];
        emxEnsureCapacity((emxArray__common *)u, i0, sizeof(double));
        ii = b->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          u->data[i0] = b->data[i0] - w->data[i0];
        }
      }

      beta2 = 0.0;
      for (ii = 0; ii + 1 <= u->size[0]; ii++) {
        beta2 += u->data[ii] * u->data[ii];
      }

      beta = sqrt(beta2);
      if (u->data[0] < 0.0) {
        beta = -beta;
      }

      beta2 = sqrt(2.0 * beta2 + 2.0 * u->data[0] * beta);
      u->data[0] += beta;
      i0 = u->size[0];
      emxEnsureCapacity((emxArray__common *)u, i0, sizeof(double));
      ii = u->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        u->data[i0] /= beta2;
      }

      y->data[0] = -beta;
      ii = u->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        V->data[i0] = u->data[i0];
      }

      j = 1;
      do {
        exitg2 = 0;
        beta2 = -2.0 * V->data[(j + V->size[0] * (j - 1)) - 1];
        ii = V->size[0];
        i0 = v->size[0];
        v->size[0] = ii;
        emxEnsureCapacity((emxArray__common *)v, i0, sizeof(double));
        for (i0 = 0; i0 < ii; i0++) {
          v->data[i0] = beta2 * V->data[i0 + V->size[0] * (j - 1)];
        }

        v->data[j - 1]++;
        for (i = j - 2; i + 1 > 0; i--) {
          beta2 = V->data[i + V->size[0] * i] * v->data[i];
          for (ii = i + 1; ii + 1 <= n; ii++) {
            beta2 += V->data[ii + V->size[0] * i] * v->data[ii];
          }

          beta2 *= 2.0;
          for (ii = i; ii + 1 <= n; ii++) {
            v->data[ii] -= beta2 * V->data[ii + V->size[0] * i];
          }
        }

        beta2 = 0.0;
        for (ii = 0; ii + 1 <= v->size[0]; ii++) {
          beta2 += v->data[ii] * v->data[ii];
        }

        beta2 = sqrt(beta2);
        i0 = v->size[0];
        emxEnsureCapacity((emxArray__common *)v, i0, sizeof(double));
        ii = v->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          v->data[i0] /= beta2;
        }

        solve_milu(M, 1, v, 0, w, y2);
        ii = v->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          Z->data[i0 + Z->size[0] * (j - 1)] = v->data[i0];
        }

        ii = Z->size[0];
        i0 = b_Z->size[0];
        b_Z->size[0] = ii;
        emxEnsureCapacity((emxArray__common *)b_Z, i0, sizeof(double));
        for (i0 = 0; i0 < ii; i0++) {
          b_Z->data[i0] = Z->data[i0 + Z->size[0] * (j - 1)];
        }

        crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, b_Z, w, nthreads);
        for (i = 0; i + 1 <= j; i++) {
          beta2 = V->data[i + V->size[0] * i] * w->data[i];
          for (ii = i + 1; ii + 1 <= n; ii++) {
            beta2 += V->data[ii + V->size[0] * i] * w->data[ii];
          }

          beta2 *= 2.0;
          for (ii = i; ii + 1 <= n; ii++) {
            w->data[ii] -= beta2 * V->data[ii + V->size[0] * i];
          }
        }

        if (j < n) {
          u->data[j - 1] = 0.0;
          u->data[j] = w->data[j];
          beta2 = w->data[j] * w->data[j];
          for (ii = j + 1; ii + 1 <= n; ii++) {
            u->data[ii] = w->data[ii];
            beta2 += w->data[ii] * w->data[ii];
          }

          if (beta2 > 0.0) {
            beta = sqrt(beta2);
            if (u->data[j] < 0.0) {
              beta = -beta;
            }

            if (j < restart) {
              beta2 = sqrt(2.0 * beta2 + 2.0 * u->data[j] * beta);
              u->data[j] += beta;
              for (ii = j; ii + 1 <= n; ii++) {
                V->data[ii + V->size[0] * j] = u->data[ii] / beta2;
              }
            }

            if (j + 2 > w->size[0]) {
              i0 = 1;
              i = 0;
            } else {
              i0 = j + 2;
              i = w->size[0];
            }

            ii = r0->size[0] * r0->size[1];
            r0->size[0] = 1;
            r0->size[1] = (i - i0) + 1;
            emxEnsureCapacity((emxArray__common *)r0, ii, sizeof(int));
            ii = (i - i0) + 1;
            for (i = 0; i < ii; i++) {
              r0->data[r0->size[0] * i] = (i0 + i) - 1;
            }

            ii = r0->size[0] * r0->size[1];
            for (i0 = 0; i0 < ii; i0++) {
              w->data[r0->data[i0]] = 0.0;
            }

            w->data[j] = -beta;
          }
        }

        for (ii = 0; ii + 1 < j; ii++) {
          beta2 = w->data[ii];
          w->data[ii] = J->data[J->size[0] * ii] * w->data[ii] + J->data[1 +
            J->size[0] * ii] * w->data[ii + 1];
          w->data[ii + 1] = -J->data[1 + J->size[0] * ii] * beta2 + J->data
            [J->size[0] * ii] * w->data[ii + 1];
        }

        if (j < n) {
          beta2 = sqrt(w->data[j - 1] * w->data[j - 1] + w->data[j] * w->data[j]);
          J->data[J->size[0] * (j - 1)] = w->data[j - 1] / beta2;
          J->data[1 + J->size[0] * (j - 1)] = w->data[j] / beta2;
          y->data[j] = -J->data[1 + J->size[0] * (j - 1)] * y->data[j - 1];
          y->data[j - 1] *= J->data[J->size[0] * (j - 1)];
          w->data[j - 1] = beta2;
        }

        for (i0 = 0; i0 < j; i0++) {
          R->data[i0 + R->size[0] * (j - 1)] = w->data[i0];
        }

        beta2 = resid;
        resid = fabs(y->data[j]) / beta0;
        if (resid >= beta2 * 0.99999999) {
          *flag = 3;
          exitg2 = 1;
        } else if (*iter >= maxit) {
          *flag = 1;
          exitg2 = 1;
        } else {
          (*iter)++;
          if (verbose > 1) {
            m2c_printf(*iter, resid);
          }

          resids->data[*iter - 1] = resid;
          if ((resid < rtol) || (j >= restart)) {
            exitg2 = 1;
          } else {
            j++;
          }
        }
      } while (exitg2 == 0);

      if ((verbose == 1) || ((verbose > 1) && (*flag != 0))) {
        m2c_printf(*iter, resid);
      }

      backsolve(R, y, j);
      for (i = 0; i + 1 <= j; i++) {
        beta2 = y->data[i];
        i0 = x->size[0];
        emxEnsureCapacity((emxArray__common *)x, i0, sizeof(double));
        ii = x->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          x->data[i0] += beta2 * Z->data[i0 + Z->size[0] * i];
        }
      }

      if ((resid < rtol) || (*flag != 0)) {
        exitg1 = true;
      } else {
        it_outer++;
      }
    }

    emxFree_real_T(&b_Z);
    emxFree_int32_T(&r0);
    emxFree_real_T(&v);
    emxFree_real_T(&u);
    emxFree_real_T(&y2);
    emxFree_real_T(&w);
    emxFree_real_T(&J);
    emxFree_real_T(&Z);
    emxFree_real_T(&y);
    emxFree_real_T(&R);
    emxFree_real_T(&V);
    i0 = resids->size[0];
    if (1 > *iter) {
      resids->size[0] = 0;
    } else {
      resids->size[0] = *iter;
    }

    emxEnsureCapacity((emxArray__common *)resids, i0, sizeof(double));
    if (resid <= rtol * 1.00000001) {
      *flag = 0;
    }
  }
}

void gmresMILU_HO_initialize(void)
{
}

void gmresMILU_HO_terminate(void)
{
}
