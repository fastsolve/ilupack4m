#include "gmresMILU_CGS.h"
#include "m2c.h"
#include "omp.h"
#include <math.h>

static void b_m2c_error(void);
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

static void crs_prodAx(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
  *A_col_ind, const emxArray_real_T *A_val, int A_nrows, const emxArray_real_T
  *x, emxArray_real_T *b, int nthreads)
{
  int n;
  if (b->size[0] < A_nrows) {
    m2c_error();
  }

  n = omp_get_nested();
  if (n == 0) {
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
  int chunk;
  int threadID;
  double t;
  int b_remainder;
  int j;
  if (varargin_1) {
    iend = omp_get_num_threads();
    if (iend == 1) {
      istart = 0;
      iend = nrows;
    } else {
      threadID = omp_get_thread_num();
      chunk = nrows / iend;
      b_remainder = nrows - iend * chunk;
      if (b_remainder < threadID) {
        iend = b_remainder;
      } else {
        iend = threadID;
      }

      istart = threadID * chunk + iend;
      iend = (istart + chunk) + (threadID < b_remainder);
    }
  } else {
    istart = 0;
    iend = nrows;
  }

  for (chunk = istart + 1; chunk <= iend; chunk++) {
    t = 0.0;
    b_remainder = row_ptr->data[chunk - 1];
    threadID = row_ptr->data[chunk] - 1;
    for (j = b_remainder; j <= threadID; j++) {
      t += val->data[j - 1] * x->data[col_ind->data[j - 1] + -1];
    }

    b->data[chunk + -1] = t;
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
      b_m2c_error();
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
      b_m2c_error();
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

void gmresMILU_CGS(const struct0_T *A, const emxArray_real_T *b, const
                   emxArray_struct1_T *M, int restart, double rtol, int maxit,
                   const emxArray_real_T *x0, int verbose, int nthreads,
                   emxArray_real_T *x, int *flag, int *iter, emxArray_real_T
                   *resids)
{
  double beta2;
  int i;
  int ii;
  double beta0;
  int max_outer_iters;
  emxArray_real_T *y;
  emxArray_real_T *R;
  emxArray_real_T *Q;
  emxArray_real_T *Z;
  emxArray_real_T *J;
  emxArray_real_T *v;
  emxArray_real_T *v2;
  double resid;
  int it_outer;
  emxArray_real_T *w;
  boolean_T exitg1;
  boolean_T guard1 = false;
  int j;
  int exitg2;
  int k;
  double vnorm;
  double tmpv;
  double d;
  double d1;
  beta2 = 0.0;
  i = b->size[0];
  for (ii = 0; ii < i; ii++) {
    beta2 += b->data[ii] * b->data[ii];
  }

  beta0 = sqrt(beta2);
  if (beta0 == 0.0) {
    i = x->size[0];
    x->size[0] = b->size[0];
    emxEnsureCapacity_real_T(x, i);
    ii = b->size[0];
    for (i = 0; i < ii; i++) {
      x->data[i] = 0.0;
    }

    *flag = 0;
    *iter = 0;
    i = resids->size[0];
    resids->size[0] = 1;
    emxEnsureCapacity_real_T(resids, i);
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
      i = x->size[0];
      x->size[0] = b->size[0];
      emxEnsureCapacity_real_T(x, i);
      ii = b->size[0];
      for (i = 0; i < ii; i++) {
        x->data[i] = 0.0;
      }
    } else {
      i = x->size[0];
      x->size[0] = x0->size[0];
      emxEnsureCapacity_real_T(x, i);
      ii = x0->size[0];
      for (i = 0; i < ii; i++) {
        x->data[i] = x0->data[i];
      }
    }

    emxInit_real_T(&y, 1);
    i = y->size[0];
    y->size[0] = restart + 1;
    emxEnsureCapacity_real_T(y, i);
    for (i = 0; i <= restart; i++) {
      y->data[i] = 0.0;
    }

    emxInit_real_T(&R, 2);
    i = R->size[0] * R->size[1];
    R->size[0] = restart;
    R->size[1] = restart;
    emxEnsureCapacity_real_T(R, i);
    ii = restart * restart;
    for (i = 0; i < ii; i++) {
      R->data[i] = 0.0;
    }

    emxInit_real_T(&Q, 2);
    i = Q->size[0] * Q->size[1];
    Q->size[0] = b->size[0];
    Q->size[1] = restart;
    emxEnsureCapacity_real_T(Q, i);
    ii = b->size[0] * restart;
    for (i = 0; i < ii; i++) {
      Q->data[i] = 0.0;
    }

    emxInit_real_T(&Z, 2);
    i = Z->size[0] * Z->size[1];
    Z->size[0] = b->size[0];
    Z->size[1] = restart;
    emxEnsureCapacity_real_T(Z, i);
    ii = b->size[0] * restart;
    for (i = 0; i < ii; i++) {
      Z->data[i] = 0.0;
    }

    emxInit_real_T(&J, 2);
    i = J->size[0] * J->size[1];
    J->size[0] = 2;
    J->size[1] = restart;
    emxEnsureCapacity_real_T(J, i);
    ii = restart << 1;
    for (i = 0; i < ii; i++) {
      J->data[i] = 0.0;
    }

    emxInit_real_T(&v, 1);
    i = v->size[0];
    v->size[0] = b->size[0];
    emxEnsureCapacity_real_T(v, i);
    ii = b->size[0];
    for (i = 0; i < ii; i++) {
      v->data[i] = 0.0;
    }

    emxInit_real_T(&v2, 1);
    i = v2->size[0];
    v2->size[0] = M->data[0].negE.nrows;
    emxEnsureCapacity_real_T(v2, i);
    ii = M->data[0].negE.nrows;
    for (i = 0; i < ii; i++) {
      v2->data[i] = 0.0;
    }

    i = resids->size[0];
    resids->size[0] = maxit;
    emxEnsureCapacity_real_T(resids, i);
    for (i = 0; i < maxit; i++) {
      resids->data[i] = 0.0;
    }

    *flag = 0;
    *iter = 0;
    resid = 1.0;
    it_outer = 0;
    emxInit_real_T(&w, 1);
    exitg1 = false;
    while ((!exitg1) && (it_outer <= max_outer_iters - 1)) {
      guard1 = false;
      if (it_outer + 1 > 1) {
        guard1 = true;
      } else {
        beta2 = 0.0;
        i = x->size[0];
        for (ii = 0; ii < i; ii++) {
          beta2 += x->data[ii] * x->data[ii];
        }

        if (beta2 > 0.0) {
          guard1 = true;
        } else {
          i = v->size[0];
          v->size[0] = b->size[0];
          emxEnsureCapacity_real_T(v, i);
          ii = b->size[0];
          for (i = 0; i < ii; i++) {
            v->data[i] = b->data[i];
          }
        }
      }

      if (guard1) {
        crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, x, v, nthreads);
        i = v->size[0];
        v->size[0] = b->size[0];
        emxEnsureCapacity_real_T(v, i);
        ii = b->size[0];
        for (i = 0; i < ii; i++) {
          v->data[i] = b->data[i] - v->data[i];
        }
      }

      beta2 = 0.0;
      i = v->size[0];
      for (ii = 0; ii < i; ii++) {
        beta2 += v->data[ii] * v->data[ii];
      }

      beta2 = sqrt(beta2);
      y->data[0] = beta2;
      ii = v->size[0];
      for (i = 0; i < ii; i++) {
        Q->data[i] = v->data[i] / beta2;
      }

      j = 0;
      do {
        exitg2 = 0;
        ii = Q->size[0];
        i = w->size[0];
        w->size[0] = Q->size[0];
        emxEnsureCapacity_real_T(w, i);
        for (i = 0; i < ii; i++) {
          w->data[i] = Q->data[i + Q->size[0] * j];
        }

        solve_milu(M, 1, w, 0, v, v2);
        ii = w->size[0];
        for (i = 0; i < ii; i++) {
          Z->data[i + Z->size[0] * j] = w->data[i];
        }

        crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, w, v, nthreads);
        i = w->size[0];
        w->size[0] = v->size[0];
        emxEnsureCapacity_real_T(w, i);
        ii = v->size[0];
        for (i = 0; i < ii; i++) {
          w->data[i] = v->data[i];
        }

        for (k = 0; k <= j; k++) {
          beta2 = 0.0;
          ii = w->size[0];
          for (i = 0; i < ii; i++) {
            beta2 += w->data[i] * Q->data[i + Q->size[0] * k];
          }

          R->data[k + R->size[0] * j] = beta2;
          beta2 = R->data[k + R->size[0] * j];
          ii = v->size[0];
          for (i = 0; i < ii; i++) {
            v->data[i] -= beta2 * Q->data[i + Q->size[0] * k];
          }
        }

        beta2 = 0.0;
        i = v->size[0];
        for (ii = 0; ii < i; ii++) {
          beta2 += v->data[ii] * v->data[ii];
        }

        vnorm = sqrt(beta2);
        if (j + 1 < restart) {
          ii = v->size[0];
          for (i = 0; i < ii; i++) {
            Q->data[i + Q->size[0] * (j + 1)] = v->data[i] / vnorm;
          }
        }

        for (ii = 0; ii < j; ii++) {
          tmpv = R->data[ii + R->size[0] * j];
          d = J->data[2 * ii];
          d1 = J->data[2 * ii + 1];
          R->data[ii + R->size[0] * j] = d * R->data[ii + R->size[0] * j] + d1 *
            R->data[(ii + R->size[0] * j) + 1];
          R->data[(ii + R->size[0] * j) + 1] = -d1 * tmpv + d * R->data[(ii +
            R->size[0] * j) + 1];
        }

        beta2 = sqrt(R->data[j + R->size[0] * j] * R->data[j + R->size[0] * j] +
                     beta2);
        J->data[2 * j] = R->data[j + R->size[0] * j] / beta2;
        i = 2 * j + 1;
        J->data[i] = vnorm / beta2;
        y->data[j + 1] = -J->data[i] * y->data[j];
        y->data[j] *= J->data[2 * j];
        R->data[j + R->size[0] * j] = beta2;
        beta2 = resid;
        resid = fabs(y->data[j + 1]) / beta0;
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
          if ((resid < rtol) || (j + 1 >= restart)) {
            exitg2 = 1;
          } else {
            j++;
          }
        }
      } while (exitg2 == 0);

      if ((verbose == 1) || ((verbose > 1) && (*flag != 0))) {
        m2c_printf(*iter, resid);
      }

      for (k = j + 1; k >= 1; k--) {
        i = k + 1;
        for (ii = i; ii <= j + 1; ii++) {
          y->data[k - 1] -= R->data[(k + R->size[0] * (ii - 1)) - 1] * y->
            data[ii - 1];
        }

        y->data[k - 1] /= R->data[(k + R->size[0] * (k - 1)) - 1];
      }

      for (k = 0; k <= j; k++) {
        ii = x->size[0];
        for (i = 0; i < ii; i++) {
          x->data[i] += y->data[k] * Z->data[i + Z->size[0] * k];
        }
      }

      if ((resid < rtol) || (*flag != 0)) {
        exitg1 = true;
      } else {
        it_outer++;
      }
    }

    emxFree_real_T(&w);
    emxFree_real_T(&v2);
    emxFree_real_T(&v);
    emxFree_real_T(&J);
    emxFree_real_T(&Z);
    emxFree_real_T(&Q);
    emxFree_real_T(&R);
    emxFree_real_T(&y);
    i = resids->size[0];
    if (1 > *iter) {
      resids->size[0] = 0;
    } else {
      resids->size[0] = *iter;
    }

    emxEnsureCapacity_real_T(resids, i);
    if (resid <= rtol * 1.00000001) {
      *flag = 0;
    }
  }
}

void gmresMILU_CGS_initialize(void)
{
}

void gmresMILU_CGS_terminate(void)
{
}
