#include "gmresMILU_CGS.h"
#include "m2c.h"
#include "omp.h"

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

void gmresMILU_CGS(const struct0_T *A, const emxArray_real_T *b, const
                   emxArray_struct1_T *M, int restart, double rtol, int maxit,
                   const emxArray_real_T *x0, int verbose, int nthreads,
                   emxArray_real_T *x, int *flag, int *iter, emxArray_real_T
                   *resids)
{
  double beta2;
  int ii;
  double beta0;
  int jj;
  int max_outer_iters;
  int loop_ub;
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
  emxArray_real_T *a;
  emxArray_real_T *b_b;
  boolean_T exitg1;
  boolean_T guard1 = false;
  int j;
  int exitg2;
  double vnorm;
  double tmpv;
  beta2 = 0.0;
  for (ii = 0; ii + 1 <= b->size[0]; ii++) {
    beta2 += b->data[ii] * b->data[ii];
  }

  beta0 = sqrt(beta2);
  if (beta0 == 0.0) {
    jj = x->size[0];
    x->size[0] = b->size[0];
    emxEnsureCapacity((emxArray__common *)x, jj, sizeof(double));
    loop_ub = b->size[0];
    for (jj = 0; jj < loop_ub; jj++) {
      x->data[jj] = 0.0;
    }

    *flag = 0;
    *iter = 0;
    jj = resids->size[0];
    resids->size[0] = 1;
    emxEnsureCapacity((emxArray__common *)resids, jj, sizeof(double));
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
      jj = x->size[0];
      x->size[0] = b->size[0];
      emxEnsureCapacity((emxArray__common *)x, jj, sizeof(double));
      loop_ub = b->size[0];
      for (jj = 0; jj < loop_ub; jj++) {
        x->data[jj] = 0.0;
      }
    } else {
      jj = x->size[0];
      x->size[0] = x0->size[0];
      emxEnsureCapacity((emxArray__common *)x, jj, sizeof(double));
      loop_ub = x0->size[0];
      for (jj = 0; jj < loop_ub; jj++) {
        x->data[jj] = x0->data[jj];
      }
    }

    emxInit_real_T(&y, 1);
    jj = y->size[0];
    y->size[0] = restart + 1;
    emxEnsureCapacity((emxArray__common *)y, jj, sizeof(double));
    for (jj = 0; jj <= restart; jj++) {
      y->data[jj] = 0.0;
    }

    emxInit_real_T(&R, 2);
    jj = R->size[0] * R->size[1];
    R->size[0] = restart;
    R->size[1] = restart;
    emxEnsureCapacity((emxArray__common *)R, jj, sizeof(double));
    loop_ub = restart * restart;
    for (jj = 0; jj < loop_ub; jj++) {
      R->data[jj] = 0.0;
    }

    emxInit_real_T(&Q, 2);
    jj = Q->size[0] * Q->size[1];
    Q->size[0] = b->size[0];
    Q->size[1] = restart;
    emxEnsureCapacity((emxArray__common *)Q, jj, sizeof(double));
    loop_ub = b->size[0] * restart;
    for (jj = 0; jj < loop_ub; jj++) {
      Q->data[jj] = 0.0;
    }

    emxInit_real_T(&Z, 2);
    jj = Z->size[0] * Z->size[1];
    Z->size[0] = b->size[0];
    Z->size[1] = restart;
    emxEnsureCapacity((emxArray__common *)Z, jj, sizeof(double));
    loop_ub = b->size[0] * restart;
    for (jj = 0; jj < loop_ub; jj++) {
      Z->data[jj] = 0.0;
    }

    emxInit_real_T(&J, 2);
    jj = J->size[0] * J->size[1];
    J->size[0] = 2;
    J->size[1] = restart;
    emxEnsureCapacity((emxArray__common *)J, jj, sizeof(double));
    loop_ub = restart << 1;
    for (jj = 0; jj < loop_ub; jj++) {
      J->data[jj] = 0.0;
    }

    emxInit_real_T(&v, 1);
    jj = v->size[0];
    v->size[0] = b->size[0];
    emxEnsureCapacity((emxArray__common *)v, jj, sizeof(double));
    loop_ub = b->size[0];
    for (jj = 0; jj < loop_ub; jj++) {
      v->data[jj] = 0.0;
    }

    emxInit_real_T(&v2, 1);
    jj = v2->size[0];
    v2->size[0] = M->data[0].negE.nrows;
    emxEnsureCapacity((emxArray__common *)v2, jj, sizeof(double));
    loop_ub = M->data[0].negE.nrows;
    for (jj = 0; jj < loop_ub; jj++) {
      v2->data[jj] = 0.0;
    }

    jj = resids->size[0];
    resids->size[0] = maxit;
    emxEnsureCapacity((emxArray__common *)resids, jj, sizeof(double));
    for (jj = 0; jj < maxit; jj++) {
      resids->data[jj] = 0.0;
    }

    *flag = 0;
    *iter = 0;
    resid = 1.0;
    it_outer = 1;
    emxInit_real_T(&w, 1);
    emxInit_real_T(&a, 2);
    emxInit_real_T(&b_b, 1);
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
          jj = v->size[0];
          v->size[0] = b->size[0];
          emxEnsureCapacity((emxArray__common *)v, jj, sizeof(double));
          loop_ub = b->size[0];
          for (jj = 0; jj < loop_ub; jj++) {
            v->data[jj] = b->data[jj];
          }
        }
      }

      if (guard1) {
        crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, x, v, nthreads);
        jj = v->size[0];
        v->size[0] = b->size[0];
        emxEnsureCapacity((emxArray__common *)v, jj, sizeof(double));
        loop_ub = b->size[0];
        for (jj = 0; jj < loop_ub; jj++) {
          v->data[jj] = b->data[jj] - v->data[jj];
        }
      }

      beta2 = 0.0;
      for (ii = 0; ii + 1 <= v->size[0]; ii++) {
        beta2 += v->data[ii] * v->data[ii];
      }

      beta2 = sqrt(beta2);
      y->data[0] = beta2;
      loop_ub = v->size[0];
      for (jj = 0; jj < loop_ub; jj++) {
        Q->data[jj] = v->data[jj] / beta2;
      }

      j = 0;
      do {
        exitg2 = 0;
        loop_ub = Q->size[0];
        jj = w->size[0];
        w->size[0] = loop_ub;
        emxEnsureCapacity((emxArray__common *)w, jj, sizeof(double));
        for (jj = 0; jj < loop_ub; jj++) {
          w->data[jj] = Q->data[jj + Q->size[0] * j];
        }

        solve_milu(M, 1, w, 0, v, v2);
        loop_ub = w->size[0];
        for (jj = 0; jj < loop_ub; jj++) {
          Z->data[jj + Z->size[0] * j] = w->data[jj];
        }

        crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, w, v, nthreads);
        jj = w->size[0];
        w->size[0] = v->size[0];
        emxEnsureCapacity((emxArray__common *)w, jj, sizeof(double));
        loop_ub = v->size[0];
        for (jj = 0; jj < loop_ub; jj++) {
          w->data[jj] = v->data[jj];
        }

        for (ii = 0; ii + 1 <= j + 1; ii++) {
          jj = a->size[0] * a->size[1];
          a->size[0] = 1;
          a->size[1] = w->size[0];
          emxEnsureCapacity((emxArray__common *)a, jj, sizeof(double));
          loop_ub = w->size[0];
          for (jj = 0; jj < loop_ub; jj++) {
            a->data[a->size[0] * jj] = w->data[jj];
          }

          loop_ub = Q->size[0];
          jj = b_b->size[0];
          b_b->size[0] = loop_ub;
          emxEnsureCapacity((emxArray__common *)b_b, jj, sizeof(double));
          for (jj = 0; jj < loop_ub; jj++) {
            b_b->data[jj] = Q->data[jj + Q->size[0] * ii];
          }

          guard1 = false;
          if (a->size[1] == 1) {
            guard1 = true;
          } else {
            jj = Q->size[0];
            if (jj == 1) {
              guard1 = true;
            } else {
              beta2 = 0.0;
              for (jj = 0; jj < a->size[1]; jj++) {
                beta2 += a->data[a->size[0] * jj] * b_b->data[jj];
              }
            }
          }

          if (guard1) {
            beta2 = 0.0;
            for (jj = 0; jj < a->size[1]; jj++) {
              beta2 += a->data[a->size[0] * jj] * b_b->data[jj];
            }
          }

          R->data[ii + R->size[0] * j] = beta2;
          beta2 = R->data[ii + R->size[0] * j];
          jj = v->size[0];
          emxEnsureCapacity((emxArray__common *)v, jj, sizeof(double));
          loop_ub = v->size[0];
          for (jj = 0; jj < loop_ub; jj++) {
            v->data[jj] -= beta2 * Q->data[jj + Q->size[0] * ii];
          }
        }

        beta2 = 0.0;
        for (ii = 0; ii + 1 <= v->size[0]; ii++) {
          beta2 += v->data[ii] * v->data[ii];
        }

        vnorm = sqrt(beta2);
        if (j + 1 < restart) {
          loop_ub = v->size[0];
          for (jj = 0; jj < loop_ub; jj++) {
            Q->data[jj + Q->size[0] * (j + 1)] = v->data[jj] / vnorm;
          }
        }

        for (ii = 0; ii + 1 <= j; ii++) {
          tmpv = R->data[ii + R->size[0] * j];
          R->data[ii + R->size[0] * j] = J->data[J->size[0] * ii] * R->data[ii +
            R->size[0] * j] + J->data[1 + J->size[0] * ii] * R->data[(ii +
            R->size[0] * j) + 1];
          R->data[(ii + R->size[0] * j) + 1] = -J->data[1 + J->size[0] * ii] *
            tmpv + J->data[J->size[0] * ii] * R->data[(ii + R->size[0] * j) + 1];
        }

        beta2 = sqrt(R->data[j + R->size[0] * j] * R->data[j + R->size[0] * j] +
                     beta2);
        J->data[J->size[0] * j] = R->data[j + R->size[0] * j] / beta2;
        J->data[1 + J->size[0] * j] = vnorm / beta2;
        y->data[j + 1] = -J->data[1 + J->size[0] * j] * y->data[j];
        y->data[j] *= J->data[J->size[0] * j];
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

      for (jj = j; jj + 1 > 0; jj--) {
        for (ii = jj + 1; ii + 1 <= j + 1; ii++) {
          y->data[jj] -= R->data[jj + R->size[0] * ii] * y->data[ii];
        }

        y->data[jj] /= R->data[jj + R->size[0] * jj];
      }

      for (ii = 0; ii + 1 <= j + 1; ii++) {
        beta2 = y->data[ii];
        jj = x->size[0];
        emxEnsureCapacity((emxArray__common *)x, jj, sizeof(double));
        loop_ub = x->size[0];
        for (jj = 0; jj < loop_ub; jj++) {
          x->data[jj] += beta2 * Z->data[jj + Z->size[0] * ii];
        }
      }

      if ((resid < rtol) || (*flag != 0)) {
        exitg1 = true;
      } else {
        it_outer++;
      }
    }

    emxFree_real_T(&b_b);
    emxFree_real_T(&a);
    emxFree_real_T(&w);
    emxFree_real_T(&v2);
    emxFree_real_T(&v);
    emxFree_real_T(&J);
    emxFree_real_T(&Z);
    emxFree_real_T(&Q);
    emxFree_real_T(&R);
    emxFree_real_T(&y);
    jj = resids->size[0];
    if (1 > *iter) {
      resids->size[0] = 0;
    } else {
      resids->size[0] = *iter;
    }

    emxEnsureCapacity((emxArray__common *)resids, jj, sizeof(double));
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
