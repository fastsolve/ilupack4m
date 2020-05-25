#include "bicgstabMILU_kernel.h"
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
static int div_nde_s32_floor(int numerator, int denominator);
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

static int div_nde_s32_floor(int numerator, int denominator)
{
  int b_numerator;
  if (((numerator < 0) != (denominator < 0)) && (numerator % denominator != 0))
  {
    b_numerator = -1;
  } else {
    b_numerator = 0;
  }

  return numerator / denominator + b_numerator;
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

void bicgstabMILU_kernel(const struct0_T *A, const emxArray_real_T *b, const
  emxArray_struct1_T *M, double rtol, int maxit, const emxArray_real_T *x0, int
  verbose, int nthreads, emxArray_real_T *x, int *flag, int *iter,
  emxArray_real_T *resids)
{
  double rho_1;
  int i;
  int ii;
  double bnrm2;
  emxArray_real_T *v;
  emxArray_real_T *p;
  emxArray_real_T *y2;
  emxArray_real_T *r;
  double resid;
  emxArray_real_T *r_tld;
  double omega;
  double alpha;
  emxArray_real_T *p_hat;
  int exitg1;
  double b_r_tld;
  *flag = 0;
  *iter = 0;
  rho_1 = 0.0;
  i = b->size[0];
  for (ii = 0; ii < i; ii++) {
    rho_1 += b->data[ii] * b->data[ii];
  }

  bnrm2 = sqrt(rho_1);
  if (bnrm2 == 0.0) {
    i = x->size[0];
    x->size[0] = b->size[0];
    emxEnsureCapacity_real_T(x, i);
    ii = b->size[0];
    for (i = 0; i < ii; i++) {
      x->data[i] = 0.0;
    }

    i = resids->size[0];
    resids->size[0] = 1;
    emxEnsureCapacity_real_T(resids, i);
    resids->data[0] = 0.0;
  } else {
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

    emxInit_real_T(&v, 1);
    i = v->size[0];
    v->size[0] = b->size[0];
    emxEnsureCapacity_real_T(v, i);
    ii = b->size[0];
    for (i = 0; i < ii; i++) {
      v->data[i] = 0.0;
    }

    emxInit_real_T(&p, 1);
    i = p->size[0];
    p->size[0] = b->size[0];
    emxEnsureCapacity_real_T(p, i);
    ii = b->size[0];
    for (i = 0; i < ii; i++) {
      p->data[i] = 0.0;
    }

    emxInit_real_T(&y2, 1);
    i = y2->size[0];
    y2->size[0] = M->data[0].negE.nrows;
    emxEnsureCapacity_real_T(y2, i);
    ii = M->data[0].negE.nrows;
    for (i = 0; i < ii; i++) {
      y2->data[i] = 0.0;
    }

    i = resids->size[0];
    resids->size[0] = maxit;
    emxEnsureCapacity_real_T(resids, i);
    for (i = 0; i < maxit; i++) {
      resids->data[i] = 0.0;
    }

    rho_1 = 0.0;
    i = x->size[0];
    for (ii = 0; ii < i; ii++) {
      rho_1 += x->data[ii] * x->data[ii];
    }

    emxInit_real_T(&r, 1);
    if (rho_1 > 0.0) {
      i = r->size[0];
      r->size[0] = b->size[0];
      emxEnsureCapacity_real_T(r, i);
      ii = b->size[0];
      for (i = 0; i < ii; i++) {
        r->data[i] = 0.0;
      }

      crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, x, r, nthreads);
      i = r->size[0];
      r->size[0] = b->size[0];
      emxEnsureCapacity_real_T(r, i);
      ii = b->size[0];
      for (i = 0; i < ii; i++) {
        r->data[i] = b->data[i] - r->data[i];
      }
    } else {
      i = r->size[0];
      r->size[0] = b->size[0];
      emxEnsureCapacity_real_T(r, i);
      ii = b->size[0];
      for (i = 0; i < ii; i++) {
        r->data[i] = b->data[i];
      }
    }

    rho_1 = 0.0;
    i = r->size[0];
    for (ii = 0; ii < i; ii++) {
      rho_1 += r->data[ii] * r->data[ii];
    }

    resid = sqrt(rho_1) / bnrm2;
    if (resid < rtol) {
      i = resids->size[0];
      resids->size[0] = 1;
      emxEnsureCapacity_real_T(resids, i);
      resids->data[0] = 0.0;
    } else {
      emxInit_real_T(&r_tld, 1);
      omega = 1.0;
      alpha = 0.0;
      rho_1 = 0.0;
      i = r_tld->size[0];
      r_tld->size[0] = r->size[0];
      emxEnsureCapacity_real_T(r_tld, i);
      ii = r->size[0];
      for (i = 0; i < ii; i++) {
        r_tld->data[i] = r->data[i];
      }

      *iter = 1;
      emxInit_real_T(&p_hat, 1);
      do {
        exitg1 = 0;
        b_r_tld = 0.0;
        ii = r_tld->size[0];
        for (i = 0; i < ii; i++) {
          b_r_tld += r_tld->data[i] * r->data[i];
        }

        if (b_r_tld == 0.0) {
          exitg1 = 1;
        } else {
          if (*iter > 1) {
            resid = b_r_tld / rho_1 * (alpha / omega);
            i = p->size[0];
            p->size[0] = r->size[0];
            emxEnsureCapacity_real_T(p, i);
            ii = r->size[0];
            for (i = 0; i < ii; i++) {
              p->data[i] = r->data[i] + resid * (p->data[i] - omega * v->data[i]);
            }
          } else {
            i = p->size[0];
            p->size[0] = r->size[0];
            emxEnsureCapacity_real_T(p, i);
            ii = r->size[0];
            for (i = 0; i < ii; i++) {
              p->data[i] = r->data[i];
            }
          }

          i = p_hat->size[0];
          p_hat->size[0] = p->size[0];
          emxEnsureCapacity_real_T(p_hat, i);
          ii = p->size[0];
          for (i = 0; i < ii; i++) {
            p_hat->data[i] = p->data[i];
          }

          solve_milu(M, 1, p_hat, 0, v, y2);
          crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, p_hat, v,
                     nthreads);
          resid = 0.0;
          ii = r_tld->size[0];
          for (i = 0; i < ii; i++) {
            resid += r_tld->data[i] * v->data[i];
          }

          alpha = b_r_tld / resid;
          ii = x->size[0];
          for (i = 0; i < ii; i++) {
            x->data[i] += alpha * p_hat->data[i];
          }

          ii = r->size[0];
          for (i = 0; i < ii; i++) {
            r->data[i] -= alpha * v->data[i];
          }

          rho_1 = 0.0;
          i = r->size[0];
          for (ii = 0; ii < i; ii++) {
            rho_1 += r->data[ii] * r->data[ii];
          }

          resid = sqrt(rho_1);
          if (resid < rtol) {
            resid /= bnrm2;
            resids->data[*iter - 1] = resid;
            exitg1 = 1;
          } else {
            i = p_hat->size[0];
            p_hat->size[0] = r->size[0];
            emxEnsureCapacity_real_T(p_hat, i);
            ii = r->size[0];
            for (i = 0; i < ii; i++) {
              p_hat->data[i] = r->data[i];
            }

            solve_milu(M, 1, p_hat, 0, v, y2);
            crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, p_hat, v,
                       nthreads);
            rho_1 = 0.0;
            i = v->size[0];
            resid = 0.0;
            for (ii = 0; ii < i; ii++) {
              rho_1 += v->data[ii] * v->data[ii];
              resid += v->data[ii] * r->data[ii];
            }

            omega = resid / rho_1;
            ii = x->size[0];
            for (i = 0; i < ii; i++) {
              x->data[i] += omega * p_hat->data[i];
            }

            ii = r->size[0];
            for (i = 0; i < ii; i++) {
              r->data[i] -= omega * v->data[i];
            }

            rho_1 = 0.0;
            i = r->size[0];
            for (ii = 0; ii < i; ii++) {
              rho_1 += r->data[ii] * r->data[ii];
            }

            resid = sqrt(rho_1) / bnrm2;
            resids->data[*iter - 1] = resid;
            if ((verbose > 1) || ((verbose > 0) && (*iter - div_nde_s32_floor
                  (*iter, 30) * 30 == 0))) {
              m2c_printf(*iter, resid);
            }

            if (resid <= rtol) {
              exitg1 = 1;
            } else if (resid > 100.0) {
              *flag = -3;
              exitg1 = 1;
            } else if (omega == 0.0) {
              exitg1 = 1;
            } else {
              rho_1 = b_r_tld;
              if (*iter >= maxit) {
                exitg1 = 1;
              } else {
                (*iter)++;
              }
            }
          }
        }
      } while (exitg1 == 0);

      emxFree_real_T(&p_hat);
      emxFree_real_T(&r_tld);
      i = resids->size[0];
      resids->size[0] = *iter;
      emxEnsureCapacity_real_T(resids, i);
      if (resid <= rtol) {
        *flag = 0;
      } else if (omega == 0.0) {
        *flag = -2;
      } else if (b_r_tld == 0.0) {
        *flag = -1;
      } else {
        if (*flag == 0) {
          *flag = 1;
        }
      }
    }

    emxFree_real_T(&y2);
    emxFree_real_T(&p);
    emxFree_real_T(&v);
    emxFree_real_T(&r);
  }
}

void bicgstabMILU_kernel_initialize(void)
{
}

void bicgstabMILU_kernel_terminate(void)
{
}
