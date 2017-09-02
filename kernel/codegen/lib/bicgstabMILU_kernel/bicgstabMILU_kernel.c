#include "bicgstabMILU_kernel.h"
#include "m2c.h"
#include "omp.h"

static void MILUsolve(const emxArray_struct1_T *M, emxArray_real_T *b,
                      emxArray_real_T *b_y1, emxArray_real_T *y2);
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
static void MILUsolve(const emxArray_struct1_T *M, emxArray_real_T *b,
                      emxArray_real_T *b_y1, emxArray_real_T *y2)
{
  solve_milu(M, 1, b, 0, b_y1, y2);
}

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

void bicgstabMILU_kernel(const struct0_T *A, const emxArray_real_T *b, const
  emxArray_struct1_T *M, double rtol, int maxit, const emxArray_real_T *x0, int
  verbose, int nthreads, emxArray_real_T *x, int *flag, int *iter,
  emxArray_real_T *resids)
{
  double resid;
  int ii;
  double bnrm2;
  int loop_ub;
  emxArray_real_T *v;
  emxArray_real_T *p;
  emxArray_real_T *y2;
  emxArray_real_T *r;
  emxArray_real_T *r_tld;
  double omega;
  double alpha;
  double rho_1;
  emxArray_real_T *p_hat;
  emxArray_real_T *a;
  int exitg1;
  double rho;
  *flag = 0;
  *iter = 0;
  resid = 0.0;
  for (ii = 0; ii + 1 <= b->size[0]; ii++) {
    resid += b->data[ii] * b->data[ii];
  }

  bnrm2 = sqrt(resid);
  if (bnrm2 == 0.0) {
    ii = x->size[0];
    x->size[0] = b->size[0];
    emxEnsureCapacity((emxArray__common *)x, ii, sizeof(double));
    loop_ub = b->size[0];
    for (ii = 0; ii < loop_ub; ii++) {
      x->data[ii] = 0.0;
    }

    ii = resids->size[0];
    resids->size[0] = 1;
    emxEnsureCapacity((emxArray__common *)resids, ii, sizeof(double));
    resids->data[0] = 0.0;
  } else {
    if (x0->size[0] == 0) {
      ii = x->size[0];
      x->size[0] = b->size[0];
      emxEnsureCapacity((emxArray__common *)x, ii, sizeof(double));
      loop_ub = b->size[0];
      for (ii = 0; ii < loop_ub; ii++) {
        x->data[ii] = 0.0;
      }
    } else {
      ii = x->size[0];
      x->size[0] = x0->size[0];
      emxEnsureCapacity((emxArray__common *)x, ii, sizeof(double));
      loop_ub = x0->size[0];
      for (ii = 0; ii < loop_ub; ii++) {
        x->data[ii] = x0->data[ii];
      }
    }

    emxInit_real_T(&v, 1);
    ii = v->size[0];
    v->size[0] = b->size[0];
    emxEnsureCapacity((emxArray__common *)v, ii, sizeof(double));
    loop_ub = b->size[0];
    for (ii = 0; ii < loop_ub; ii++) {
      v->data[ii] = 0.0;
    }

    emxInit_real_T(&p, 1);
    ii = p->size[0];
    p->size[0] = b->size[0];
    emxEnsureCapacity((emxArray__common *)p, ii, sizeof(double));
    loop_ub = b->size[0];
    for (ii = 0; ii < loop_ub; ii++) {
      p->data[ii] = 0.0;
    }

    emxInit_real_T(&y2, 1);
    ii = y2->size[0];
    y2->size[0] = M->data[0].negE.nrows;
    emxEnsureCapacity((emxArray__common *)y2, ii, sizeof(double));
    loop_ub = M->data[0].negE.nrows;
    for (ii = 0; ii < loop_ub; ii++) {
      y2->data[ii] = 0.0;
    }

    ii = resids->size[0];
    resids->size[0] = maxit;
    emxEnsureCapacity((emxArray__common *)resids, ii, sizeof(double));
    for (ii = 0; ii < maxit; ii++) {
      resids->data[ii] = 0.0;
    }

    resid = 0.0;
    for (ii = 0; ii + 1 <= x->size[0]; ii++) {
      resid += x->data[ii] * x->data[ii];
    }

    emxInit_real_T(&r, 1);
    if (resid > 0.0) {
      ii = r->size[0];
      r->size[0] = b->size[0];
      emxEnsureCapacity((emxArray__common *)r, ii, sizeof(double));
      loop_ub = b->size[0];
      for (ii = 0; ii < loop_ub; ii++) {
        r->data[ii] = 0.0;
      }

      crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, x, r, nthreads);
      ii = r->size[0];
      r->size[0] = b->size[0];
      emxEnsureCapacity((emxArray__common *)r, ii, sizeof(double));
      loop_ub = b->size[0];
      for (ii = 0; ii < loop_ub; ii++) {
        r->data[ii] = b->data[ii] - r->data[ii];
      }
    } else {
      ii = r->size[0];
      r->size[0] = b->size[0];
      emxEnsureCapacity((emxArray__common *)r, ii, sizeof(double));
      loop_ub = b->size[0];
      for (ii = 0; ii < loop_ub; ii++) {
        r->data[ii] = b->data[ii];
      }
    }

    resid = 0.0;
    for (ii = 0; ii + 1 <= r->size[0]; ii++) {
      resid += r->data[ii] * r->data[ii];
    }

    resid = sqrt(resid) / bnrm2;
    if (resid < rtol) {
      ii = resids->size[0];
      resids->size[0] = 1;
      emxEnsureCapacity((emxArray__common *)resids, ii, sizeof(double));
      resids->data[0] = 0.0;
    } else {
      emxInit_real_T(&r_tld, 1);
      omega = 1.0;
      alpha = 0.0;
      rho_1 = 0.0;
      ii = r_tld->size[0];
      r_tld->size[0] = r->size[0];
      emxEnsureCapacity((emxArray__common *)r_tld, ii, sizeof(double));
      loop_ub = r->size[0];
      for (ii = 0; ii < loop_ub; ii++) {
        r_tld->data[ii] = r->data[ii];
      }

      *iter = 1;
      emxInit_real_T(&p_hat, 1);
      emxInit_real_T(&a, 2);
      do {
        exitg1 = 0;
        ii = a->size[0] * a->size[1];
        a->size[0] = 1;
        a->size[1] = r_tld->size[0];
        emxEnsureCapacity((emxArray__common *)a, ii, sizeof(double));
        loop_ub = r_tld->size[0];
        for (ii = 0; ii < loop_ub; ii++) {
          a->data[a->size[0] * ii] = r_tld->data[ii];
        }

        if ((a->size[1] == 1) || (r->size[0] == 1)) {
          rho = 0.0;
          for (ii = 0; ii < a->size[1]; ii++) {
            rho += a->data[a->size[0] * ii] * r->data[ii];
          }
        } else {
          rho = 0.0;
          for (ii = 0; ii < a->size[1]; ii++) {
            rho += a->data[a->size[0] * ii] * r->data[ii];
          }
        }

        if (rho == 0.0) {
          exitg1 = 1;
        } else {
          if (*iter > 1) {
            resid = rho / rho_1 * (alpha / omega);
            ii = p->size[0];
            p->size[0] = r->size[0];
            emxEnsureCapacity((emxArray__common *)p, ii, sizeof(double));
            loop_ub = r->size[0];
            for (ii = 0; ii < loop_ub; ii++) {
              p->data[ii] = r->data[ii] + resid * (p->data[ii] - omega * v->
                data[ii]);
            }
          } else {
            ii = p->size[0];
            p->size[0] = r->size[0];
            emxEnsureCapacity((emxArray__common *)p, ii, sizeof(double));
            loop_ub = r->size[0];
            for (ii = 0; ii < loop_ub; ii++) {
              p->data[ii] = r->data[ii];
            }
          }

          ii = p_hat->size[0];
          p_hat->size[0] = p->size[0];
          emxEnsureCapacity((emxArray__common *)p_hat, ii, sizeof(double));
          loop_ub = p->size[0];
          for (ii = 0; ii < loop_ub; ii++) {
            p_hat->data[ii] = p->data[ii];
          }

          MILUsolve(M, p_hat, v, y2);
          crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, p_hat, v,
                     nthreads);
          ii = a->size[0] * a->size[1];
          a->size[0] = 1;
          a->size[1] = r_tld->size[0];
          emxEnsureCapacity((emxArray__common *)a, ii, sizeof(double));
          loop_ub = r_tld->size[0];
          for (ii = 0; ii < loop_ub; ii++) {
            a->data[a->size[0] * ii] = r_tld->data[ii];
          }

          if ((a->size[1] == 1) || (v->size[0] == 1)) {
            rho_1 = 0.0;
            for (ii = 0; ii < a->size[1]; ii++) {
              rho_1 += a->data[a->size[0] * ii] * v->data[ii];
            }
          } else {
            rho_1 = 0.0;
            for (ii = 0; ii < a->size[1]; ii++) {
              rho_1 += a->data[a->size[0] * ii] * v->data[ii];
            }
          }

          alpha = rho / rho_1;
          ii = x->size[0];
          emxEnsureCapacity((emxArray__common *)x, ii, sizeof(double));
          loop_ub = x->size[0];
          for (ii = 0; ii < loop_ub; ii++) {
            x->data[ii] += alpha * p_hat->data[ii];
          }

          ii = r->size[0];
          emxEnsureCapacity((emxArray__common *)r, ii, sizeof(double));
          loop_ub = r->size[0];
          for (ii = 0; ii < loop_ub; ii++) {
            r->data[ii] -= alpha * v->data[ii];
          }

          resid = 0.0;
          for (ii = 0; ii + 1 <= r->size[0]; ii++) {
            resid += r->data[ii] * r->data[ii];
          }

          resid = sqrt(resid);
          if (resid < rtol) {
            resid /= bnrm2;
            resids->data[*iter - 1] = resid;
            exitg1 = 1;
          } else {
            ii = p_hat->size[0];
            p_hat->size[0] = r->size[0];
            emxEnsureCapacity((emxArray__common *)p_hat, ii, sizeof(double));
            loop_ub = r->size[0];
            for (ii = 0; ii < loop_ub; ii++) {
              p_hat->data[ii] = r->data[ii];
            }

            MILUsolve(M, p_hat, v, y2);
            crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, p_hat, v,
                       nthreads);
            ii = a->size[0] * a->size[1];
            a->size[0] = 1;
            a->size[1] = v->size[0];
            emxEnsureCapacity((emxArray__common *)a, ii, sizeof(double));
            loop_ub = v->size[0];
            for (ii = 0; ii < loop_ub; ii++) {
              a->data[a->size[0] * ii] = v->data[ii];
            }

            if ((a->size[1] == 1) || (r->size[0] == 1)) {
              rho_1 = 0.0;
              for (ii = 0; ii < a->size[1]; ii++) {
                rho_1 += a->data[a->size[0] * ii] * r->data[ii];
              }
            } else {
              rho_1 = 0.0;
              for (ii = 0; ii < a->size[1]; ii++) {
                rho_1 += a->data[a->size[0] * ii] * r->data[ii];
              }
            }

            resid = 0.0;
            for (ii = 0; ii + 1 <= v->size[0]; ii++) {
              resid += v->data[ii] * v->data[ii];
            }

            omega = rho_1 / resid;
            ii = x->size[0];
            emxEnsureCapacity((emxArray__common *)x, ii, sizeof(double));
            loop_ub = x->size[0];
            for (ii = 0; ii < loop_ub; ii++) {
              x->data[ii] += omega * p_hat->data[ii];
            }

            ii = r->size[0];
            emxEnsureCapacity((emxArray__common *)r, ii, sizeof(double));
            loop_ub = r->size[0];
            for (ii = 0; ii < loop_ub; ii++) {
              r->data[ii] -= omega * v->data[ii];
            }

            resid = 0.0;
            for (ii = 0; ii + 1 <= r->size[0]; ii++) {
              resid += r->data[ii] * r->data[ii];
            }

            resid = sqrt(resid) / bnrm2;
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
              rho_1 = rho;
              if (*iter >= maxit) {
                exitg1 = 1;
              } else {
                (*iter)++;
              }
            }
          }
        }
      } while (exitg1 == 0);

      emxFree_real_T(&a);
      emxFree_real_T(&p_hat);
      emxFree_real_T(&r_tld);
      ii = resids->size[0];
      resids->size[0] = *iter;
      emxEnsureCapacity((emxArray__common *)resids, ii, sizeof(double));
      if (resid <= rtol) {
        *flag = 0;
      } else if (omega == 0.0) {
        *flag = -2;
      } else if (rho == 0.0) {
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
