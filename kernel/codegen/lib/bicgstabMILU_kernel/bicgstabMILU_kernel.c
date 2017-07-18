#include "bicgstabMILU_kernel.h"
#include "m2c.h"
#include "omp.h"
#include "ilupack.h"

static void b_m2c_error(const emxArray_char_T *varargin_3);
static int b_mod(int x);
static void b_sqrt(double *x);
static void c_m2c_error(void);
static void crs_prodAx(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
  *A_col_ind, const emxArray_real_T *A_val, int A_nrows, const emxArray_real_T
  *x, emxArray_real_T *b, int nthreads);
static void crs_prodAx_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, emxArray_real_T *b, int nrows, boolean_T ismt);
static int div_nde_s32_floor(int numerator, int denominator);
static void m2c_error(const emxArray_char_T *varargin_3);
static void m2c_printf(int varargin_2, double varargin_3);
static void m2c_warn(void);
static void b_m2c_error(const emxArray_char_T *varargin_3)
{
  emxArray_char_T *b_varargin_3;
  const char * msgid;
  const char * fmt;
  int i2;
  int loop_ub;
  emxInit_char_T(&b_varargin_3, 2);
  msgid = "m2c_opaque_obj:WrongInput";
  fmt = "Incorrect data type %s. Expected DILUPACKparam *.\n";
  i2 = b_varargin_3->size[0] * b_varargin_3->size[1];
  b_varargin_3->size[0] = 1;
  b_varargin_3->size[1] = varargin_3->size[1];
  emxEnsureCapacity_char_T(b_varargin_3, i2);
  loop_ub = varargin_3->size[0] * varargin_3->size[1];
  for (i2 = 0; i2 < loop_ub; i2++) {
    b_varargin_3->data[i2] = varargin_3->data[i2];
  }

  M2C_error(msgid, fmt, &b_varargin_3->data[0]);
  emxFree_char_T(&b_varargin_3);
}

static int b_mod(int x)
{
  return x - div_nde_s32_floor(x, 30) * 30;
}

static void b_sqrt(double *x)
{
  *x = sqrt(*x);
}

static void c_m2c_error(void)
{
  const char * msgid;
  const char * fmt;
  msgid = "crs_prodAx:BufferTooSmal";
  fmt = "Buffer space for output b is too small.";
  M2C_error(msgid, fmt);
}

static void crs_prodAx(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
  *A_col_ind, const emxArray_real_T *A_val, int A_nrows, const emxArray_real_T
  *x, emxArray_real_T *b, int nthreads)
{
  int n;
  int b_n;
  if (b->size[0] < A_nrows) {
    c_m2c_error();
  }

  n = omp_get_num_threads();
  b_n = omp_get_nested();
  if ((!(b_n != 0)) && (n > 1) && (nthreads > 1)) {

#pragma omp master
    {
      m2c_warn();
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
  *x, emxArray_real_T *b, int nrows, boolean_T ismt)
{
  int istart;
  int iend;
  double t;
  int chunk;
  int b_remainder;
  if (ismt) {
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

static void m2c_error(const emxArray_char_T *varargin_3)
{
  emxArray_char_T *b_varargin_3;
  const char * msgid;
  const char * fmt;
  int i1;
  int loop_ub;
  emxInit_char_T(&b_varargin_3, 2);
  msgid = "m2c_opaque_obj:WrongInput";
  fmt = "Incorrect data type %s. Expected DAMGlevelmat *.\n";
  i1 = b_varargin_3->size[0] * b_varargin_3->size[1];
  b_varargin_3->size[0] = 1;
  b_varargin_3->size[1] = varargin_3->size[1];
  emxEnsureCapacity_char_T(b_varargin_3, i1);
  loop_ub = varargin_3->size[0] * varargin_3->size[1];
  for (i1 = 0; i1 < loop_ub; i1++) {
    b_varargin_3->data[i1] = varargin_3->data[i1];
  }

  M2C_error(msgid, fmt, &b_varargin_3->data[0]);
  emxFree_char_T(&b_varargin_3);
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

void bicgstabMILU_kernel(const struct0_T *A, const emxArray_real_T *b, const
  struct1_T *prec, double rtol, int maxit, const emxArray_real_T *x0, int
  verbose, int nthreads, const struct1_T *param, const emxArray_real_T *rowscal,
  const emxArray_real_T *colscal, emxArray_real_T *x, int *flag, int *iter,
  emxArray_real_T *resids)
{
  double resid;
  int ii;
  double bnrm2;
  int i0;
  emxArray_real_T *v;
  emxArray_real_T *p;
  emxArray_real_T *p_hat;
  emxArray_real_T *dbuff;
  boolean_T need_rowscaling;
  boolean_T need_colscaling;
  boolean_T exitg1;
  emxArray_char_T *b_prec;
  static const char cv0[14] = { 'D', 'A', 'M', 'G', 'l', 'e', 'v', 'e', 'l', 'm',
    'a', 't', ' ', '*' };

  emxArray_uint8_T *data;
  DAMGlevelmat * t_prec;
  static const char cv1[15] = { 'D', 'I', 'L', 'U', 'P', 'A', 'C', 'K', 'p', 'a',
    'r', 'a', 'm', ' ', '*' };

  DILUPACKparam * t_param;
  emxArray_real_T *r;
  emxArray_real_T *r_tld;
  double omega;
  double alpha;
  double rho_1;
  emxArray_real_T *a;
  int exitg2;
  double rho;
  *flag = 0;
  *iter = 0;
  resid = 0.0;
  for (ii = 0; ii + 1 <= b->size[0]; ii++) {
    resid += b->data[ii] * b->data[ii];
  }

  bnrm2 = sqrt(resid);
  if (bnrm2 == 0.0) {
    i0 = x->size[0];
    x->size[0] = b->size[0];
    emxEnsureCapacity_real_T(x, i0);
    ii = b->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      x->data[i0] = 0.0;
    }

    i0 = resids->size[0];
    resids->size[0] = 1;
    emxEnsureCapacity_real_T(resids, i0);
    resids->data[0] = 0.0;
  } else {
    if (x0->size[0] == 0) {
      i0 = x->size[0];
      x->size[0] = b->size[0];
      emxEnsureCapacity_real_T(x, i0);
      ii = b->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        x->data[i0] = 0.0;
      }
    } else {
      i0 = x->size[0];
      x->size[0] = x0->size[0];
      emxEnsureCapacity_real_T(x, i0);
      ii = x0->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        x->data[i0] = x0->data[i0];
      }
    }

    emxInit_real_T(&v, 1);
    i0 = v->size[0];
    v->size[0] = b->size[0];
    emxEnsureCapacity_real_T(v, i0);
    ii = b->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      v->data[i0] = 0.0;
    }

    emxInit_real_T(&p, 1);
    i0 = p->size[0];
    p->size[0] = b->size[0];
    emxEnsureCapacity_real_T(p, i0);
    ii = b->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      p->data[i0] = 0.0;
    }

    emxInit_real_T(&p_hat, 1);
    i0 = p_hat->size[0];
    p_hat->size[0] = b->size[0];
    emxEnsureCapacity_real_T(p_hat, i0);
    ii = b->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      p_hat->data[i0] = 0.0;
    }

    emxInit_real_T(&dbuff, 1);
    ii = 3 * b->size[0];
    i0 = dbuff->size[0];
    dbuff->size[0] = ii;
    emxEnsureCapacity_real_T(dbuff, i0);
    for (i0 = 0; i0 < ii; i0++) {
      dbuff->data[i0] = 0.0;
    }

    i0 = resids->size[0];
    resids->size[0] = maxit;
    emxEnsureCapacity_real_T(resids, i0);
    for (i0 = 0; i0 < maxit; i0++) {
      resids->data[i0] = 0.0;
    }

    need_rowscaling = false;
    need_colscaling = false;
    if (prec->type->size[1] == 14) {
      need_colscaling = true;
    }

    if (need_colscaling && (!(prec->type->size[1] == 0))) {
      ii = 0;
      exitg1 = false;
      while ((!exitg1) && (ii < 14)) {
        if (!(prec->type->data[ii] == cv0[ii])) {
          need_colscaling = false;
          exitg1 = true;
        } else {
          ii++;
        }
      }
    }

    if (need_colscaling) {
      need_rowscaling = true;
    }

    emxInit_char_T(&b_prec, 2);
    if (!need_rowscaling) {
      i0 = b_prec->size[0] * b_prec->size[1];
      b_prec->size[0] = 1;
      b_prec->size[1] = prec->type->size[1] + 1;
      emxEnsureCapacity_char_T(b_prec, i0);
      ii = prec->type->size[1];
      for (i0 = 0; i0 < ii; i0++) {
        b_prec->data[b_prec->size[0] * i0] = prec->type->data[prec->type->size[0]
          * i0];
      }

      b_prec->data[b_prec->size[0] * prec->type->size[1]] = '\x00';
      m2c_error(b_prec);
    }

    emxInit_uint8_T(&data, 1);
    i0 = data->size[0];
    data->size[0] = prec->data->size[0];
    emxEnsureCapacity_uint8_T(data, i0);
    ii = prec->data->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      data->data[i0] = prec->data->data[i0];
    }

    t_prec = *(DAMGlevelmat **)(&data->data[0]);
    need_rowscaling = false;
    need_colscaling = false;
    if (param->type->size[1] == 15) {
      need_colscaling = true;
    }

    if (need_colscaling && (!(param->type->size[1] == 0))) {
      ii = 0;
      exitg1 = false;
      while ((!exitg1) && (ii < 15)) {
        if (!(param->type->data[ii] == cv1[ii])) {
          need_colscaling = false;
          exitg1 = true;
        } else {
          ii++;
        }
      }
    }

    if (need_colscaling) {
      need_rowscaling = true;
    }

    if (!need_rowscaling) {
      i0 = b_prec->size[0] * b_prec->size[1];
      b_prec->size[0] = 1;
      b_prec->size[1] = param->type->size[1] + 1;
      emxEnsureCapacity_char_T(b_prec, i0);
      ii = param->type->size[1];
      for (i0 = 0; i0 < ii; i0++) {
        b_prec->data[b_prec->size[0] * i0] = param->type->data[param->type->
          size[0] * i0];
      }

      b_prec->data[b_prec->size[0] * param->type->size[1]] = '\x00';
      b_m2c_error(b_prec);
    }

    emxFree_char_T(&b_prec);
    i0 = data->size[0];
    data->size[0] = param->data->size[0];
    emxEnsureCapacity_uint8_T(data, i0);
    ii = param->data->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      data->data[i0] = param->data->data[i0];
    }

    t_param = *(DILUPACKparam **)(&data->data[0]);
    need_rowscaling = false;
    i0 = rowscal->size[0];
    ii = 1;
    emxFree_uint8_T(&data);
    exitg1 = false;
    while ((!exitg1) && (ii <= i0)) {
      if (rowscal->data[ii - 1] != 1.0) {
        need_rowscaling = true;
        exitg1 = true;
      } else {
        ii++;
      }
    }

    need_colscaling = false;
    i0 = colscal->size[0];
    ii = 1;
    exitg1 = false;
    while ((!exitg1) && (ii <= i0)) {
      if (colscal->data[ii - 1] != 1.0) {
        need_colscaling = true;
        exitg1 = true;
      } else {
        ii++;
      }
    }

    resid = 0.0;
    for (ii = 0; ii + 1 <= x->size[0]; ii++) {
      resid += x->data[ii] * x->data[ii];
    }

    emxInit_real_T(&r, 1);
    if (resid > 0.0) {
      i0 = r->size[0];
      r->size[0] = b->size[0];
      emxEnsureCapacity_real_T(r, i0);
      ii = b->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        r->data[i0] = 0.0;
      }

      crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, x, r, nthreads);
      i0 = r->size[0];
      r->size[0] = b->size[0];
      emxEnsureCapacity_real_T(r, i0);
      ii = b->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        r->data[i0] = b->data[i0] - r->data[i0];
      }
    } else {
      i0 = r->size[0];
      r->size[0] = b->size[0];
      emxEnsureCapacity_real_T(r, i0);
      ii = b->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        r->data[i0] = b->data[i0];
      }
    }

    resid = 0.0;
    for (ii = 0; ii + 1 <= r->size[0]; ii++) {
      resid += r->data[ii] * r->data[ii];
    }

    resid = sqrt(resid) / bnrm2;
    if (resid < rtol) {
      i0 = resids->size[0];
      resids->size[0] = 1;
      emxEnsureCapacity_real_T(resids, i0);
      resids->data[0] = 0.0;
    } else {
      emxInit_real_T(&r_tld, 1);
      omega = 1.0;
      alpha = 0.0;
      rho_1 = 0.0;
      i0 = r_tld->size[0];
      r_tld->size[0] = r->size[0];
      emxEnsureCapacity_real_T(r_tld, i0);
      ii = r->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        r_tld->data[i0] = r->data[i0];
      }

      *iter = 1;
      emxInit_real_T(&a, 2);
      do {
        exitg2 = 0;
        i0 = a->size[0] * a->size[1];
        a->size[0] = 1;
        a->size[1] = r_tld->size[0];
        emxEnsureCapacity_real_T(a, i0);
        ii = r_tld->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          a->data[a->size[0] * i0] = r_tld->data[i0];
        }

        if ((a->size[1] == 1) || (r->size[0] == 1)) {
          rho = 0.0;
          for (i0 = 0; i0 < a->size[1]; i0++) {
            rho += a->data[a->size[0] * i0] * r->data[i0];
          }
        } else {
          rho = 0.0;
          for (i0 = 0; i0 < a->size[1]; i0++) {
            rho += a->data[a->size[0] * i0] * r->data[i0];
          }
        }

        if (rho == 0.0) {
          exitg2 = 1;
        } else {
          if (*iter > 1) {
            resid = rho / rho_1 * (alpha / omega);
            i0 = p->size[0];
            p->size[0] = r->size[0];
            emxEnsureCapacity_real_T(p, i0);
            ii = r->size[0];
            for (i0 = 0; i0 < ii; i0++) {
              p->data[i0] = r->data[i0] + resid * (p->data[i0] - omega * v->
                data[i0]);
            }
          } else {
            i0 = p->size[0];
            p->size[0] = r->size[0];
            emxEnsureCapacity_real_T(p, i0);
            ii = r->size[0];
            for (i0 = 0; i0 < ii; i0++) {
              p->data[i0] = r->data[i0];
            }
          }

          if (need_rowscaling) {
            i0 = v->size[0];
            v->size[0] = p->size[0];
            emxEnsureCapacity_real_T(v, i0);
            ii = p->size[0];
            for (i0 = 0; i0 < ii; i0++) {
              v->data[i0] = p->data[i0] * rowscal->data[i0];
            }
          }

          DGNLAMGsol_internal(t_prec, t_param, &v->data[0], &p_hat->data[0],
                              &dbuff->data[0]);
          if (need_colscaling) {
            i0 = p_hat->size[0];
            emxEnsureCapacity_real_T(p_hat, i0);
            ii = p_hat->size[0];
            for (i0 = 0; i0 < ii; i0++) {
              p_hat->data[i0] *= colscal->data[i0];
            }
          }

          crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, p_hat, v,
                     nthreads);
          i0 = a->size[0] * a->size[1];
          a->size[0] = 1;
          a->size[1] = r_tld->size[0];
          emxEnsureCapacity_real_T(a, i0);
          ii = r_tld->size[0];
          for (i0 = 0; i0 < ii; i0++) {
            a->data[a->size[0] * i0] = r_tld->data[i0];
          }

          if ((a->size[1] == 1) || (v->size[0] == 1)) {
            rho_1 = 0.0;
            for (i0 = 0; i0 < a->size[1]; i0++) {
              rho_1 += a->data[a->size[0] * i0] * v->data[i0];
            }
          } else {
            rho_1 = 0.0;
            for (i0 = 0; i0 < a->size[1]; i0++) {
              rho_1 += a->data[a->size[0] * i0] * v->data[i0];
            }
          }

          alpha = rho / rho_1;
          i0 = x->size[0];
          emxEnsureCapacity_real_T(x, i0);
          ii = x->size[0];
          for (i0 = 0; i0 < ii; i0++) {
            x->data[i0] += alpha * p_hat->data[i0];
          }

          i0 = r->size[0];
          emxEnsureCapacity_real_T(r, i0);
          ii = r->size[0];
          for (i0 = 0; i0 < ii; i0++) {
            r->data[i0] -= alpha * v->data[i0];
          }

          resid = 0.0;
          for (ii = 0; ii + 1 <= r->size[0]; ii++) {
            resid += r->data[ii] * r->data[ii];
          }

          resid = sqrt(resid);
          if (resid < rtol) {
            resid /= bnrm2;
            resids->data[*iter - 1] = resid;
            exitg2 = 1;
          } else {
            if (need_rowscaling) {
              i0 = v->size[0];
              v->size[0] = r->size[0];
              emxEnsureCapacity_real_T(v, i0);
              ii = r->size[0];
              for (i0 = 0; i0 < ii; i0++) {
                v->data[i0] = r->data[i0] * rowscal->data[i0];
              }
            }

            DGNLAMGsol_internal(t_prec, t_param, &v->data[0], &p_hat->data[0],
                                &dbuff->data[0]);
            if (need_colscaling) {
              i0 = p_hat->size[0];
              emxEnsureCapacity_real_T(p_hat, i0);
              ii = p_hat->size[0];
              for (i0 = 0; i0 < ii; i0++) {
                p_hat->data[i0] *= colscal->data[i0];
              }
            }

            crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, p_hat, v,
                       nthreads);
            i0 = a->size[0] * a->size[1];
            a->size[0] = 1;
            a->size[1] = v->size[0];
            emxEnsureCapacity_real_T(a, i0);
            ii = v->size[0];
            for (i0 = 0; i0 < ii; i0++) {
              a->data[a->size[0] * i0] = v->data[i0];
            }

            if ((a->size[1] == 1) || (r->size[0] == 1)) {
              rho_1 = 0.0;
              for (i0 = 0; i0 < a->size[1]; i0++) {
                rho_1 += a->data[a->size[0] * i0] * r->data[i0];
              }
            } else {
              rho_1 = 0.0;
              for (i0 = 0; i0 < a->size[1]; i0++) {
                rho_1 += a->data[a->size[0] * i0] * r->data[i0];
              }
            }

            resid = 0.0;
            for (ii = 0; ii + 1 <= v->size[0]; ii++) {
              resid += v->data[ii] * v->data[ii];
            }

            omega = rho_1 / resid;
            i0 = x->size[0];
            emxEnsureCapacity_real_T(x, i0);
            ii = x->size[0];
            for (i0 = 0; i0 < ii; i0++) {
              x->data[i0] += omega * p_hat->data[i0];
            }

            i0 = r->size[0];
            emxEnsureCapacity_real_T(r, i0);
            ii = r->size[0];
            for (i0 = 0; i0 < ii; i0++) {
              r->data[i0] -= omega * v->data[i0];
            }

            resid = 0.0;
            for (ii = 0; ii + 1 <= r->size[0]; ii++) {
              resid += r->data[ii] * r->data[ii];
            }

            b_sqrt(&resid);
            resid /= bnrm2;
            resids->data[*iter - 1] = resid;
            if ((verbose > 1) || ((verbose > 0) && (b_mod(*iter) == 0))) {
              m2c_printf(*iter, resid);
            }

            if (resid <= rtol) {
              exitg2 = 1;
            } else if (resid > 100.0) {
              *flag = -3;
              exitg2 = 1;
            } else if (omega == 0.0) {
              exitg2 = 1;
            } else {
              rho_1 = rho;
              if (*iter >= maxit) {
                exitg2 = 1;
              } else {
                (*iter)++;
              }
            }
          }
        }
      } while (exitg2 == 0);

      emxFree_real_T(&a);
      emxFree_real_T(&r_tld);
      i0 = resids->size[0];
      resids->size[0] = *iter;
      emxEnsureCapacity_real_T(resids, i0);
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

    emxFree_real_T(&dbuff);
    emxFree_real_T(&p_hat);
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
