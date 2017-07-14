#include "gmresMILU_HO.h"
#include "m2c.h"
#include "omp.h"
#include "ilupack.h"

static boolean_T any(const emxArray_boolean_T *x);
static void b_m2c_error(const emxArray_char_T *varargin_3);
static void backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int cend);
static void c_m2c_error(void);
static void crs_prodAx(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
  *A_col_ind, const emxArray_real_T *A_val, int A_nrows, const emxArray_real_T
  *x, emxArray_real_T *b, int nthreads);
static void crs_prodAx_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, emxArray_real_T *b, int nrows, boolean_T ismt);
static void m2c_error(const emxArray_char_T *varargin_3);
static void m2c_printf(int varargin_2, double varargin_3);
static void m2c_warn(void);
static boolean_T any(const emxArray_boolean_T *x)
{
  boolean_T y;
  int ix;
  boolean_T exitg1;
  boolean_T b0;
  y = false;
  ix = 1;
  exitg1 = false;
  while ((!exitg1) && (ix <= x->size[0])) {
    b0 = !x->data[ix - 1];
    if (!b0) {
      y = true;
      exitg1 = true;
    } else {
      ix++;
    }
  }

  return y;
}

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
  emxEnsureCapacity((emxArray__common *)b_varargin_3, i2, sizeof(char));
  loop_ub = varargin_3->size[0] * varargin_3->size[1];
  for (i2 = 0; i2 < loop_ub; i2++) {
    b_varargin_3->data[i2] = varargin_3->data[i2];
  }

  M2C_error(msgid, fmt, &b_varargin_3->data[0]);
  emxFree_char_T(&b_varargin_3);
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
  emxEnsureCapacity((emxArray__common *)b_varargin_3, i1, sizeof(char));
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

void gmresMILU_HO(const struct0_T *A, const emxArray_real_T *b, const struct1_T *
                  prec, int restart, double rtol, int maxit, const
                  emxArray_real_T *x0, int verbose, int nthreads, const
                  struct1_T *param, const emxArray_real_T *rowscal, const
                  emxArray_real_T *colscal, emxArray_real_T *x, int *flag, int
                  *iter, emxArray_real_T *resids)
{
  int n;
  double resid;
  int ii;
  double beta0;
  int i0;
  int max_outer_iters;
  emxArray_real_T *V;
  emxArray_real_T *R;
  emxArray_real_T *y;
  emxArray_real_T *Z;
  emxArray_real_T *J;
  emxArray_real_T *dx;
  emxArray_real_T *w;
  emxArray_real_T *dbuff;
  boolean_T need_rowscaling;
  boolean_T need_colscaling;
  boolean_T exitg1;
  emxArray_char_T *b_prec;
  static const char cv0[14] = { 'D', 'A', 'M', 'G', 'l', 'e', 'v', 'e', 'l', 'm',
    'a', 't', ' ', '*' };

  emxArray_uint8_T *data;
  DAMGlevelmat * t_prec;
  emxArray_char_T *b_param;
  static const char cv1[15] = { 'D', 'I', 'L', 'U', 'P', 'A', 'C', 'K', 'p', 'a',
    'r', 'a', 'm', ' ', '*' };

  emxArray_boolean_T *b_rowscal;
  DILUPACKparam * t_param;
  emxArray_boolean_T *b_colscal;
  int it_outer;
  emxArray_real_T *u;
  emxArray_real_T *v;
  emxArray_int32_T *r0;
  emxArray_real_T *b_Z;
  boolean_T guard1 = false;
  double beta;
  int j;
  int exitg2;
  int i;
  n = b->size[0];
  resid = 0.0;
  for (ii = 0; ii + 1 <= b->size[0]; ii++) {
    resid += b->data[ii] * b->data[ii];
  }

  beta0 = sqrt(resid);
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

    emxInit_real_T(&dx, 1);
    i0 = dx->size[0];
    dx->size[0] = b->size[0];
    emxEnsureCapacity((emxArray__common *)dx, i0, sizeof(double));
    ii = b->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      dx->data[i0] = 0.0;
    }

    emxInit_real_T(&w, 1);
    i0 = w->size[0];
    w->size[0] = b->size[0];
    emxEnsureCapacity((emxArray__common *)w, i0, sizeof(double));
    ii = b->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      w->data[i0] = 0.0;
    }

    emxInit_real_T(&dbuff, 1);
    ii = 3 * b->size[0];
    i0 = dbuff->size[0];
    dbuff->size[0] = ii;
    emxEnsureCapacity((emxArray__common *)dbuff, i0, sizeof(double));
    for (i0 = 0; i0 < ii; i0++) {
      dbuff->data[i0] = 0.0;
    }

    i0 = resids->size[0];
    resids->size[0] = maxit;
    emxEnsureCapacity((emxArray__common *)resids, i0, sizeof(double));
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

    if (!need_rowscaling) {
      emxInit_char_T(&b_prec, 2);
      i0 = b_prec->size[0] * b_prec->size[1];
      b_prec->size[0] = 1;
      b_prec->size[1] = prec->type->size[1] + 1;
      emxEnsureCapacity((emxArray__common *)b_prec, i0, sizeof(char));
      ii = prec->type->size[1];
      for (i0 = 0; i0 < ii; i0++) {
        b_prec->data[b_prec->size[0] * i0] = prec->type->data[prec->type->size[0]
          * i0];
      }

      b_prec->data[b_prec->size[0] * prec->type->size[1]] = '\x00';
      m2c_error(b_prec);
      emxFree_char_T(&b_prec);
    }

    emxInit_uint8_T(&data, 1);
    i0 = data->size[0];
    data->size[0] = prec->data->size[0];
    emxEnsureCapacity((emxArray__common *)data, i0, sizeof(unsigned char));
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
      emxInit_char_T(&b_param, 2);
      i0 = b_param->size[0] * b_param->size[1];
      b_param->size[0] = 1;
      b_param->size[1] = param->type->size[1] + 1;
      emxEnsureCapacity((emxArray__common *)b_param, i0, sizeof(char));
      ii = param->type->size[1];
      for (i0 = 0; i0 < ii; i0++) {
        b_param->data[b_param->size[0] * i0] = param->type->data[param->
          type->size[0] * i0];
      }

      b_param->data[b_param->size[0] * param->type->size[1]] = '\x00';
      b_m2c_error(b_param);
      emxFree_char_T(&b_param);
    }

    i0 = data->size[0];
    data->size[0] = param->data->size[0];
    emxEnsureCapacity((emxArray__common *)data, i0, sizeof(unsigned char));
    ii = param->data->size[0];
    for (i0 = 0; i0 < ii; i0++) {
      data->data[i0] = param->data->data[i0];
    }

    emxInit_boolean_T(&b_rowscal, 1);
    t_param = *(DILUPACKparam **)(&data->data[0]);
    i0 = b_rowscal->size[0];
    b_rowscal->size[0] = rowscal->size[0];
    emxEnsureCapacity((emxArray__common *)b_rowscal, i0, sizeof(boolean_T));
    ii = rowscal->size[0];
    emxFree_uint8_T(&data);
    for (i0 = 0; i0 < ii; i0++) {
      b_rowscal->data[i0] = (rowscal->data[i0] != 1.0);
    }

    emxInit_boolean_T(&b_colscal, 1);
    need_rowscaling = any(b_rowscal);
    i0 = b_colscal->size[0];
    b_colscal->size[0] = colscal->size[0];
    emxEnsureCapacity((emxArray__common *)b_colscal, i0, sizeof(boolean_T));
    ii = colscal->size[0];
    emxFree_boolean_T(&b_rowscal);
    for (i0 = 0; i0 < ii; i0++) {
      b_colscal->data[i0] = (colscal->data[i0] != 1.0);
    }

    need_colscaling = any(b_colscal);
    *iter = 0;
    resid = 1.0;
    it_outer = 1;
    emxFree_boolean_T(&b_colscal);
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
        resid = 0.0;
        for (ii = 0; ii + 1 <= x->size[0]; ii++) {
          resid += x->data[ii] * x->data[ii];
        }

        if (resid > 0.0) {
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

      resid = 0.0;
      for (ii = 0; ii + 1 <= u->size[0]; ii++) {
        resid += u->data[ii] * u->data[ii];
      }

      beta = sqrt(resid);
      if (u->data[0] < 0.0) {
        beta = -beta;
      }

      resid = sqrt(2.0 * resid + 2.0 * u->data[0] * beta);
      u->data[0] += beta;
      i0 = u->size[0];
      emxEnsureCapacity((emxArray__common *)u, i0, sizeof(double));
      ii = u->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        u->data[i0] /= resid;
      }

      y->data[0] = -beta;
      ii = u->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        V->data[i0] = u->data[i0];
      }

      j = 1;
      do {
        exitg2 = 0;
        resid = -2.0 * V->data[(j + V->size[0] * (j - 1)) - 1];
        ii = V->size[0];
        i0 = v->size[0];
        v->size[0] = ii;
        emxEnsureCapacity((emxArray__common *)v, i0, sizeof(double));
        for (i0 = 0; i0 < ii; i0++) {
          v->data[i0] = resid * V->data[i0 + V->size[0] * (j - 1)];
        }

        v->data[j - 1]++;
        for (i = j - 2; i + 1 > 0; i--) {
          resid = V->data[i + V->size[0] * i] * v->data[i];
          for (ii = i + 1; ii + 1 <= n; ii++) {
            resid += V->data[ii + V->size[0] * i] * v->data[ii];
          }

          resid *= 2.0;
          for (ii = i; ii + 1 <= n; ii++) {
            v->data[ii] -= resid * V->data[ii + V->size[0] * i];
          }
        }

        resid = 0.0;
        for (ii = 0; ii + 1 <= v->size[0]; ii++) {
          resid += v->data[ii] * v->data[ii];
        }

        resid = sqrt(resid);
        i0 = v->size[0];
        emxEnsureCapacity((emxArray__common *)v, i0, sizeof(double));
        ii = v->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          v->data[i0] /= resid;
        }

        if (need_rowscaling) {
          i0 = v->size[0];
          emxEnsureCapacity((emxArray__common *)v, i0, sizeof(double));
          ii = v->size[0];
          for (i0 = 0; i0 < ii; i0++) {
            v->data[i0] *= rowscal->data[i0];
          }
        }

        DGNLAMGsol_internal(t_prec, t_param, &v->data[0], &dx->data[0],
                            &dbuff->data[0]);
        if (need_colscaling) {
          ii = dx->size[0];
          for (i0 = 0; i0 < ii; i0++) {
            Z->data[i0 + Z->size[0] * (j - 1)] = dx->data[i0] * colscal->data[i0];
          }
        } else {
          ii = dx->size[0];
          for (i0 = 0; i0 < ii; i0++) {
            Z->data[i0 + Z->size[0] * (j - 1)] = dx->data[i0];
          }
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
          resid = V->data[i + V->size[0] * i] * w->data[i];
          for (ii = i + 1; ii + 1 <= n; ii++) {
            resid += V->data[ii + V->size[0] * i] * w->data[ii];
          }

          resid *= 2.0;
          for (ii = i; ii + 1 <= n; ii++) {
            w->data[ii] -= resid * V->data[ii + V->size[0] * i];
          }
        }

        if (j < n) {
          u->data[j - 1] = 0.0;
          u->data[j] = w->data[j];
          resid = w->data[j] * w->data[j];
          for (ii = j + 1; ii + 1 <= n; ii++) {
            u->data[ii] = w->data[ii];
            resid += w->data[ii] * w->data[ii];
          }

          if (resid > 0.0) {
            beta = sqrt(resid);
            if (u->data[j] < 0.0) {
              beta = -beta;
            }

            if (j < restart) {
              resid = sqrt(2.0 * resid + 2.0 * u->data[j] * beta);
              u->data[j] += beta;
              for (ii = j; ii + 1 <= n; ii++) {
                V->data[ii + V->size[0] * j] = u->data[ii] / resid;
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
          resid = w->data[ii];
          w->data[ii] = J->data[J->size[0] * ii] * w->data[ii] + J->data[1 +
            J->size[0] * ii] * w->data[ii + 1];
          w->data[ii + 1] = -J->data[1 + J->size[0] * ii] * resid + J->data
            [J->size[0] * ii] * w->data[ii + 1];
        }

        if (j < n) {
          resid = sqrt(w->data[j - 1] * w->data[j - 1] + w->data[j] * w->data[j]);
          J->data[J->size[0] * (j - 1)] = w->data[j - 1] / resid;
          J->data[1 + J->size[0] * (j - 1)] = w->data[j] / resid;
          y->data[j] = -J->data[1 + J->size[0] * (j - 1)] * y->data[j - 1];
          y->data[j - 1] *= J->data[J->size[0] * (j - 1)];
          w->data[j - 1] = resid;
        }

        for (i0 = 0; i0 < j; i0++) {
          R->data[i0 + R->size[0] * (j - 1)] = w->data[i0];
        }

        resid = fabs(y->data[j]) / beta0;
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
      } while (exitg2 == 0);

      if (verbose == 1) {
        m2c_printf(*iter, resid);
      }

      backsolve(R, y, j);
      for (i = 0; i + 1 <= j; i++) {
        beta = y->data[i];
        i0 = x->size[0];
        emxEnsureCapacity((emxArray__common *)x, i0, sizeof(double));
        ii = x->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          x->data[i0] += beta * Z->data[i0 + Z->size[0] * i];
        }
      }

      if (resid < rtol) {
        exitg1 = true;
      } else {
        it_outer++;
      }
    }

    emxFree_real_T(&b_Z);
    emxFree_int32_T(&r0);
    emxFree_real_T(&v);
    emxFree_real_T(&u);
    emxFree_real_T(&dbuff);
    emxFree_real_T(&w);
    emxFree_real_T(&dx);
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
    *flag = (resid > rtol);
  }
}

void gmresMILU_HO_initialize(void)
{
}

void gmresMILU_HO_terminate(void)
{
}
