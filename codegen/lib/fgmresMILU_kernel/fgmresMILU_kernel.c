#include "fgmresMILU_kernel.h"
#include "m2c.h"
#include "omp.h"
#include "ilupack.h"

static void b_m2c_error(const emxArray_char_T *varargin_3);
static void crs_prodAx(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
  *A_col_ind, const emxArray_real_T *A_val, int A_nrows, const emxArray_real_T
  *x, emxArray_real_T *b);
static void crs_prodAx_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, int x_m, emxArray_real_T *b, int b_m, int nrows, int nrhs, boolean_T ismt);
static void m2c_error(const emxArray_char_T *varargin_3);
static void m2c_printf(int varargin_2, double varargin_3);
static void b_m2c_error(const emxArray_char_T *varargin_3)
{
  emxArray_char_T *b_varargin_3;
  int i2;
  int loop_ub;
  emxInit_char_T(&b_varargin_3, 2);
  i2 = b_varargin_3->size[0] * b_varargin_3->size[1];
  b_varargin_3->size[0] = 1;
  b_varargin_3->size[1] = varargin_3->size[1];
  emxEnsureCapacity((emxArray__common *)b_varargin_3, i2, sizeof(char));
  loop_ub = varargin_3->size[0] * varargin_3->size[1];
  for (i2 = 0; i2 < loop_ub; i2++) {
    b_varargin_3->data[i2] = varargin_3->data[i2];
  }

  M2C_error("m2c_opaque_obj:WrongInput",
            "Incorrect data type %s. Expected DILUPACKparam *.\n",
            &b_varargin_3->data[0]);
  emxFree_char_T(&b_varargin_3);
}

static void crs_prodAx(const emxArray_int32_T *A_row_ptr, const emxArray_int32_T
  *A_col_ind, const emxArray_real_T *A_val, int A_nrows, const emxArray_real_T
  *x, emxArray_real_T *b)
{
  int i3;
  i3 = b->size[0];
  b->size[0] = A_nrows;
  emxEnsureCapacity((emxArray__common *)b, i3, sizeof(double));
  i3 = b->size[0];
  crs_prodAx_kernel(A_row_ptr, A_col_ind, A_val, x, x->size[0], b, i3, A_nrows,
                    1, false);
}

static void crs_prodAx_kernel(const emxArray_int32_T *row_ptr, const
  emxArray_int32_T *col_ind, const emxArray_real_T *val, const emxArray_real_T
  *x, int x_m, emxArray_real_T *b, int b_m, int nrows, int nrhs, boolean_T ismt)
{
  int istart;
  int iend;
  int b_remainder;
  int threadID;
  int chunk;
  int i;
  double t;
  int j;
  if (ismt) {
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

  b_remainder = -1;
  threadID = -1;
  for (chunk = 1; chunk <= nrhs; chunk++) {
    for (i = istart + 1; i <= iend; i++) {
      t = 0.0;
      for (j = row_ptr->data[i - 1]; j < row_ptr->data[i]; j++) {
        t += val->data[j - 1] * x->data[b_remainder + col_ind->data[j - 1]];
      }

      b->data[threadID + i] = t;
    }

    b_remainder += x_m;
    threadID += b_m;
  }
}

static void m2c_error(const emxArray_char_T *varargin_3)
{
  emxArray_char_T *b_varargin_3;
  int i1;
  int loop_ub;
  emxInit_char_T(&b_varargin_3, 2);
  i1 = b_varargin_3->size[0] * b_varargin_3->size[1];
  b_varargin_3->size[0] = 1;
  b_varargin_3->size[1] = varargin_3->size[1];
  emxEnsureCapacity((emxArray__common *)b_varargin_3, i1, sizeof(char));
  loop_ub = varargin_3->size[0] * varargin_3->size[1];
  for (i1 = 0; i1 < loop_ub; i1++) {
    b_varargin_3->data[i1] = varargin_3->data[i1];
  }

  M2C_error("m2c_opaque_obj:WrongInput",
            "Incorrect data type %s. Expected DAMGlevelmat *.\n",
            &b_varargin_3->data[0]);
  emxFree_char_T(&b_varargin_3);
}

static void m2c_printf(int varargin_2, double varargin_3)
{
  M2C_printf("At iteration %d, relative residual is %g.\n", varargin_2,
             varargin_3);
}

void fgmresMILU_kernel(const struct0_T *A, const emxArray_real_T *b, const
  struct1_T *prec, int restart, double rtol, int maxit, const emxArray_real_T
  *x0, int verbose, const struct1_T *param, const emxArray_real_T *rowscal,
  const emxArray_real_T *colscal, emxArray_real_T *x, int *flag, int *iter,
  emxArray_real_T *resids)
{
  int n;
  double resid;
  int ii;
  double beta0;
  int i0;
  int max_outer_iters;
  emxArray_real_T *y;
  emxArray_real_T *R;
  emxArray_real_T *J;
  emxArray_real_T *dx;
  emxArray_real_T *V;
  emxArray_real_T *Z;
  emxArray_real_T *dbuff;
  boolean_T p;
  boolean_T b_p;
  boolean_T exitg1;
  emxArray_char_T *b_prec;
  static const char cv0[14] = { 'D', 'A', 'M', 'G', 'l', 'e', 'v', 'e', 'l', 'm',
    'a', 't', ' ', '*' };

  emxArray_uint8_T *data;
  DAMGlevelmat * t_prec;
  emxArray_char_T *b_param;
  static const char cv1[15] = { 'D', 'I', 'L', 'U', 'P', 'A', 'C', 'K', 'p', 'a',
    'r', 'a', 'm', ' ', '*' };

  DILUPACKparam * t_param;
  int it_outer;
  emxArray_real_T *Ax;
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

    emxInit_real_T(&y, 1);
    i0 = y->size[0];
    y->size[0] = restart + 1;
    emxEnsureCapacity((emxArray__common *)y, i0, sizeof(double));
    for (i0 = 0; i0 <= restart; i0++) {
      y->data[i0] = 0.0;
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

    emxInit_real_T(&V, 2);
    i0 = V->size[0] * V->size[1];
    V->size[0] = b->size[0];
    V->size[1] = restart;
    emxEnsureCapacity((emxArray__common *)V, i0, sizeof(double));
    ii = b->size[0] * restart;
    for (i0 = 0; i0 < ii; i0++) {
      V->data[i0] = 0.0;
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

    *iter = 0;
    resid = 0.0;
    p = false;
    b_p = false;
    if (prec->type->size[1] == 14) {
      b_p = true;
    }

    if (b_p && (!(prec->type->size[1] == 0))) {
      ii = 0;
      exitg1 = false;
      while ((!exitg1) && (ii < 14)) {
        if (!(prec->type->data[ii] == cv0[ii])) {
          b_p = false;
          exitg1 = true;
        } else {
          ii++;
        }
      }
    }

    if (b_p) {
      p = true;
    }

    if (!p) {
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
    p = false;
    b_p = false;
    if (param->type->size[1] == 15) {
      b_p = true;
    }

    if (b_p && (!(param->type->size[1] == 0))) {
      ii = 0;
      exitg1 = false;
      while ((!exitg1) && (ii < 15)) {
        if (!(param->type->data[ii] == cv1[ii])) {
          b_p = false;
          exitg1 = true;
        } else {
          ii++;
        }
      }
    }

    if (b_p) {
      p = true;
    }

    if (!p) {
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

    t_param = *(DILUPACKparam **)(&data->data[0]);
    it_outer = 1;
    emxFree_uint8_T(&data);
    emxInit_real_T(&Ax, 1);
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
          i0 = Ax->size[0];
          Ax->size[0] = b->size[0];
          emxEnsureCapacity((emxArray__common *)Ax, i0, sizeof(double));
          ii = b->size[0];
          for (i0 = 0; i0 < ii; i0++) {
            Ax->data[i0] = b->data[i0];
          }
        }
      }

      if (guard1) {
        crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, x, Ax);
        i0 = Ax->size[0];
        Ax->size[0] = b->size[0];
        emxEnsureCapacity((emxArray__common *)Ax, i0, sizeof(double));
        ii = b->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          Ax->data[i0] = b->data[i0] - Ax->data[i0];
        }
      }

      resid = 0.0;
      for (ii = 0; ii + 1 <= Ax->size[0]; ii++) {
        resid += Ax->data[ii] * Ax->data[ii];
      }

      beta = sqrt(resid);
      if (Ax->data[0] < 0.0) {
        beta = -beta;
      }

      resid = sqrt(2.0 * resid + 2.0 * Ax->data[0] * beta);
      Ax->data[0] += beta;
      i0 = Ax->size[0];
      emxEnsureCapacity((emxArray__common *)Ax, i0, sizeof(double));
      ii = Ax->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        Ax->data[i0] /= resid;
      }

      y->data[0] = -beta;
      ii = Ax->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        V->data[i0] = Ax->data[i0];
      }

      j = 0;
      do {
        exitg2 = 0;
        resid = -2.0 * V->data[j + V->size[0] * j];
        ii = V->size[0];
        i0 = v->size[0];
        v->size[0] = ii;
        emxEnsureCapacity((emxArray__common *)v, i0, sizeof(double));
        for (i0 = 0; i0 < ii; i0++) {
          v->data[i0] = resid * V->data[i0 + V->size[0] * j];
        }

        v->data[j]++;
        for (i = j - 1; i + 1 > 0; i--) {
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

        i0 = v->size[0];
        emxEnsureCapacity((emxArray__common *)v, i0, sizeof(double));
        ii = v->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          v->data[i0] *= rowscal->data[i0];
        }

        DGNLAMGsol_internal(t_prec, t_param, &v->data[0], &dx->data[0],
                            &dbuff->data[0]);
        ii = dx->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          Z->data[i0 + Z->size[0] * j] = dx->data[i0] * colscal->data[i0];
        }

        ii = Z->size[0];
        i0 = b_Z->size[0];
        b_Z->size[0] = ii;
        emxEnsureCapacity((emxArray__common *)b_Z, i0, sizeof(double));
        for (i0 = 0; i0 < ii; i0++) {
          b_Z->data[i0] = Z->data[i0 + Z->size[0] * j];
        }

        crs_prodAx(A->row_ptr, A->col_ind, A->val, A->nrows, b_Z, v);
        for (i = 0; i + 1 <= j + 1; i++) {
          resid = V->data[i + V->size[0] * i] * v->data[i];
          for (ii = i + 1; ii + 1 <= n; ii++) {
            resid += V->data[ii + V->size[0] * i] * v->data[ii];
          }

          resid *= 2.0;
          for (ii = i; ii + 1 <= n; ii++) {
            v->data[ii] -= resid * V->data[ii + V->size[0] * i];
          }
        }

        if (j + 1 < n) {
          Ax->data[j] = 0.0;
          Ax->data[j + 1] = v->data[j + 1];
          resid = v->data[j + 1] * v->data[j + 1];
          for (ii = j + 2; ii + 1 <= n; ii++) {
            Ax->data[ii] = v->data[ii];
            resid += v->data[ii] * v->data[ii];
          }

          if (resid > 0.0) {
            beta = sqrt(resid);
            if (Ax->data[j + 1] < 0.0) {
              beta = -beta;
            }

            if (j + 1 < restart) {
              resid = sqrt(2.0 * resid + 2.0 * Ax->data[j + 1] * beta);
              Ax->data[j + 1] += beta;
              for (ii = j + 1; ii + 1 <= n; ii++) {
                V->data[ii + V->size[0] * (j + 1)] = Ax->data[ii] / resid;
              }
            }

            i0 = v->size[0];
            if (j + 3 > i0) {
              i = 1;
              i0 = 0;
            } else {
              i = j + 3;
            }

            ii = r0->size[0] * r0->size[1];
            r0->size[0] = 1;
            r0->size[1] = (i0 - i) + 1;
            emxEnsureCapacity((emxArray__common *)r0, ii, sizeof(int));
            ii = (i0 - i) + 1;
            for (i0 = 0; i0 < ii; i0++) {
              r0->data[r0->size[0] * i0] = (i + i0) - 1;
            }

            ii = r0->size[0] * r0->size[1];
            for (i0 = 0; i0 < ii; i0++) {
              v->data[r0->data[i0]] = 0.0;
            }

            v->data[j + 1] = -beta;
          }
        }

        for (ii = 0; ii + 1 <= j; ii++) {
          resid = v->data[ii];
          v->data[ii] = J->data[J->size[0] * ii] * v->data[ii] + J->data[1 +
            J->size[0] * ii] * v->data[ii + 1];
          v->data[ii + 1] = -J->data[1 + J->size[0] * ii] * resid + J->data
            [J->size[0] * ii] * v->data[ii + 1];
        }

        if (j + 1 != v->size[0]) {
          resid = sqrt(v->data[j] * v->data[j] + v->data[j + 1] * v->data[j + 1]);
          J->data[J->size[0] * j] = v->data[j] / resid;
          J->data[1 + J->size[0] * j] = v->data[j + 1] / resid;
          y->data[j + 1] = -J->data[1 + J->size[0] * j] * y->data[j];
          y->data[j] *= J->data[J->size[0] * j];
          v->data[j] = resid;
          v->data[j + 1] = 0.0;
        }

        if (1 > restart) {
          ii = -1;
        } else {
          ii = restart - 1;
        }

        for (i0 = 0; i0 <= ii; i0++) {
          R->data[i0 + R->size[0] * j] = v->data[i0];
        }

        resid = fabs(y->data[j + 1]) / beta0;
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
      } while (exitg2 == 0);

      if (verbose > 0) {
        m2c_printf(*iter, resid);
      }

      for (i = j; i + 1 > 0; i--) {
        for (ii = i + 1; ii + 1 <= j + 1; ii++) {
          y->data[i] -= R->data[i + R->size[0] * ii] * y->data[ii];
        }

        y->data[i] /= R->data[i + R->size[0] * i];
      }

      beta = y->data[j];
      ii = Z->size[0];
      i0 = dx->size[0];
      dx->size[0] = ii;
      emxEnsureCapacity((emxArray__common *)dx, i0, sizeof(double));
      for (i0 = 0; i0 < ii; i0++) {
        dx->data[i0] = beta * Z->data[i0 + Z->size[0] * j];
      }

      while (j > 0) {
        beta = y->data[j - 1];
        i0 = dx->size[0];
        emxEnsureCapacity((emxArray__common *)dx, i0, sizeof(double));
        ii = dx->size[0];
        for (i0 = 0; i0 < ii; i0++) {
          dx->data[i0] += beta * Z->data[i0 + Z->size[0] * (j - 1)];
        }

        j--;
      }

      i0 = x->size[0];
      emxEnsureCapacity((emxArray__common *)x, i0, sizeof(double));
      ii = x->size[0];
      for (i0 = 0; i0 < ii; i0++) {
        x->data[i0] += dx->data[i0];
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
    emxFree_real_T(&Ax);
    emxFree_real_T(&dbuff);
    emxFree_real_T(&Z);
    emxFree_real_T(&V);
    emxFree_real_T(&dx);
    emxFree_real_T(&J);
    emxFree_real_T(&R);
    emxFree_real_T(&y);
    i0 = resids->size[0];
    if (1 > *iter) {
      resids->size[0] = 0;
    } else {
      resids->size[0] = *iter;
    }

    emxEnsureCapacity((emxArray__common *)resids, i0, sizeof(double));
    if (resid > rtol) {
      *flag = 3;
    } else {
      *flag = 0;
    }
  }
}

void fgmresMILU_kernel_initialize(void)
{
}

void fgmresMILU_kernel_terminate(void)
{
}
