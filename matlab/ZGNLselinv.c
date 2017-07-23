/* $Id: ZGNLselinv.c 804 2015-10-22 19:24:41Z bolle $ */
/* ========================================================================== */
/* === ZGNLselinv mexFunction =============================================== */
/* ========================================================================== */

/*
    Usage:

    Return general sparse selective inverse based on LU decomposition

    Example:

    % for initializing parameters
    Ainv=ZGNLselinv(L,D,UT,p, Deltal, Deltar)


    Authors:

        Matthias Bollhoefer, TU Braunschweig

    Date:

        October 21, 2015. ILUPACK V2.5.

    Notice:

        Copyright (c) 2015 by TU Braunschweig.  All Rights Reserved.

        THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
        EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.

    Availability:

        This file is located at

        http://ilupack.tu-bs.de/
*/

/* ========================================================================== */
/* === Include files and prototypes ========================================= */
/* ========================================================================== */

#include "matrix.h"
#include "mex.h"
#include <ilupack.h>
#include <ilupackmacros.h>
#include <lapack.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FIELDS 100
#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define MIN(A, B) (((A) >= (B)) ? (B) : (A))
#define ELBOW MAX(4.0, 2.0)
/* #define PRINT_CHECK */
/* #define PRINT_INFO  */

/* ========================================================================== */
/* === mexFunction ========================================================== */
/* ========================================================================== */

void mexFunction(
    /* === Parameters ======================================================= */

    int nlhs,             /* number of left-hand sides */
    mxArray *plhs[],      /* left-hand side matrices */
    int nrhs,             /* number of right--hand sides */
    const mxArray *prhs[] /* right-hand side matrices */
    ) {
  mxArray *L_input, *D_input, *UT_input, *p_input, *Deltal_input, *Deltar_input;
  integer i, j, k, l, m, kk, ll, mm, n, flagkm1, flag, flagkp1, nnz, *p, *invp,
      ierr, *ibuff, flag_piL, flag_piD, flag_piUT;
  doublecomplex Djm1jm1, Djjm1, Djm1j, Djj, det, *Linv_a, *Dinv_a, *UTinv_a;
  size_t mrows, ncols;
  double *L_valuesR, *D_valuesR, *UT_valuesR, *Ainv_valuesR, *L_valuesI,
      *D_valuesI, *UT_valuesI, *Ainv_valuesI, *p_valuesR, *Deltal_valuesR,
      *Deltar_valuesR, nrm;
  mwIndex *Linv_ja, *Dinv_ja, *UTinv_ja,
      *Ainv_ja, /* row indices of output matrix
                   Linv,Dinv,UTinv and finally Ainv */
      *Linv_ia, *Dinv_ia, *UTinv_ia,
      *Ainv_ia,      /* column pointers of output matrix
                        Linv,Dinv,UTinv and finally Ainv */
      *L_ja, *UT_ja, /* row indices of input matrix L,U^T     */
      *L_ia, *UT_ia, /* column pointers of input matrix L,U^T */
      *D_ja,         /* row indices of input matrix D         */
      *D_ia;         /* column pointers of input matrix D     */

  if (nrhs != 6)
    mexErrMsgTxt("Six input arguments required.");
  else if (nlhs != 1)
    mexErrMsgTxt("wrong number of output arguments.");
  else if (!mxIsNumeric(prhs[0]))
    mexErrMsgTxt("First input must be a matrix.");
  else if (!mxIsNumeric(prhs[1]))
    mexErrMsgTxt("Second input must be a matrix.");
  else if (!mxIsNumeric(prhs[2]))
    mexErrMsgTxt("Third input must be a matrix.");
  else if (!mxIsNumeric(prhs[3]))
    mexErrMsgTxt("Fourth input must be a matrix.");
  else if (!mxIsNumeric(prhs[4]))
    mexErrMsgTxt("Fifth input must be a matrix.");
  else if (!mxIsNumeric(prhs[5]))
    mexErrMsgTxt("Sixth input must be a matrix.");

  /* The first input must be a square matrix.*/
  L_input = (mxArray *)prhs[0];
  /* get size of input matrix L */
  mrows = mxGetM(L_input);
  ncols = mxGetN(L_input);
  nnz = mxGetNzmax(L_input);
  if (mrows != ncols) {
    mexErrMsgTxt("First input must be a square matrix.");
  }
  if (!mxIsSparse(L_input)) {
    mexErrMsgTxt("First input matrix must be in sparse format.");
  }
  n = mrows;
  L_ja = (mwIndex *)mxGetIr(L_input);
  L_ia = (mwIndex *)mxGetJc(L_input);
  L_valuesR = (double *)mxGetPr(L_input);
  L_valuesI = (double *)mxGetPi(L_input);
  flag_piL = (L_valuesI != NULL);
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: input parameter L imported\n");
  fflush(stdout);
#endif

  /* The second input must be a square matrix.*/
  D_input = (mxArray *)prhs[1];
  /* get size of input matrix D */
  mrows = mxGetM(D_input);
  ncols = mxGetN(D_input);
  if (mrows != ncols || mrows != n) {
    mexErrMsgTxt("Second input must be a square matrix of same size as the "
                 "first matrix.");
  }
  if (!mxIsSparse(D_input)) {
    mexErrMsgTxt("Second input matrix must be in sparse format.");
  }
  D_ja = (mwIndex *)mxGetIr(D_input);
  D_ia = (mwIndex *)mxGetJc(D_input);
  D_valuesR = (double *)mxGetPr(D_input);
  D_valuesI = (double *)mxGetPi(D_input);
  flag_piD = (D_valuesI != NULL);
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: input parameter D imported\n");
  fflush(stdout);
#endif

  /* The third input must be a square matrix.*/
  UT_input = (mxArray *)prhs[2];
  /* get size of input matrix U^T */
  mrows = mxGetM(UT_input);
  ncols = mxGetN(UT_input);
  nnz = mxGetNzmax(UT_input);
  if (mrows != ncols || mrows != n) {
    mexErrMsgTxt("Third input must be a square matrix of same size as the "
                 "first matrix.");
  }
  if (!mxIsSparse(UT_input)) {
    mexErrMsgTxt("Third input matrix must be in sparse format.");
  }
  n = mrows;
  UT_ja = (mwIndex *)mxGetIr(UT_input);
  UT_ia = (mwIndex *)mxGetJc(UT_input);
  UT_valuesR = (double *)mxGetPr(UT_input);
  UT_valuesI = (double *)mxGetPi(UT_input);
  flag_piUT = (UT_valuesI != NULL);
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: input parameter UT imported\n");
  fflush(stdout);
#endif

  /* The fourth input must be a integer vector */
  p_input = (mxArray *)prhs[3];
  /* get size of input matrix D */
  mrows = mxGetM(p_input);
  ncols = mxGetN(p_input);
  if (n != ncols && mrows != n) {
    mexErrMsgTxt(
        "Fourth argument must be a vector of same size as the first matrix.");
  }
  if (mxIsSparse(p_input)) {
    mexErrMsgTxt("Fourth input vector must be in dense format.");
  }
  p_valuesR = (double *)mxGetPr(p_input);

  /* memory for scaling and permutation */
  p = (integer *)MAlloc((size_t)n * sizeof(integer), "ZGNLselinv");
  invp = (integer *)MAlloc((size_t)n * sizeof(integer), "ZGNLselinv");
  ibuff = (integer *)MAlloc((size_t)n * sizeof(integer), "ZGNLselinv");
  for (i = 0; i < n; i++) {
    j = p_valuesR[i] - 1;
    if (j < 0 || n <= j) {
      mexErrMsgTxt(
          "permutation vector must have integer values within 1,...,n");
    }
    p[i] = j;
    invp[j] = i;
    ibuff[i] = 0;
  } /* end for i */
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: input parameter p imported\n");
  fflush(stdout);
#endif

  /* The fifth input must be a  vector */
  Deltal_input = (mxArray *)prhs[4];
  /* get size of input matrix Delta */
  mrows = mxGetM(Deltal_input);
  ncols = mxGetN(Deltal_input);
  if (n != ncols && mrows != n) {
    mexErrMsgTxt(
        "Fourth argument must be a vector of same size as the first matrix.");
  }
  if (mxIsSparse(Deltal_input)) {
    mexErrMsgTxt("Fourth input vector must be in dense format.");
  }
  Deltal_valuesR = (double *)mxGetPr(Deltal_input);
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: input parameter Deltal imported\n");
  fflush(stdout);
#endif

  /* The sixth input must be a  vector */
  Deltar_input = (mxArray *)prhs[5];
  /* get size of input matrix Delta */
  mrows = mxGetM(Deltar_input);
  ncols = mxGetN(Deltar_input);
  if (n != ncols && mrows != n) {
    mexErrMsgTxt(
        "Sixth argument must be a vector of same size as the first matrix.");
  }
  if (mxIsSparse(Deltar_input)) {
    mexErrMsgTxt("Fourth input vector must be in dense format.");
  }
  Deltar_valuesR = (double *)mxGetPr(Deltar_input);
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: input parameter Deltar imported\n");
  fflush(stdout);
#endif

#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: input parameters imported\n");
  fflush(stdout);
#endif

  /* set up preliminary structure for every column of Linv,Dinv,UTinv */
  /* create memory for output matrix Linv,Dinv,UTinv, here: Linv <-L */
  Linv_ia = (integer *)MAlloc((size_t)(n + 1) * sizeof(integer), "ZGNLselinv");
  i = 0;
  Linv_ia[0] = 0;
  /* count additional fill-in for block columns of size 2 */
  nnz = 0;
  while (i < n) {
/* mexPrintf("i=%d\n",i+1);fflush(stdout); */
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (ibuff[m] != 0)
        mexPrintf("ibuff dirty at position %d\n", m + 1);
      fflush(stdout);
    }
#endif

    /* 1x1 case */
    if (D_ia[i + 1] - D_ia[i] == 1) {
/* in this case the first entry must be the diagonal entry */
#ifdef PRINT_CHECK
      if (D_ja[D_ia[i]] != i) {
        mexPrintf("D: diagonal entry missmatch at entry %d\n", i + 1);
        fflush(stdout);
      }
#endif
      kk = Linv_ia[i];
/* skip diagonal entry and copy sub-diagonal indices of L(:,i) */
#ifdef PRINT_CHECK
      if (L_ja[L_ia[i]] != i) {
        mexPrintf("L: diagonal entry missmatch at entry %d\n", i + 1);
        fflush(stdout);
      }
#endif
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
#ifdef PRINT_CHECK
        if (k <= i && m > L_ia[i]) {
          mexPrintf("L: sub-diagonal entry missmatch at entry %d\n", i + 1);
          fflush(stdout);
        }
#endif
        if (k > i)
          kk++;
      } /* end for i */
      Linv_ia[i + 1] = kk;

      i++;
    } else { /* 2x2 case */
#ifdef PRINT_CHECK
      if (D_ja[D_ia[i]] != i) {
        mexPrintf("D: diagonal entry missmatch at entry       (%d,%d)\n", i + 1,
                  i + 1);
        fflush(stdout);
      }
      if (D_ja[D_ia[i] + 1] != i + 1) {
        mexPrintf("D: sub-diagonal entry missmatch at entry   (%d,%d)\n", i + 2,
                  i + 1);
        fflush(stdout);
      }
      if (D_ja[D_ia[i + 1]] != i) {
        mexPrintf("D: super-diagonal entry missmatch at entry (%d,%d)\n", i + 2,
                  i + 2);
        fflush(stdout);
      }
      if (D_ja[D_ia[i + 1] + 1] != i + 1) {
        mexPrintf("D: diagonal entry missmatch at entry       (%d,%d)\n", i + 2,
                  i + 2);
        fflush(stdout);
      }
#endif

      kk = Linv_ia[i];
#ifdef PRINT_CHECK
      if (L_ja[L_ia[i]] != i) {
        mexPrintf("L: diagonal entry missmatch at entry %d\n", i + 1);
        fflush(stdout);
      }
#endif
      /* skip diagonal entry, skip sub-diagonal entry and copy sub-sub-diagonal
       * indices of L(:,i) */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
#ifdef PRINT_CHECK
        if (k <= i && m > L_ia[i]) {
          mexPrintf("L: sub-sub-diagonal entry missmatch at entry %d\n", i + 1);
          fflush(stdout);
        }
#endif
#ifdef PRINT_CHECK
        if (k == i + 1) {
          mexPrintf("L: fill-in ignored at entry (%d,%d)\n", k + 1, i + 1);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          kk++;
          ibuff[k] = 1;
        }
      } /* end for i */
      Linv_ia[i + 1] = kk;

#ifdef PRINT_CHECK
      if (L_ja[L_ia[i + 1]] != i + 1) {
        mexPrintf("L: diagonal entry missmatch at entry %d\n", i + 2);
        fflush(stdout);
      }
#endif
      /* skip diagonal entry and copy sub-diagonal indices of L(:,i+1) */
      for (m = L_ia[i + 1]; m < L_ia[i + 2]; m++) {
        k = L_ja[m];
#ifdef PRINT_CHECK
        if (k <= i && m > L_ia[i + 1]) {
          mexPrintf("L: sub-sub-diagonal entry missmatch at entry %d\n", i + 2);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          kk++;
          /* do we visit index k for the first time ? */
          if (!ibuff[k]) {
            /* column i will obtain another fill-in */
            nnz++;
          }
          /* use negative check mark to distinguish between those
             indices that have only been visited by column i and those ones
             that are visited by column i+1 */
          ibuff[k] = -1;
        }
      } /* end for i */
      Linv_ia[i + 2] = kk;

      /* clear ibuff, column i */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
#ifdef PRINT_CHECK
        if (k == i + 1) {
          mexPrintf("L: fill-in ignored at entry (%d,%d)\n", k + 1, i + 1);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          /* did only column i visit this index? */
          if (ibuff[k] > 0)
            /* column i+1 will have another fill-in */
            nnz++;
          /* clear buff */
          ibuff[k] = 0;
        }
      } /* end for i */
      /* clear ibuff, column i+1 */
      for (m = L_ia[i + 1]; m < L_ia[i + 2]; m++) {
        k = L_ja[m];
        if (k > i + 1)
          ibuff[k] = 0;
      } /* end for i */

      i += 2;
    }
  } /* end while i */
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: additional memory for blocked columns: %d\n", nnz);
  fflush(stdout);
#endif
  nnz += Linv_ia[n];

  Linv_ja =
      (integer *)MAlloc((size_t)nnz * sizeof(integer), "ZGNLselinv:Linv_ja");
  Linv_a = (doublecomplex *)MAlloc((size_t)nnz * sizeof(doublecomplex),
                                   "ZGNLselinv:Linv_a");
  for (i = 0; i < nnz; i++) {
    Linv_a[i].r = Linv_a[i].i = 0.0;
  } /* end for i */
  /* set up completed structure for every column of Linv_a */
  i = 0; /* column counter */
  Linv_ia[0] = 0;
  /* now insert additional fill-in for block columns of column size 2 */
  while (i < n) {
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (ibuff[m] != 0)
        mexPrintf("ibuff dirty at position %d\n", m + 1);
      fflush(stdout);
    }
#endif
    /* 1x1 case */
    if (D_ia[i + 1] - D_ia[i] == 1) {
      kk = Linv_ia[i];
      /* skip diagonal entry and copy sub-diagonal indices of L(:,i) */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
        if (k > i)
          Linv_ja[kk++] = k;
      } /* end for i */
      Linv_ia[i + 1] = kk;

      i++;
    } else { /* 2x2 case */
      kk = Linv_ia[i];
      /* skip diagonal entry, skip sub-diagonal entry and copy sub-sub-diagonal
       * indices of L(:,i) */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
#ifdef PRINT_CHECK
        if (k == i + 1) {
          mexPrintf("Linv: fill-in ignored at entry (%d,%d)\n", k + 1, i + 1);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          Linv_ja[kk++] = k;
          /* flag L(k,i) */
          ibuff[k] = 1;
        }
      } /* end for i */
      flag = 0;
      /* check sub-diagonal indices of L(:,i+1) for column i
         and insert additional fill-in for L(:,i)
      */
      for (m = L_ia[i + 1]; m < L_ia[i + 2]; m++) {
        k = L_ja[m];
        if (k > i + 1) {
          /* do we visit index k for the first time ? */
          if (!ibuff[k]) {
            /* column i will obtain another fill-in */
            Linv_ja[kk++] = k;
            flag |= 1;
          }
          /* use negative check mark to distinguish between those
             indices that have only been visited by column i and those ones
             that are visited by column i+1 */
          ibuff[k] = -1;
        }
      } /* end for i */
      Linv_ia[i + 1] = kk;

      /* skip diagonal entry column i+1 and copy sub-diagonal indices of
       * L(:,i+1) */
      for (m = L_ia[i + 1]; m < L_ia[i + 2]; m++) {
        k = L_ja[m];
        if (k > i + 1) {
          Linv_ja[kk++] = k;
        }
      } /* end for i */
      /* clear ibuff w.r.t. column i and add fill-in to column i+1 */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
#ifdef PRINT_CHECK
        if (k == i + 1) {
          mexPrintf("Linv: fill-in ignored at entry (%d,%d)\n", k + 1, i + 1);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          /* did only column i visit this index? */
          if (ibuff[k] > 0) {
            /* column i+1 will have another fill-in */
            Linv_ja[kk++] = k;
            flag |= 2;
          }
          /* clear buff */
          ibuff[k] = 0;
        }
      } /* end for i */
      Linv_ia[i + 2] = kk;

      /* clear ibuff, column i+1 */
      for (m = L_ia[i + 1]; m < L_ia[i + 2]; m++) {
        k = L_ja[m];
        if (k > i + 1)
          ibuff[k] = 0;
      } /* end for i */

      /* sort column i */
      m = Linv_ia[i];
      l = Linv_ia[i + 1] - m;
      if (flag & 1) {
        qqsorti(Linv_ja + m, ibuff, &l);
        while (l)
          ibuff[--l] = 0;
      }
      /* sort column i */
      m = Linv_ia[i + 1];
      l = Linv_ia[i + 2] - m;
      if (flag & 2) {
        qqsorti(Linv_ja + m, ibuff, &l);
        while (l)
          ibuff[--l] = 0;
      }
      flag = 0;

      i += 2;
    }
  } /* end while i */
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: row index structure inserted\n");
  fflush(stdout);
  for (i = 0; i < n; i++) {
    mexPrintf("row indices column %d\n", i + 1);
    for (j = Linv_ia[i]; j < Linv_ia[i + 1]; j++) {
      mexPrintf("%8ld", Linv_ja[j] + 1);
    }
    mexPrintf("\n");
  }
#endif

  /* set up preliminary structure for every column of Linv,Dinv,UTinv */
  /* create memory for output matrix Linv,Dinv,UTinv, here: UTinv <-UT */
  UTinv_ia = (integer *)MAlloc((size_t)(n + 1) * sizeof(integer), "ZGNLselinv");
  i = 0;
  UTinv_ia[0] = 0;
  /* count additional fill-in for block columns of size 2 */
  nnz = 0;
  while (i < n) {
/* mexPrintf("i=%d\n",i+1);fflush(stdout); */
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (ibuff[m] != 0)
        mexPrintf("ibuff dirty at position %d\n", m + 1);
      fflush(stdout);
    }
#endif

    /* 1x1 case */
    if (D_ia[i + 1] - D_ia[i] == 1) {
/* in this case the first entry must be the diagonal entry */
#ifdef PRINT_CHECK
      if (D_ja[D_ia[i]] != i) {
        mexPrintf("D: diagonal entry missmatch at entry %d\n", i + 1);
        fflush(stdout);
      }
#endif
      kk = UTinv_ia[i];
/* skip diagonal entry and copy sub-diagonal indices of UT(:,i) */
#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[i]] != i) {
        mexPrintf("UT: diagonal entry missmatch at entry %d\n", i + 1);
        fflush(stdout);
      }
#endif
      for (m = UT_ia[i]; m < UT_ia[i + 1]; m++) {
        k = UT_ja[m];
#ifdef PRINT_CHECK
        if (k <= i && m > UT_ia[i]) {
          mexPrintf("UT: sub-diagonal entry missmatch at entry %d\n", i + 1);
          fflush(stdout);
        }
#endif
        if (k > i)
          kk++;
      } /* end for i */
      UTinv_ia[i + 1] = kk;

      i++;
    } else { /* 2x2 case */
#ifdef PRINT_CHECK
      if (D_ja[D_ia[i]] != i) {
        mexPrintf("D: diagonal entry missmatch at entry       (%d,%d)\n", i + 1,
                  i + 1);
        fflush(stdout);
      }
      if (D_ja[D_ia[i] + 1] != i + 1) {
        mexPrintf("D: sub-diagonal entry missmatch at entry   (%d,%d)\n", i + 2,
                  i + 1);
        fflush(stdout);
      }
      if (D_ja[D_ia[i + 1]] != i) {
        mexPrintf("D: super-diagonal entry missmatch at entry (%d,%d)\n", i + 2,
                  i + 2);
        fflush(stdout);
      }
      if (D_ja[D_ia[i + 1] + 1] != i + 1) {
        mexPrintf("D: diagonal entry missmatch at entry       (%d,%d)\n", i + 2,
                  i + 2);
        fflush(stdout);
      }
#endif

      kk = UTinv_ia[i];
#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[i]] != i) {
        mexPrintf("UT: diagonal entry missmatch at entry %d\n", i + 1);
        fflush(stdout);
      }
#endif
      /* skip diagonal entry, skip sub-diagonal entry and copy sub-sub-diagonal
       * indices of UT(:,i) */
      for (m = UT_ia[i]; m < UT_ia[i + 1]; m++) {
        k = UT_ja[m];
#ifdef PRINT_CHECK
        if (k <= i && m > UT_ia[i]) {
          mexPrintf("UT: sub-sub-diagonal entry missmatch at entry %d\n",
                    i + 1);
          fflush(stdout);
        }
#endif
#ifdef PRINT_CHECK
        if (k == i + 1) {
          mexPrintf("UT: fill-in ignored at entry (%d,%d)\n", k + 1, i + 1);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          kk++;
          ibuff[k] = 1;
        }
      } /* end for i */
      UTinv_ia[i + 1] = kk;

#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[i + 1]] != i + 1) {
        mexPrintf("UT: diagonal entry missmatch at entry %d\n", i + 2);
        fflush(stdout);
      }
#endif
      /* skip diagonal entry and copy sub-diagonal indices of UT(:,i+1) */
      for (m = UT_ia[i + 1]; m < UT_ia[i + 2]; m++) {
        k = UT_ja[m];
#ifdef PRINT_CHECK
        if (k <= i && m > UT_ia[i + 1]) {
          mexPrintf("UT: sub-sub-diagonal entry missmatch at entry %d\n",
                    i + 2);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          kk++;
          /* do we visit index k for the first time ? */
          if (!ibuff[k]) {
            /* column i will obtain another fill-in */
            nnz++;
          }
          /* use negative check mark to distinguish between those
             indices that have only been visited by column i and those ones
             that are visited by column i+1 */
          ibuff[k] = -1;
        }
      } /* end for i */
      UTinv_ia[i + 2] = kk;

      /* clear ibuff, column i */
      for (m = UT_ia[i]; m < UT_ia[i + 1]; m++) {
        k = UT_ja[m];
#ifdef PRINT_CHECK
        if (k == i + 1) {
          mexPrintf("UT: fill-in ignored at entry (%d,%d)\n", k + 1, i + 1);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          /* did only column i visit this index? */
          if (ibuff[k] > 0)
            /* column i+1 will have another fill-in */
            nnz++;
          /* clear buff */
          ibuff[k] = 0;
        }
      } /* end for i */
      /* clear ibuff, column i+1 */
      for (m = UT_ia[i + 1]; m < UT_ia[i + 2]; m++) {
        k = UT_ja[m];
        if (k > i + 1)
          ibuff[k] = 0;
      } /* end for i */

      i += 2;
    }
  } /* end while i */
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: additional memory for blocked columns: %d\n", nnz);
  fflush(stdout);
#endif
  nnz += UTinv_ia[n];

  UTinv_ja =
      (integer *)MAlloc((size_t)nnz * sizeof(integer), "ZGNLselinv:UTinv_ja");
  UTinv_a = (doublecomplex *)MAlloc((size_t)nnz * sizeof(doublecomplex),
                                    "ZGNLselinv:UTinv_a");
  for (i = 0; i < nnz; i++) {
    UTinv_a[i].r = UTinv_a[i].i = 0.0;
  } /* end for i */
  /* set up completed structure for every column of UTinv_a */
  i = 0; /* column counter */
  UTinv_ia[0] = 0;
  /* now insert additional fill-in for block columns of column size 2 */
  while (i < n) {
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (ibuff[m] != 0)
        mexPrintf("ibuff dirty at position %d\n", m + 1);
      fflush(stdout);
    }
#endif
    /* 1x1 case */
    if (D_ia[i + 1] - D_ia[i] == 1) {
      kk = UTinv_ia[i];
      /* skip diagonal entry and copy sub-diagonal indices of UT(:,i) */
      for (m = UT_ia[i]; m < UT_ia[i + 1]; m++) {
        k = UT_ja[m];
        if (k > i)
          UTinv_ja[kk++] = k;
      } /* end for i */
      UTinv_ia[i + 1] = kk;

      i++;
    } else { /* 2x2 case */
      kk = UTinv_ia[i];
      /* skip diagonal entry, skip sub-diagonal entry and copy sub-sub-diagonal
       * indices of UT(:,i) */
      for (m = UT_ia[i]; m < UT_ia[i + 1]; m++) {
        k = UT_ja[m];
#ifdef PRINT_CHECK
        if (k == i + 1) {
          mexPrintf("UTinv: fill-in ignored at entry (%d,%d)\n", k + 1, i + 1);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          UTinv_ja[kk++] = k;
          /* flag UT(k,i) */
          ibuff[k] = 1;
        }
      } /* end for i */
      flag = 0;
      /* check sub-diagonal indices of UT(:,i+1) for column i
         and insert additional fill-in for UT(:,i)
      */
      for (m = UT_ia[i + 1]; m < UT_ia[i + 2]; m++) {
        k = UT_ja[m];
        if (k > i + 1) {
          /* do we visit index k for the first time ? */
          if (!ibuff[k]) {
            /* column i will obtain another fill-in */
            UTinv_ja[kk++] = k;
            flag |= 1;
          }
          /* use negative check mark to distinguish between those
             indices that have only been visited by column i and those ones
             that are visited by column i+1 */
          ibuff[k] = -1;
        }
      } /* end for i */
      UTinv_ia[i + 1] = kk;

      /* skip diagonal entry column i+1 and copy sub-diagonal indices of
       * UT(:,i+1) */
      for (m = UT_ia[i + 1]; m < UT_ia[i + 2]; m++) {
        k = UT_ja[m];
        if (k > i + 1) {
          UTinv_ja[kk++] = k;
        }
      } /* end for i */
      /* clear ibuff w.r.t. column i and add fill-in to column i+1 */
      for (m = UT_ia[i]; m < UT_ia[i + 1]; m++) {
        k = UT_ja[m];
#ifdef PRINT_CHECK
        if (k == i + 1) {
          mexPrintf("UTinv: fill-in ignored at entry (%d,%d)\n", k + 1, i + 1);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          /* did only column i visit this index? */
          if (ibuff[k] > 0) {
            /* column i+1 will have another fill-in */
            UTinv_ja[kk++] = k;
            flag |= 2;
          }
          /* clear buff */
          ibuff[k] = 0;
        }
      } /* end for i */
      UTinv_ia[i + 2] = kk;

      /* clear ibuff, column i+1 */
      for (m = UT_ia[i + 1]; m < UT_ia[i + 2]; m++) {
        k = UT_ja[m];
        if (k > i + 1)
          ibuff[k] = 0;
      } /* end for i */

      /* sort column i */
      m = UTinv_ia[i];
      l = UTinv_ia[i + 1] - m;
      if (flag & 1) {
        qqsorti(UTinv_ja + m, ibuff, &l);
        while (l)
          ibuff[--l] = 0;
      }
      /* sort column i */
      m = UTinv_ia[i + 1];
      l = UTinv_ia[i + 2] - m;
      if (flag & 2) {
        qqsorti(UTinv_ja + m, ibuff, &l);
        while (l)
          ibuff[--l] = 0;
      }
      flag = 0;

      i += 2;
    }
  } /* end while i */
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: row index structure inserted\n");
  fflush(stdout);
  for (i = 0; i < n; i++) {
    mexPrintf("row indices column %d\n", i + 1);
    for (j = UTinv_ia[i]; j < UTinv_ia[i + 1]; j++) {
      mexPrintf("%8ld", UTinv_ja[j] + 1);
    }
    mexPrintf("\n");
  }
#endif

  /* set up preliminary structure for every column of Linv,Dinv,UTinv */
  /* create memory for output matrix Linv,Dinv,UTinv, here: Dinv <-D */
  Dinv_ia = (integer *)MAlloc((size_t)(n + 1) * sizeof(integer),
                              "ZGNLselinv:Dinv_ia");
  i = 0;
  Dinv_ia[0] = 0;
  /* count additional fill-in for block columns of size 2 */
  nnz = 0;
  while (i < n) {
/* mexPrintf("i=%d\n",i+1);fflush(stdout); */
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (ibuff[m] != 0)
        mexPrintf("ibuff dirty at position %d\n", m + 1);
      fflush(stdout);
    }
#endif

    /* 1x1 case */
    if (D_ia[i + 1] - D_ia[i] == 1) {
/* in this case the first entry must be the diagonal entry */
#ifdef PRINT_CHECK
      if (D_ja[D_ia[i]] != i) {
        mexPrintf("D: diagonal entry missmatch at entry %d\n", i + 1);
        fflush(stdout);
      }
#endif
      Dinv_ia[i + 1] = Dinv_ia[i] + 1;

      i++;
    } else { /* 2x2 case */
#ifdef PRINT_CHECK
      if (D_ja[D_ia[i]] != i) {
        mexPrintf("D: diagonal entry missmatch at entry       (%d,%d)\n", i + 1,
                  i + 1);
        fflush(stdout);
      }
      if (D_ja[D_ia[i] + 1] != i + 1) {
        mexPrintf("D: sub-diagonal entry missmatch at entry   (%d,%d)\n", i + 2,
                  i + 1);
        fflush(stdout);
      }
      if (D_ja[D_ia[i + 1]] != i) {
        mexPrintf("D: super-diagonal entry missmatch at entry (%d,%d)\n", i + 2,
                  i + 2);
        fflush(stdout);
      }
      if (D_ja[D_ia[i + 1] + 1] != i + 1) {
        mexPrintf("D: diagonal entry missmatch at entry       (%d,%d)\n", i + 2,
                  i + 2);
        fflush(stdout);
      }
#endif

      kk = Dinv_ia[i];
      kk += 2;
      Dinv_ia[i + 1] = kk;
      kk += 2;
      Dinv_ia[i + 2] = kk;

      i += 2;
    }
  } /* end while i */
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: additional memory for blocked columns: %d\n", nnz);
  fflush(stdout);
#endif
  nnz += Dinv_ia[n];

  Dinv_ja =
      (integer *)MAlloc((size_t)nnz * sizeof(integer), "ZGNLselinv:Dinv_ja");
  Dinv_a = (doublecomplex *)MAlloc((size_t)nnz * sizeof(doublecomplex),
                                   "ZGNLselinv:Dinv_a");
  for (i = 0; i < nnz; i++) {
    Dinv_a[i].r = Dinv_a[i].i = 0.0;
  } /* end for i */
  /* set up completed structure for every column of Dinv_a */
  i = 0; /* column counter */
  Dinv_ia[0] = 0;
  /* now insert additional fill-in for block columns of column size 2 */
  while (i < n) {
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (ibuff[m] != 0)
        mexPrintf("ibuff dirty at position %d\n", m + 1);
      fflush(stdout);
    }
#endif
    /* 1x1 case */
    if (D_ia[i + 1] - D_ia[i] == 1) {
      kk = Dinv_ia[i];
      Dinv_ja[kk++] = i;
      Dinv_ia[i + 1] = kk;

      i++;
    } else { /* 2x2 case */
      kk = Dinv_ia[i];
      Dinv_ja[kk++] = i;
      Dinv_ja[kk++] = i + 1;
      Dinv_ia[i + 1] = kk;
      Dinv_ja[kk++] = i;
      Dinv_ja[kk++] = i + 1;
      Dinv_ia[i + 2] = kk;

      i += 2;
    }
  } /* end while i */
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: row index structure inserted\n");
  fflush(stdout);
  for (i = 0; i < n; i++) {
    mexPrintf("row indices column %d\n", i + 1);
    for (j = Dinv_ia[i]; j < Dinv_ia[i + 1]; j++) {
      mexPrintf("%8ld", Dinv_ja[j] + 1);
    }
    mexPrintf("\n");
  }
#endif

  /* start from the right corner */
  j = n - 1;

  /* 1x1 or 2x2 case ? */
  if (j == 0)
    flag = 0;
  else if (D_ia[j + 1] - D_ia[j] == 1)
    flag = 0;
  else /* 2x2 case */
    flag = -1;

  if (!flag) {
    /* starting position in column j */
    /* mexPrintf("j=%d\n",j+1);fflush(stdout); */
    m = D_ia[j];
#ifdef PRINT_CHECK
    if (D_ja[m] != j) {
      mexPrintf("D: diagonal mismatch (%d,%d)\n", D_ja[m] + 1, D_ja[m] + 1);
      fflush(stdout);
    }
#endif
    mm = Dinv_ia[j];
    nrm =
        (flag_piD)
            ? 1.0 / (D_valuesR[m] * D_valuesR[m] + D_valuesI[m] * D_valuesI[m])
            : 1.0 / (D_valuesR[m] * D_valuesR[m]);

    Dinv_a[mm].r = D_valuesR[m] * nrm;
    Dinv_a[mm].i = (flag_piD) ? -D_valuesI[m] * nrm : 0.0;

    j--;
  } else {
    /* columns j-1,j */
    /* extract D(j-1:j,j-1:j) */
    m = D_ia[j - 1];
#ifdef PRINT_CHECK
    if (D_ja[m] != j - 1) {
      mexPrintf("D: diagonal mismatch (%d,%d)\n", D_ja[m] + 1, j);
      fflush(stdout);
    }
#endif
#ifdef PRINT_CHECK
    if (D_ja[m + 1] != j) {
      mexPrintf("D: sub-diagonal mismatch (%d,%d)\n", D_ja[m + 1] + 1, j + 1);
      fflush(stdout);
    }
#endif
    Djm1jm1.r = D_valuesR[m];
    Djm1jm1.i = (flag_piD) ? D_valuesI[m] : 0.0;
    Djjm1.r = D_valuesR[m + 1];
    Djjm1.i = (flag_piD) ? D_valuesI[m + 1] : 0.0;

    m = D_ia[j];
#ifdef PRINT_CHECK
    if (D_ja[m] != j - 1) {
      mexPrintf("D: super-diagonal mismatch (%d,%d)\n", D_ja[m] + 1, j);
      fflush(stdout);
    }
#endif
#ifdef PRINT_CHECK
    if (D_ja[m + 1] != j) {
      mexPrintf("D: diagonal mismatch (%d,%d)\n", D_ja[m + 1] + 1, j + 1);
      fflush(stdout);
    }
#endif
    Djm1j.r = D_valuesR[m];
    Djm1j.i = (flag_piD) ? D_valuesI[m] : 0.0;
    Djj.r = D_valuesR[m + 1];
    Djj.i = (flag_piD) ? D_valuesI[m + 1] : 0.0;
    /* determinant for 2x2 matrix inverse */
    det.r = Djm1jm1.r * Djj.r - Djm1jm1.i * Djj.i -
            (Djjm1.r * Djm1j.r - Djjm1.i * Djm1j.i);
    det.i = Djm1jm1.r * Djj.i + Djm1jm1.i * Djj.r -
            (Djjm1.r * Djm1j.i + Djjm1.i * Djm1j.r);
    nrm = 1.0 / (det.r * det.r + det.i * det.i);
    /* set Ainv(j-1:j,j-1:j) */
    mm = Dinv_ia[j - 1];
#ifdef PRINT_CHECK
    if (Dinv_ja[mm] != j - 1) {
      mexPrintf("Dinv: mismatch (%d,%d)\n", Dinv_ja[mm] + 1, j);
      fflush(stdout);
    }
    if (Dinv_ja[mm + 1] != j) {
      mexPrintf("Dinv: mismatch (%d,%d)\n", Dinv_ja[mm + 1] + 1, j + 1);
      fflush(stdout);
    }
#endif
    Dinv_a[mm].r = (Djj.r * det.r + Djj.i * det.i) * nrm;
    Dinv_a[mm].i = (-Djj.r * det.i + Djj.i * det.r) * nrm;
    Dinv_a[mm + 1].r = -(Djjm1.r * det.r + Djjm1.i * det.i) * nrm;
    Dinv_a[mm + 1].i = -(-Djjm1.r * det.i + Djjm1.i * det.r) * nrm;
    mm = Dinv_ia[j];
#ifdef PRINT_CHECK
    if (Dinv_ja[mm] != j - 1) {
      mexPrintf("ainv: mismatch (%d,%d)\n", Dinv_ja[mm] + 1, j);
      fflush(stdout);
    }
    if (Dinv_ja[mm + 1] != j) {
      mexPrintf("Dinv: mismatch (%d,%d)\n", Dinv_ja[mm + 1] + 1, j + 1);
      fflush(stdout);
    }
#endif
    Dinv_a[mm].r = -(Djm1j.r * det.r + Djm1j.i * det.i) * nrm;
    Dinv_a[mm].i = -(-Djm1j.r * det.i + Djm1j.i * det.r) * nrm;

    Dinv_a[mm + 1].r = (Djm1jm1.r * det.r + Djm1jm1.i * det.i) * nrm;
    Dinv_a[mm + 1].i = (-Djm1jm1.r * det.i + Djm1jm1.i * det.r) * nrm;

    j -= 2;
  }

  while (j >= 0) {
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (ibuff[m] != 0)
        mexPrintf("ibuff dirty at position %d\n", m + 1);
      fflush(stdout);
    }
#endif

    /* 1x1 or 2x2 case ? */
    if (j == 0)
      flag = 0;
    else if (D_ia[j + 1] - D_ia[j] == 1)
      flag = 0;
    else /* 2x2 case */
      flag = -1;

    /* 1x1 case */

    if (!flag) {
/* mexPrintf("j=%d\n",j+1);fflush(stdout); */

/*****************     Compute  UTinv(IT,j), Linv(I,j)     *****************/
/* here we have IT=nz(UT(IT,j)), I=nz(L(I,j)), excluding the diagonal blocks
   UTinv(IT,j)=-UTinv(IT,I)*L(I,j) -Dinv(IT,I)*L(I,j)  -Linv(I,IT)^T *L(I,j)

   1. part:   -UTinv(IT,I) *L(I,j)=-sum_{k in I} UTinv(IT,k)*L(k,j)
   2. part:   -Dinv(IT,I)  *L(I,j)=-sum_{k in I} Dinv(IT,k) *L(k,j)
   3. part:   -Linv(I,IT)^T*L(I,j)=-(Linv(I,k)^T*L(I,j))_{k in IT}


   Linv(I,j)  =-Linv(I,IT) *UT(IT,j)-Dinv(I,IT)*UT(IT,j)-UTinv(IT,I)^T*UT(IT,j)

   4. part:   -Linv(I,IT)   *UT(IT,j)=-sum_{k in IT} Linv(I,k)*UT(k,j)
   5. part:   -Dinv(I,IT)   *UT(IT,j)=-sum_{k in IT} Dinv(I,k)*UT(k,j)
   6. part:   -UTinv(IT,I)^T*UT(IT,j)=-(UTinv(IT,k)^T*UT(IT,j))_{k in I}


   recall that the entries of Linv_a,Dinv_a,UTinv_a are already initialized with
   0.0
*/

/* starting position in L(j+1:n,j)*/
#ifdef PRINT_CHECK
      if (L_ja[L_ia[j]] != j) {
        mexPrintf("L: mismatch (%d,%d)\n", L_ja[L_ia[j]] + 1, j + 1);
        fflush(stdout);
      }
#endif
      /* 1. part + 2. part */
      for (m = L_ia[j] + 1; m < L_ia[j + 1]; m++) {
        /* column index k>j of L(k,j) */
        k = L_ja[m];
/* mexPrintf("(%d,%d)\n",k+1,j+1);fflush(stdout); */

/* starting positions column of UTinv(j+1:n,j) and k of UTinv(k:n,k) */
#ifdef PRINT_CHECK
        if (UTinv_ja[UTinv_ia[j]] == j) {
          mexPrintf("UTinv: mismatch (%d,%d)\n", UTinv_ja[UTinv_ia[j]] + 1,
                    j + 1);
          fflush(stdout);
        }
#endif

        /* 1. part: -sum_{k in I} UTinv(IT,k)*L(k,j) */
        l = UTinv_ia[j];
        mm = UTinv_ia[k];
        while (l < UTinv_ia[j + 1] && mm < UTinv_ia[k + 1]) {
          /* UTinv(kk,j) */
          kk = UTinv_ja[l];
          /* UTinv(ll,k) */
          ll = UTinv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* UTinv(kk,j)=UTinv(kk,j)-UTinv(kk,k)*L(k,j) */
            UTinv_a[l].r -= (flag_piL)
                                ? UTinv_a[mm].r * L_valuesR[m] -
                                      UTinv_a[mm].i * L_valuesI[m]
                                : UTinv_a[mm].r * L_valuesR[m];
            UTinv_a[l].i -= (flag_piL)
                                ? UTinv_a[mm].r * L_valuesI[m] +
                                      UTinv_a[mm].i * L_valuesR[m]
                                : UTinv_a[mm].i * L_valuesR[m];
            l++;
            mm++;
          }
        } /* end while */

        /* 2. part: -sum_{k in I} Dinv(IT,k) *L(k,j) */
        /* check IT cap {k-1,k,k+1} */
        flagkm1 = -1;
        flag = -1;
        flagkp1 = -1;
        for (l = UTinv_ia[j]; l < UTinv_ia[j + 1]; l++) {
          /* UTinv(kk,j) */
          kk = UTinv_ja[l];
          /* store the location l of UTinv(k-1,j) */
          if (kk == k - 1)
            flagkm1 = l;
          /* store the location l of UTinv(k,j) */
          if (kk == k)
            flag = l;
          /* store the location l of UTinv(k+1,j) */
          if (kk == k + 1)
            flagkp1 = l;
        } /* end for l */
#ifdef PRINT_INFO
        if (flagkm1 < 0) {
          mexPrintf("UTinv: row index %d is missing in column %d\n", k, j + 1);
          fflush(stdout);
        }
        if (flag < 0) {
          mexPrintf("UTinv: row index %d is missing in column %d\n", k + 1,
                    j + 1);
          fflush(stdout);
        }
        if (flagkp1 < 0) {
          mexPrintf("UTinv: row index %d is missing in column %d\n", k + 2,
                    j + 1);
          fflush(stdout);
        }
#endif

        /* k is member of IT and Dinv(:,k) is diagonal */
        kk = Dinv_ia[k];
        if (flag >= 0 && Dinv_ia[k + 1] - kk == 1) {
          /* UTinv(k,j)=UTinv(k,j)-Dinv(k,k)*L(k,j) */
          UTinv_a[flag].r -=
              (flag_piL)
                  ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                  : Dinv_a[kk].r * L_valuesR[m];
          UTinv_a[flag].i -=
              (flag_piL)
                  ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                  : Dinv_a[kk].i * L_valuesR[m];
        }
        /* block diagonal case Dinv(:,k) */
        else if (Dinv_ia[k + 1] - kk == 2) {
          if (Dinv_ja[kk] == k - 1 && flagkm1 >= 0) {
            /* UTinv(k-1,j)=UTinv(k-1,j)-Dinv(k-1,k)*L(k,j) */
            UTinv_a[flagkm1].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flagkm1].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          } else if (Dinv_ja[kk] == k && flag >= 0) {
            /* UTinv(k,j)=UTinv(k,j)-Dinv(k,k)*L(k,j) */
            UTinv_a[flag].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flag].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          }
          kk++;
          if (Dinv_ja[kk] == k && flag >= 0) {
            /* UTinv(k,j)=UTinv(k,j)-Dinv(k,k)*L(k,j) */
            UTinv_a[flag].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flag].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          } else if (Dinv_ja[kk] == k + 1 && flagkp1 >= 0) {
            /* UTinv(k+1,j)=UTinv(k+1,j)-Dinv(k+1,k)*L(k,j) */
            UTinv_a[flagkp1].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flagkp1].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          }
        }
      } /* end for m */

      /* 3. part:   -(Linv(I,k)^T*L(I,j))_{k in IT} */
      for (m = UTinv_ia[j]; m < UTinv_ia[j + 1]; m++) {
        /* column index k in IT of UTinv(IT,j) */
        k = UTinv_ja[m];

#ifdef PRINT_CHECK
        if (L_ja[L_ia[j]] != j) {
          mexPrintf("L: mismatch (%d,%d)\n", L_ja[L_ia[j]] + 1, j + 1);
          fflush(stdout);
        }
#endif
        l = L_ia[j] + 1;
        mm = Linv_ia[k];
        while (l < L_ia[j + 1] && mm < Linv_ia[k + 1]) {
          /* L(kk,j) */
          kk = L_ja[l];
          /* Linv(ll,k) */
          ll = Linv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* UTinv(k,j)=UTinv(k,j)-Linv(kk,k)^T*L(kk,j) */
            UTinv_a[m].r -=
                (flag_piL)
                    ? Linv_a[mm].r * L_valuesR[l] - Linv_a[mm].i * L_valuesI[l]
                    : Linv_a[mm].r * L_valuesR[l];
            UTinv_a[m].i -=
                (flag_piL)
                    ? Linv_a[mm].r * L_valuesI[l] + Linv_a[mm].i * L_valuesR[l]
                    : Linv_a[mm].i * L_valuesR[l];
            l++;
            mm++;
          }
        } /* end while */
      }   /* end for m */

/* starting position in UT(j+1:n,j)*/
#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[j]] != j) {
        mexPrintf("UT: mismatch (%d,%d)\n", UT_ja[UT_ia[j]] + 1, j + 1);
        fflush(stdout);
      }
#endif
      /* 4. part + 5. part */
      for (m = UT_ia[j] + 1; m < UT_ia[j + 1]; m++) {
        /* column index k>j of UT(k,j) */
        k = UT_ja[m];
/* mexPrintf("(%d,%d)\n",k+1,j+1);fflush(stdout); */

/* starting positions column of Linv(j+1:n,j) and k of Linv(k:n,k) */
#ifdef PRINT_CHECK
        if (Linv_ja[Linv_ia[j]] == j) {
          mexPrintf("Linv: mismatch (%d,%d)\n", Linv_ja[Linv_ia[j]] + 1, j + 1);
          fflush(stdout);
        }
#endif

        /* 4. part: -sum_{k in IT} Linv(I,k)*UT(k,j) */
        l = Linv_ia[j];
        mm = Linv_ia[k];
        while (l < Linv_ia[j + 1] && mm < Linv_ia[k + 1]) {
          /* Linv(kk,j) */
          kk = Linv_ja[l];
          /* Linv(ll,k) */
          ll = Linv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* Linv(kk,j)=Linv(kk,j)-Linv(kk,k)*UT(k,j) */
            Linv_a[l].r -= (flag_piUT)
                               ? Linv_a[mm].r * UT_valuesR[m] -
                                     Linv_a[mm].i * UT_valuesI[m]
                               : Linv_a[mm].r * UT_valuesR[m];
            Linv_a[l].i -= (flag_piUT)
                               ? Linv_a[mm].r * UT_valuesI[m] +
                                     Linv_a[mm].i * UT_valuesR[m]
                               : Linv_a[mm].i * UT_valuesR[m];
            l++;
            mm++;
          }
        } /* end while */

        /* 5. part: -sum_{k in I} Dinv(IT,k) *UT(k,j) */
        /* check IT cap {k-1,k,k+1} */
        flagkm1 = -1;
        flag = -1;
        flagkp1 = -1;
        for (l = Linv_ia[j]; l < Linv_ia[j + 1]; l++) {
          /* Linv(kk,j) */
          kk = Linv_ja[l];
          /* store the location l of Linv(k-1,j) */
          if (kk == k - 1)
            flagkm1 = l;
          /* store the location l of Linv(k,j) */
          if (kk == k)
            flag = l;
          /* store the location l of Linv(k+1,j) */
          if (kk == k + 1)
            flagkp1 = l;
        } /* end for l */
#ifdef PRINT_INFO
        if (flagkm1 < 0) {
          mexPrintf("Linv: row index %d is missing in column %d\n", k, j + 1);
          fflush(stdout);
        }
        if (flag < 0) {
          mexPrintf("Linv: row index %d is missing in column %d\n", k + 1,
                    j + 1);
          fflush(stdout);
        }
        if (flagkp1 < 0) {
          mexPrintf("Linv: row index %d is missing in column %d\n", k + 2,
                    j + 1);
          fflush(stdout);
        }
#endif

        /* k is member of IT and Dinv(:,k) is diagonal */
        kk = Dinv_ia[k];
        if (flag >= 0 && Dinv_ia[k + 1] - kk == 1) {
          /* Linv(k,j)=Linv(k,j)-Dinv(k,k)*UT(k,j) */
          Linv_a[flag].r -=
              (flag_piUT)
                  ? Dinv_a[kk].r * UT_valuesR[m] - Dinv_a[kk].i * UT_valuesI[m]
                  : Dinv_a[kk].r * UT_valuesR[m];
          Linv_a[flag].i -=
              (flag_piUT)
                  ? Dinv_a[kk].r * UT_valuesI[m] + Dinv_a[kk].i * UT_valuesR[m]
                  : Dinv_a[kk].i * UT_valuesR[m];
        }
        /* block diagonal case Dinv(:,k) */
        else if (Dinv_ia[k + 1] - kk == 2) {
          if (Dinv_ja[kk] == k - 1 && flagkm1 >= 0) {
            /* Linv(k-1,j)=Linv(k-1,j)-Dinv(k-1,k)*UT(k,j) */
            Linv_a[flagkm1].r -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesR[m] -
                                           Dinv_a[kk].i * UT_valuesI[m]
                                     : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flagkm1].i -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesI[m] +
                                           Dinv_a[kk].i * UT_valuesR[m]
                                     : Dinv_a[kk].i * UT_valuesR[m];
          } else if (Dinv_ja[kk] == k && flag >= 0) {
            /* Linv(k,j)=Linv(k,j)-Dinv(k,k)*UT(k,j) */
            Linv_a[flag].r -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesR[m] -
                                        Dinv_a[kk].i * UT_valuesI[m]
                                  : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flag].i -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesI[m] +
                                        Dinv_a[kk].i * UT_valuesR[m]
                                  : Dinv_a[kk].i * UT_valuesR[m];
          }
          kk++;
          if (Dinv_ja[kk] == k && flag >= 0) {
            /* Linv(k,j)=Linv(k,j)-Dinv(k,k)*UT(k,j) */
            Linv_a[flag].r -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesR[m] -
                                        Dinv_a[kk].i * UT_valuesI[m]
                                  : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flag].i -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesI[m] +
                                        Dinv_a[kk].i * UT_valuesR[m]
                                  : Dinv_a[kk].i * UT_valuesR[m];
          } else if (Dinv_ja[kk] == k + 1 && flagkp1 >= 0) {
            /* UTinv(k+1,j)=UTinv(k+1,j)-Dinv(k+1,k)*UT(k,j) */
            Linv_a[flagkp1].r -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesR[m] -
                                           Dinv_a[kk].i * UT_valuesI[m]
                                     : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flagkp1].i -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesI[m] +
                                           Dinv_a[kk].i * UT_valuesR[m]
                                     : Dinv_a[kk].i * UT_valuesR[m];
          }
        }
      } /* end for m */

      /* 6. part:   -(UTinv(IT,k)^T*UT(IT,j))_{k in I} */
      for (m = Linv_ia[j]; m < Linv_ia[j + 1]; m++) {
        /* column index k in I of Linv(I,j) */
        k = Linv_ja[m];

#ifdef PRINT_CHECK
        if (UT_ja[UT_ia[j]] != j) {
          mexPrintf("UT: mismatch (%d,%d)\n", UT_ja[UT_ia[j]] + 1, j + 1);
          fflush(stdout);
        }
#endif
        l = UT_ia[j] + 1;
        mm = UTinv_ia[k];
        while (l < UT_ia[j + 1] && mm < UTinv_ia[k + 1]) {
          /* UT(kk,j) */
          kk = UT_ja[l];
          /* UTinv(ll,k) */
          ll = UTinv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* Linv(k,j)=Linv(k,j)-UTinv(kk,k)^T*UT(kk,j) */
            Linv_a[m].r -= (flag_piUT)
                               ? UTinv_a[mm].r * UT_valuesR[l] -
                                     UTinv_a[mm].i * UT_valuesI[l]
                               : UTinv_a[mm].r * UT_valuesR[l];
            Linv_a[m].i -= (flag_piUT)
                               ? UTinv_a[mm].r * UT_valuesI[l] +
                                     UTinv_a[mm].i * UT_valuesR[l]
                               : UTinv_a[mm].i * UT_valuesR[l];
            l++;
            mm++;
          }
        } /* end while */
      }   /* end for m */
      /*********   END Computation UTinv(IT,j), Linv(I,j)   *****************/

      /*********** Compute Dinv(j,j)=1/D(j,j)-UT(IT,j)^T*UTinv(IT,j)
       * ***********/
      mm = Dinv_ia[j];
      m = D_ia[j];
      nrm = (flag_piD)
                ? 1.0 / (D_valuesR[m] * D_valuesR[m] +
                         D_valuesI[m] * D_valuesI[m])
                : 1.0 / (D_valuesR[m] * D_valuesR[m]);

      Dinv_a[mm].r = D_valuesR[m] * nrm;
      Dinv_a[mm].i = (flag_piD) ? -D_valuesI[m] * nrm : 0.0;
#ifdef PRINT_CHECK
      if (Dinv_ja[mm] != D_ja[m]) {
        mexPrintf("Dinv,D: index mismatch at (%d,%d)\n", Dinv_ja[mm] + 1,
                  D_ja[m] + 1);
        fflush(stdout);
      }
#endif
#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[j]] != j) {
        mexPrintf("UT: mismatch (%d,%d)\n", UT_ja[UT_ia[j]] + 1, j + 1);
        fflush(stdout);
      }
#endif
#ifdef PRINT_CHECK
      if (UTinv_ja[UTinv_ia[j]] == j) {
        mexPrintf("UTinv: mismatch (%d,%d)\n", UTinv_ja[UTinv_ia[j]] + 1,
                  j + 1);
        fflush(stdout);
      }
#endif
      l = UTinv_ia[j];
      m = UT_ia[j] + 1;
      while (l < UTinv_ia[j + 1] && m < UT_ia[j + 1]) {
        /* UT(kk,j) */
        kk = UT_ja[m];
        /* UTinv(ll,j) */
        ll = UTinv_ja[l];
        if (kk < ll)
          m++;
        else if (ll < kk)
          l++;
        else { /* indices match */
          /* Dinv(j,j)=Dinv(j,j)-UT(kk,j)^T*UTinv(kk,j) */
          Dinv_a[mm].r -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].r - UT_valuesI[m] * UTinv_a[l].i
                  : UT_valuesR[m] * UTinv_a[l].r;
          Dinv_a[mm].i -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].i + UT_valuesI[m] * UTinv_a[l].r
                  : UT_valuesR[m] * UTinv_a[l].i;
          m++;
          l++;
        }
      } /* end while */
      /******* END Computation Dinv(j,j)=1/D(j,j)-UT(I,j)^T*UTinv(I,j) *******/

      j = j - 1;
    } else { /* 2x2 case */
             /* mexPrintf("j=%d:%d\n",j,j+1);fflush(stdout); */
/*****************   Compute  UTinv(IT,j-1:j), Linv(I,j-1:j) *****************/
/* here we have IT=nz(UT(IT,j-1:j)), I=nz(L(I,j-1:j)), excluding the diagonal
   blocks
   UTinv(IT,j-1:j)=-UTinv(IT,I) *L(I,j-1:j)
                   -Dinv(IT,I)  *L(I,j-1:j)
                   -Linv(I,IT)^T*L(I,j-1:j)

   1. part:  -UTinv(IT,I) *L(I,j-1:j)=-sum_{k in I} UTinv(IT,k)*L(k,j-1:j)
   2. part:  -Dinv(IT,I)  *L(I,j-1:j)=-sum_{k in I} Dinv(IT,k) *L(k,j-1:j)
   3. part:  -Linv(I,IT)^T*L(I,j-1:j)=-(Linv(I,k)^T*L(I,j-1:j))_{k in IT}


   Linv(I,j-1:j)  =-Linv(I,IT)   *UT(IT,j-1:j)
                   -Dinv(I,IT)   *UT(IT,j-1:j)
                   -UTinv(IT,I)^T*UT(IT,j-1:j)

   4. part:  -Linv(I,IT)   *UT(IT,j-1:j)=-sum_{k in IT} Linv(I,k)*UT(k,j-1:j)
   5. part:  -Dinv(I,IT)   *UT(IT,j-1:j)=-sum_{k in IT} Dinv(I,k)*UT(k,j-1:j)
   6. part:  -UTinv(IT,I)^T*UT(IT,j-1:j)=-(UTinv(IT,k)^T*UT(IT,j-1:j))_{k in I}


   recall that the entries of Linv_a,Dinv_a,UTinv_a are already initialized with
   0.0
*/

/* starting position in column L(j+1:n,j-1) */
#ifdef PRINT_CHECK
      if (L_ja[L_ia[j - 1]] != j - 1) {
        mexPrintf("L: diagonal index mismatch at (%d,%d)\n",
                  L_ja[L_ia[j - 1]] + 1, j);
        fflush(stdout);
      }
#endif
      /* 1. part + 2. part */
      m = L_ia[j - 1] + 1;
/* skip L(j,j-1) if present (which should be zero) */
#ifdef PRINT_CHECK
      if (L_ja[m] <= j) {
        mexPrintf("L: L(%d,%d)=%8.1le=0?\n", L_ja[m] + 1, j, L_valuesR[m]);
        fflush(stdout);
      }
#endif
      if (L_ja[m] <= j)
        m++;
      for (; m < L_ia[j]; m++) {
        /* row index k of L(k,j-1) */
        k = L_ja[m];

/* starting positions column of UTinv(j+1:n,j-1) and k of UTinv(k:n,k) */
#ifdef PRINT_CHECK
        if (UTinv_ja[UTinv_ia[j - 1]] == j - 1 ||
            UTinv_ja[UTinv_ia[j - 1] + 1] == j) {
          mexPrintf("UTinv: diagonal or sub-diagonal index mismatch at "
                    "(%d,%d),(%d,%d)\n",
                    UTinv_ja[UTinv_ia[j - 1]] + 1, j,
                    UTinv_ja[UTinv_ia[j - 1] + 1] + 1, j);
          fflush(stdout);
        }
#endif
        /* 2. part: -sum_{k in I} UTinv(I,k)*L(k,j-1) */
        l = UTinv_ia[j - 1];
        mm = UTinv_ia[k];
        while (l < UTinv_ia[j] && mm < UTinv_ia[k + 1]) {
          /* UTinv(kk,j-1) */
          kk = UTinv_ja[l];
          /* UTinv(ll,k) */
          ll = UTinv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* UTinv(kk,j-1)=UTinv(kk,j-1)-UTinv(kk,k)*L(k,j-1) */
            UTinv_a[l].r -= (flag_piL)
                                ? UTinv_a[mm].r * L_valuesR[m] -
                                      UTinv_a[mm].i * L_valuesI[m]
                                : UTinv_a[mm].r * L_valuesR[m];
            UTinv_a[l].i -= (flag_piL)
                                ? UTinv_a[mm].r * L_valuesI[m] +
                                      UTinv_a[mm].i * L_valuesR[m]
                                : UTinv_a[mm].i * L_valuesR[m];
            l++;
            mm++;
          }
        } /* end while */

        /* 2. part: -sum_{k in I} Dinv(IT,k) *L(k,j-1) */
        /* check IT cap {k-1,k,k+1} */
        flagkm1 = -1;
        flag = -1;
        flagkp1 = -1;
        for (l = UTinv_ia[j - 1]; l < UTinv_ia[j]; l++) {
          /* UTinv(kk,j-1) */
          kk = UTinv_ja[l];
          /* store the location l of UTinv(k-1,j-1) */
          if (kk == k - 1)
            flagkm1 = l;
          /* store the location l of UTinv(k,j-1) */
          if (kk == k)
            flag = l;
          /* store the location l of UTinv(k+1,j-1) */
          if (kk == k + 1)
            flagkp1 = l;
        } /* end for l */
#ifdef PRINT_INFO
        if (flagkm1 < 0) {
          mexPrintf("UTinv: row index %d is missing in column %d\n", k, j);
          fflush(stdout);
        }
        if (flag < 0) {
          mexPrintf("UTinv: row index %d is missing in column %d\n", k + 1, j);
          fflush(stdout);
        }
        if (flagkp1 < 0) {
          mexPrintf("UTinv: row index %d is missing in column %d\n", k + 2, j);
          fflush(stdout);
        }
#endif

        /* k is member of IT and Dinv(:,k) is diagonal */
        kk = Dinv_ia[k];
        if (flag >= 0 && Dinv_ia[k + 1] - kk == 1) {
          /* UTinv(k,j-1)=UTinv(k,j-1)-Dinv(k,k)*L(k,j-1) */
          UTinv_a[flag].r -=
              (flag_piL)
                  ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                  : Dinv_a[kk].r * L_valuesR[m];
          UTinv_a[flag].i -=
              (flag_piL)
                  ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                  : Dinv_a[kk].i * L_valuesR[m];
        }
        /* block diagonal case Dinv(:,k) */
        else if (Dinv_ia[k + 1] - kk == 2) {
          if (Dinv_ja[kk] == k - 1 && flagkm1 >= 0) {
            /* UTinv(k-1,j-1)=UTinv(k-1,j-1)-Dinv(k-1,k)*L(k,j-1) */
            UTinv_a[flagkm1].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flagkm1].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          } else if (Dinv_ja[kk] == k && flag >= 0) {
            /* UTinv(k,j-1)=UTinv(k,j-1)-Dinv(k,k)*L(k,j-1) */
            UTinv_a[flag].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flag].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          }
          kk++;
          if (Dinv_ja[kk] == k && flag >= 0) {
            /* UTinv(k,j-1)=UTinv(k,j-1)-Dinv(k,k)*L(k,j-1) */
            UTinv_a[flag].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flag].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          } else if (Dinv_ja[kk] == k + 1 && flagkp1 >= 0) {
            /* UTinv(k+1,j-1)=UTinv(k+1,j-1)-Dinv(k+1,k)*L(k,j-1) */
            UTinv_a[flagkp1].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flagkp1].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          }
        }
      } /* end for m */

      /* 3. part:   -(Linv(I,k)^T*L(I,j-1))_{k in IT} */
      for (m = UTinv_ia[j - 1]; m < UTinv_ia[j]; m++) {
        /* column index k in IT of UTinv(IT,j-1) */
        k = UTinv_ja[m];

#ifdef PRINT_CHECK
        if (L_ja[L_ia[j - 1]] != j - 1) {
          mexPrintf("L: mismatch (%d,%d)\n", L_ja[L_ia[j - 1]] + 1, j);
          fflush(stdout);
        }
#endif
        l = L_ia[j - 1] + 1;
        mm = Linv_ia[k];
/* skip L(j,j-1) if present (which should be zero) */
#ifdef PRINT_CHECK
        if (L_ja[l] <= j) {
          mexPrintf("L: L(%d,%d)=%8.1le=0??\n", L_ja[l] + 1, j, L_valuesR[l]);
          fflush(stdout);
        }
#endif
        if (L_ja[l] <= j)
          l++;
        while (l < L_ia[j] && mm < Linv_ia[k + 1]) {
          /* L(kk,j-1) */
          kk = L_ja[l];
          /* Linv(ll,k) */
          ll = Linv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* UTinv(k,j-1)=UTinv(k,j-1)-Linv(kk,k)^T*L(kk,j-1) */
            UTinv_a[m].r -=
                (flag_piL)
                    ? Linv_a[mm].r * L_valuesR[l] - Linv_a[mm].i * L_valuesI[l]
                    : Linv_a[mm].r * L_valuesR[l];
            UTinv_a[m].i -=
                (flag_piL)
                    ? Linv_a[mm].r * L_valuesI[l] + Linv_a[mm].i * L_valuesR[l]
                    : Linv_a[mm].i * L_valuesR[l];
            l++;
            mm++;
          }
        } /* end while */
      }   /* end for m */

/* next column, starting position in L(j+1:n,j) */
#ifdef PRINT_CHECK
      if (L_ja[L_ia[j]] != j) {
        mexPrintf("L: diagonal index mismatch at (%d,%d)\n", L_ja[L_ia[j]] + 1,
                  j + 1);
        fflush(stdout);
      }
#endif
      /* 1. part + 2. part */
      for (m = L_ia[j] + 1; m < L_ia[j + 1]; m++) {
        /* column index k>j of L(k,j) */
        k = L_ja[m];

/* starting positions column of UTinv(j+1:n,j) and k of UTinv(k:n,k) */
#ifdef PRINT_CHECK
        if (UTinv_ja[UTinv_ia[j]] == j) {
          mexPrintf("UTinv: mismatch (%d,%d)\n", UTinv_ja[UTinv_ia[j]] + 1,
                    j + 1);
          fflush(stdout);
        }
#endif

        /* 1. part: -sum_{k in I} UTinv(IT,k)*L(k,j) */
        l = UTinv_ia[j];
        mm = UTinv_ia[k];
        while (l < UTinv_ia[j + 1] && mm < UTinv_ia[k + 1]) {
          /* UTinv(kk,j) */
          kk = UTinv_ja[l];
          /* UTinv(ll,k) */
          ll = UTinv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* UTinv(kk,j)=UTinv(kk,j)-UTinv(kk,k)*L(k,j) */
            UTinv_a[l].r -= (flag_piL)
                                ? UTinv_a[mm].r * L_valuesR[m] -
                                      UTinv_a[mm].i * L_valuesI[m]
                                : UTinv_a[mm].r * L_valuesR[m];
            UTinv_a[l].i -= (flag_piL)
                                ? UTinv_a[mm].r * L_valuesI[m] +
                                      UTinv_a[mm].i * L_valuesR[m]
                                : UTinv_a[mm].i * L_valuesR[m];
            l++;
            mm++;
          }
        } /* end while */

        /* 2. part: -sum_{k in I} Dinv(IT,k) *L(k,j) */
        /* check IT cap {k-1,k,k+1} */
        flagkm1 = -1;
        flag = -1;
        flagkp1 = -1;
        for (l = UTinv_ia[j]; l < UTinv_ia[j + 1]; l++) {
          /* UTinv(kk,j) */
          kk = UTinv_ja[l];
          /* store the location l of UTinv(k-1,j) */
          if (kk == k - 1)
            flagkm1 = l;
          /* store the location l of UTinv(k,j) */
          if (kk == k)
            flag = l;
          /* store the location l of UTinv(k+1,j) */
          if (kk == k + 1)
            flagkp1 = l;
        } /* end while */
#ifdef PRINT_INFO
        if (flagkm1 < 0) {
          mexPrintf("UTinv: row index %d is missing in column %d\n", k, j + 1);
          fflush(stdout);
        }
        if (flag < 0) {
          mexPrintf("UTinv: row index %d is missing in column %d\n", k + 1,
                    j + 1);
          fflush(stdout);
        }
        if (flagkp1 < 0) {
          mexPrintf("UTinv: row index %d is missing in column %d\n", k + 2,
                    j + 1);
          fflush(stdout);
        }
#endif

        /* k is member of IT and Dinv(:,k) is diagonal */
        kk = Dinv_ia[k];
        if (flag >= 0 && Dinv_ia[k + 1] - kk == 1) {
          /* UTinv(k,j)=UTinv(k,j)-Dinv(k,k)*L(k,j) */
          UTinv_a[flag].r -=
              (flag_piL)
                  ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                  : Dinv_a[kk].r * L_valuesR[m];
          UTinv_a[flag].i -=
              (flag_piL)
                  ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                  : Dinv_a[kk].i * L_valuesR[m];
        }
        /* block diagonal case Dinv(:,k) */
        else if (Dinv_ia[k + 1] - kk == 2) {
          if (Dinv_ja[kk] == k - 1 && flagkm1 >= 0) {
            /* UTinv(k-1,j)=UTinv(k-1,j)-Dinv(k-1,k)*L(k,j) */
            UTinv_a[flagkm1].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flagkm1].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          } else if (Dinv_ja[kk] == k && flag >= 0) {
            /* UTinv(k,j)=UTinv(k,j)-Dinv(k,k)*L(k,j) */
            UTinv_a[flag].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flag].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          }
          kk++;
          if (Dinv_ja[kk] == k && flag >= 0) {
            /* UTinv(k,j)=UTinv(k,j)-Dinv(k,k)*L(k,j) */
            UTinv_a[flag].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flag].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          } else if (Dinv_ja[kk] == k + 1 && flagkp1 >= 0) {
            /* UTinv(k+1,j)=UTinv(k+1,j)-Dinv(k+1,k)*L(k,j) */
            UTinv_a[flagkp1].r -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesR[m] - Dinv_a[kk].i * L_valuesI[m]
                    : Dinv_a[kk].r * L_valuesR[m];
            UTinv_a[flagkp1].i -=
                (flag_piL)
                    ? Dinv_a[kk].r * L_valuesI[m] + Dinv_a[kk].i * L_valuesR[m]
                    : Dinv_a[kk].i * L_valuesR[m];
          }
        }
      } /* end for m */

      /* 3. part:   -(Linv(I,k)^T*L(I,j))_{k in IT} */
      for (m = UTinv_ia[j]; m < UTinv_ia[j + 1]; m++) {
        /* column index k in IT of UTinv(IT,j) */
        k = UTinv_ja[m];

#ifdef PRINT_CHECK
        if (L_ja[L_ia[j]] != j) {
          mexPrintf("L: mismatch (%d,%d)\n", L_ja[L_ia[j]] + 1, j + 1);
          fflush(stdout);
        }
#endif
        l = L_ia[j] + 1;
        mm = Linv_ia[k];
        while (l < L_ia[j + 1] && mm < Linv_ia[k + 1]) {
          /* L(kk,j) */
          kk = L_ja[l];
          /* Linv(ll,k) */
          ll = Linv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* UTinv(k,j)=UTinv(k,j)-Linv(kk,k)^T*L(kk,j) */
            UTinv_a[m].r -=
                (flag_piL)
                    ? Linv_a[mm].r * L_valuesR[l] - Linv_a[mm].i * L_valuesI[l]
                    : Linv_a[mm].r * L_valuesR[l];
            UTinv_a[m].i -=
                (flag_piL)
                    ? Linv_a[mm].r * L_valuesI[l] + Linv_a[mm].i * L_valuesR[l]
                    : Linv_a[mm].i * L_valuesR[l];
            l++;
            mm++;
          }
        } /* end while */
      }   /* end for m */

/* starting position in UT(j+1:n,j-1)*/
#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[j - 1]] != j - 1) {
        mexPrintf("UT: diagonal index mismatch (%d,%d)\n",
                  UT_ja[UT_ia[j - 1]] + 1, j);
        fflush(stdout);
      }
#endif
      /* 4. part + 5. part */
      m = UT_ia[j - 1] + 1;
/* skip UT(j,j-1) if present (which should be zero) */
#ifdef PRINT_CHECK
      if (UT_ja[m] <= j) {
        mexPrintf("UT: UT(%d,%d)=%8.1le=0?\n", UT_ja[m] + 1, j, UT_valuesR[m]);
        fflush(stdout);
      }
#endif
      if (UT_ja[m] <= j)
        m++;
      for (; m < UT_ia[j]; m++) {

        /* column index k>j of UT(k,j-1) */
        k = UT_ja[m];

/* starting positions column of Linv(j+1:n,j-1) and k of Linv(k:n,k) */
#ifdef PRINT_CHECK
        if (Linv_ja[Linv_ia[j - 1]] == j - 1 ||
            Linv_ja[Linv_ia[j - 1] + 1] == j - 1) {
          mexPrintf("Linv: diagonal or sub-diagonal index mismatch at "
                    "(%d,%d),(%d,%d)\n",
                    Linv_ja[Linv_ia[j - 1]] + 1, j,
                    Linv_ja[Linv_ia[j - 1] + 1] + 1, j);
          fflush(stdout);
        }
#endif

        /* 4. part: -sum_{k in IT} Linv(I,k)*UT(k,j-1) */
        l = Linv_ia[j - 1];
        mm = Linv_ia[k];
        while (l < Linv_ia[j] && mm < Linv_ia[k + 1]) {
          /* Linv(kk,j-1) */
          kk = Linv_ja[l];
          /* Linv(ll,k) */
          ll = Linv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* Linv(kk,j-1)=Linv(kk,j-1)-Linv(kk,k)*UT(k,j-1) */
            Linv_a[l].r -= (flag_piUT)
                               ? Linv_a[mm].r * UT_valuesR[m] -
                                     Linv_a[mm].i * UT_valuesI[m]
                               : Linv_a[mm].r * UT_valuesR[m];
            Linv_a[l].i -= (flag_piUT)
                               ? Linv_a[mm].r * UT_valuesI[m] +
                                     Linv_a[mm].i * UT_valuesR[m]
                               : Linv_a[mm].i * UT_valuesR[m];
            l++;
            mm++;
          }
        } /* end while */

        /* 5. part: -sum_{k in IT} Dinv(I,k) *UT(k,j-1) */
        /* check I cap {k-1,k,k+1} */
        flagkm1 = -1;
        flag = -1;
        flagkp1 = -1;
        for (l = Linv_ia[j - 1]; l < Linv_ia[j]; l++) {
          /* Linv(kk,j-1) */
          kk = Linv_ja[l];
          /* store the location l of Linv(k-1,j-1) */
          if (kk == k - 1)
            flagkm1 = l;
          /* store the location l of Linv(k,j-1) */
          if (kk == k)
            flag = l;
          /* store the location l of Linv(k+1,j-1) */
          if (kk == k + 1)
            flagkp1 = l;
        } /* end for l */
#ifdef PRINT_INFO
        if (flagkm1 < 0) {
          mexPrintf("Linv: row index %d is missing in column %d\n", k, j);
          fflush(stdout);
        }
        if (flag < 0) {
          mexPrintf("Linv: row index %d is missing in column %d\n", k + 1, j);
          fflush(stdout);
        }
        if (flagkp1 < 0) {
          mexPrintf("Linv: row index %d is missing in column %d\n", k + 2, j);
          fflush(stdout);
        }
#endif

        /* k is member of I and Dinv(:,k) is diagonal */
        kk = Dinv_ia[k];
        if (flag >= 0 && Dinv_ia[k + 1] - kk == 1) {
          /* Linv(k,j-1)=Linv(k,j-1)-Dinv(k,k)*UT(k,j-1) */
          Linv_a[flag].r -=
              (flag_piUT)
                  ? Dinv_a[kk].r * UT_valuesR[m] - Dinv_a[kk].i * UT_valuesI[m]
                  : Dinv_a[kk].r * UT_valuesR[m];
          Linv_a[flag].i -=
              (flag_piUT)
                  ? Dinv_a[kk].r * UT_valuesI[m] + Dinv_a[kk].i * UT_valuesR[m]
                  : Dinv_a[kk].i * UT_valuesR[m];
        }
        /* block diagonal case Dinv(:,k) */
        else if (Dinv_ia[k + 1] - kk == 2) {
          if (Dinv_ja[kk] == k - 1 && flagkm1 >= 0) {
            /* Linv(k-1,j-1)=Linv(k-1,j-1)-Dinv(k-1,k)*UT(k,j-1) */
            Linv_a[flagkm1].r -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesR[m] -
                                           Dinv_a[kk].i * UT_valuesI[m]
                                     : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flagkm1].i -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesI[m] +
                                           Dinv_a[kk].i * UT_valuesR[m]
                                     : Dinv_a[kk].i * UT_valuesR[m];
          } else if (Dinv_ja[kk] == k && flag >= 0) {
            /* Linv(k,j-1)=Linv(k,j-1)-Dinv(k,k)*UT(k,j-1) */
            Linv_a[flag].r -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesR[m] -
                                        Dinv_a[kk].i * UT_valuesI[m]
                                  : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flag].i -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesI[m] +
                                        Dinv_a[kk].i * UT_valuesR[m]
                                  : Dinv_a[kk].i * UT_valuesR[m];
          }
          kk++;
          if (Dinv_ja[kk] == k && flag >= 0) {
            /* Linv(k,j-1)=Linv(k,j-1)-Dinv(k,k)*UT(k,j-1) */
            Linv_a[flag].r -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesR[m] -
                                        Dinv_a[kk].i * UT_valuesI[m]
                                  : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flag].i -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesI[m] +
                                        Dinv_a[kk].i * UT_valuesR[m]
                                  : Dinv_a[kk].i * UT_valuesR[m];
          } else if (Dinv_ja[kk] == k + 1 && flagkp1 >= 0) {
            /* UTinv(k+1,j-1)=UTinv(k+1,j-1)-Dinv(k+1,k)*UT(k,j-1) */
            Linv_a[flagkp1].r -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesR[m] -
                                           Dinv_a[kk].i * UT_valuesI[m]
                                     : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flagkp1].i -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesI[m] +
                                           Dinv_a[kk].i * UT_valuesR[m]
                                     : Dinv_a[kk].i * UT_valuesR[m];
          }
        }
      } /* end for m */

      /* 6. part:   -(UTinv(IT,k)^T*UT(IT,j-1))_{k in I} */
      for (m = Linv_ia[j - 1]; m < Linv_ia[j]; m++) {
        /* column index k in I of Linv(I,j-1) */
        k = Linv_ja[m];

#ifdef PRINT_CHECK
        if (UT_ja[UT_ia[j - 1]] != j - 1) {
          mexPrintf("UT: mismatch (%d,%d)\n", UT_ja[UT_ia[j - 1]] + 1, j);
          fflush(stdout);
        }
#endif
        l = UT_ia[j - 1] + 1;
        mm = UTinv_ia[k];
/* skip UT(j,j-1) if present (which should be zero) */
#ifdef PRINT_CHECK
        if (UT_ja[l] <= j) {
          mexPrintf("UT: UT(%d,%d)=%8.1le=0??\n", UT_ja[l] + 1, j,
                    UT_valuesR[l]);
          fflush(stdout);
        }
#endif
        if (UT_ja[l] <= j)
          l++;
        while (l < UT_ia[j] && mm < UTinv_ia[k + 1]) {
          /* UT(kk,j-1) */
          kk = UT_ja[l];
          /* UTinv(ll,k) */
          ll = UTinv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* Linv(k,j-1)=Linv(k,j-1)-UTinv(kk,k)^T*UT(kk,j-1) */
            Linv_a[m].r -= (flag_piUT)
                               ? UTinv_a[mm].r * UT_valuesR[l] -
                                     UTinv_a[mm].i * UT_valuesI[l]
                               : UTinv_a[mm].r * UT_valuesR[l];
            Linv_a[m].i -= (flag_piUT)
                               ? UTinv_a[mm].r * UT_valuesI[l] +
                                     UTinv_a[mm].i * UT_valuesR[l]
                               : UTinv_a[mm].i * UT_valuesR[l];
            l++;
            mm++;
          }
        } /* end while */
      }   /* end for m */

/* next column, starting position in UT(j+1:n,j) */
#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[j]] != j) {
        mexPrintf("UT: diagonal index mismatch (%d,%d)\n", UT_ja[UT_ia[j]] + 1,
                  j + 1);
        fflush(stdout);
      }
#endif
      /* 4. part + 5. part */
      for (m = UT_ia[j] + 1; m < UT_ia[j + 1]; m++) {
        /* column index k>j of UT(k,j) */
        k = UT_ja[m];

/* starting positions column of Linv(j+1:n,j) and k of Linv(k:n,k) */
#ifdef PRINT_CHECK
        if (Linv_ja[Linv_ia[j]] == j) {
          mexPrintf("Linv: mismatch (%d,%d)\n", Linv_ja[Linv_ia[j]] + 1, j + 1);
          fflush(stdout);
        }
#endif

        /* 4. part: -sum_{k in IT} Linv(I,k)*UT(k,j) */
        l = Linv_ia[j];
        mm = Linv_ia[k];
        while (l < Linv_ia[j + 1] && mm < Linv_ia[k + 1]) {
          /* Linv(kk,j) */
          kk = Linv_ja[l];
          /* Linv(ll,k) */
          ll = Linv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* Linv(kk,j)=Linv(kk,j)-Linv(kk,k)*UT(k,j) */
            Linv_a[l].r -= (flag_piUT)
                               ? Linv_a[mm].r * UT_valuesR[m] -
                                     Linv_a[mm].i * UT_valuesI[m]
                               : Linv_a[mm].r * UT_valuesR[m];
            Linv_a[l].i -= (flag_piUT)
                               ? Linv_a[mm].r * UT_valuesI[m] +
                                     Linv_a[mm].i * UT_valuesR[m]
                               : Linv_a[mm].i * UT_valuesR[m];
            l++;
            mm++;
          }
        } /* end while */

        /* 5. part: -sum_{k in I} Dinv(IT,k) *UT(k,j) */
        /* check IT cap {k-1,k,k+1} */
        flagkm1 = -1;
        flag = -1;
        flagkp1 = -1;
        for (l = Linv_ia[j]; l < Linv_ia[j + 1]; l++) {
          /* Linv(kk,j) */
          kk = Linv_ja[l];
          /* store the location l of Linv(k-1,j) */
          if (kk == k - 1)
            flagkm1 = l;
          /* store the location l of Linv(k,j) */
          if (kk == k)
            flag = l;
          /* store the location l of Linv(k+1,j) */
          if (kk == k + 1)
            flagkp1 = l;
        } /* end for */
#ifdef PRINT_INFO
        if (flagkm1 < 0) {
          mexPrintf("Linv: row index %d is missing in column %d\n", k, j + 1);
          fflush(stdout);
        }
        if (flag < 0) {
          mexPrintf("Linv: row index %d is missing in column %d\n", k + 1,
                    j + 1);
          fflush(stdout);
        }
        if (flagkp1 < 0) {
          mexPrintf("Linv: row index %d is missing in column %d\n", k + 2,
                    j + 1);
          fflush(stdout);
        }
#endif

        /* k is member of IT and Dinv(:,k) is diagonal */
        kk = Dinv_ia[k];
        if (flag >= 0 && Dinv_ia[k + 1] - kk == 1) {
          /* Linv(k,j)=Linv(k,j)-Dinv(k,k)*UT(k,j) */
          Linv_a[flag].r -=
              (flag_piUT)
                  ? Dinv_a[kk].r * UT_valuesR[m] - Dinv_a[kk].i * UT_valuesI[m]
                  : Dinv_a[kk].r * UT_valuesR[m];
          Linv_a[flag].i -=
              (flag_piUT)
                  ? Dinv_a[kk].r * UT_valuesI[m] + Dinv_a[kk].i * UT_valuesR[m]
                  : Dinv_a[kk].i * UT_valuesR[m];
        }
        /* block diagonal case Dinv(:,k) */
        else if (Dinv_ia[k + 1] - kk == 2) {
          if (Dinv_ja[kk] == k - 1 && flagkm1 >= 0) {
            /* Linv(k-1,j)=Linv(k-1,j)-Dinv(k-1,k)*UT(k,j) */
            Linv_a[flagkm1].r -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesR[m] -
                                           Dinv_a[kk].i * UT_valuesI[m]
                                     : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flagkm1].i -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesI[m] +
                                           Dinv_a[kk].i * UT_valuesR[m]
                                     : Dinv_a[kk].i * UT_valuesR[m];
          } else if (Dinv_ja[kk] == k && flag >= 0) {
            /* Linv(k,j)=Linv(k,j)-Dinv(k,k)*UT(k,j) */
            Linv_a[flag].r -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesR[m] -
                                        Dinv_a[kk].i * UT_valuesI[m]
                                  : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flag].i -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesI[m] +
                                        Dinv_a[kk].i * UT_valuesR[m]
                                  : Dinv_a[kk].i * UT_valuesR[m];
          }
          kk++;
          if (Dinv_ja[kk] == k && flag >= 0) {
            /* Linv(k,j)=Linv(k,j)-Dinv(k,k)*UT(k,j) */
            Linv_a[flag].r -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesR[m] -
                                        Dinv_a[kk].i * UT_valuesI[m]
                                  : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flag].i -= (flag_piUT)
                                  ? Dinv_a[kk].r * UT_valuesI[m] +
                                        Dinv_a[kk].i * UT_valuesR[m]
                                  : Dinv_a[kk].i * UT_valuesR[m];
          } else if (Dinv_ja[kk] == k + 1 && flagkp1 >= 0) {
            /* UTinv(k+1,j)=UTinv(k+1,j)-Dinv(k+1,k)*UT(k,j) */
            Linv_a[flagkp1].r -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesR[m] -
                                           Dinv_a[kk].i * UT_valuesI[m]
                                     : Dinv_a[kk].r * UT_valuesR[m];
            Linv_a[flagkp1].i -= (flag_piUT)
                                     ? Dinv_a[kk].r * UT_valuesI[m] +
                                           Dinv_a[kk].i * UT_valuesR[m]
                                     : Dinv_a[kk].i * UT_valuesR[m];
          }
        }
      } /* end for m */

      /* 6. part:   -(UTinv(IT,k)^T*UT(IT,j))_{k in I} */
      for (m = Linv_ia[j]; m < Linv_ia[j + 1]; m++) {
        /* column index k in I of Linv(I,j) */
        k = Linv_ja[m];

#ifdef PRINT_CHECK
        if (UT_ja[UT_ia[j]] != j) {
          mexPrintf("UT: mismatch (%d,%d)\n", UT_ja[UT_ia[j]] + 1, j + 1);
          fflush(stdout);
        }
#endif
        l = UT_ia[j] + 1;
        mm = UTinv_ia[k];
        while (l < UT_ia[j + 1] && mm < UTinv_ia[k + 1]) {
          /* UT(kk,j) */
          kk = UT_ja[l];
          /* UTinv(ll,k) */
          ll = UTinv_ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* Linv(k,j)=Linv(k,j)-UTinv(kk,k)^T*UT(kk,j) */
            Linv_a[m].r -= (flag_piUT)
                               ? UTinv_a[mm].r * UT_valuesR[l] -
                                     UTinv_a[mm].i * UT_valuesI[l]
                               : UTinv_a[mm].r * UT_valuesR[l];
            Linv_a[m].i -= (flag_piUT)
                               ? UTinv_a[mm].r * UT_valuesI[l] +
                                     UTinv_a[mm].i * UT_valuesR[l]
                               : UTinv_a[mm].i * UT_valuesR[l];
            l++;
            mm++;
          }
        } /* end while */
      }   /* end for m */
      /******   END Computation UTinv(IT,j-1:j), Linv(I,j-1:j)   ************/

      /* Compute
       * Dinv(j-1:j,j-1:j)=D(j-1:j,j-1:j)^{-1}-UT(IT,j-1:j)^T*UTinv(IT,j-1:j) */
      /* extract D(j-1:j,j-1:j) */
      m = D_ia[j - 1];
#ifdef PRINT_CHECK
      if (D_ja[m] != j - 1) {
        mexPrintf("D: diagonal mismatch (%d,%d)\n", D_ja[m] + 1, j);
        fflush(stdout);
      }
#endif
#ifdef PRINT_CHECK
      if (D_ja[m + 1] != j) {
        mexPrintf("D: sub-diagonal mismatch (%d,%d)\n", D_ja[m + 1] + 1, j + 1);
        fflush(stdout);
      }
#endif
      Djm1jm1.r = D_valuesR[m];
      Djm1jm1.i = (flag_piD) ? D_valuesI[m] : 0.0;
      Djjm1.r = D_valuesR[m + 1];
      Djjm1.i = (flag_piD) ? D_valuesI[m + 1] : 0.0;

      m = D_ia[j];
#ifdef PRINT_CHECK
      if (D_ja[m] != j - 1) {
        mexPrintf("D: super-diagonal mismatch (%d,%d)\n", D_ja[m] + 1, j);
        fflush(stdout);
      }
      if (D_ja[m + 1] != j) {
        mexPrintf("D: diagonal mismatch (%d,%d)\n", D_ja[m + 1] + 1, j + 1);
        fflush(stdout);
      }
#endif
      Djm1j.r = D_valuesR[m];
      Djm1j.i = (flag_piD) ? D_valuesI[m] : 0.0;
      Djj.r = D_valuesR[m + 1];
      Djj.i = (flag_piD) ? D_valuesI[m + 1] : 0.0;
      /* determinant for 2x2 matrix inverse */
      det.r = Djm1jm1.r * Djj.r - Djm1jm1.i * Djj.i -
              (Djjm1.r * Djm1j.r - Djjm1.i * Djm1j.i);
      det.i = Djm1jm1.r * Djj.i + Djm1jm1.i * Djj.r -
              (Djjm1.r * Djm1j.i + Djjm1.i * Djm1j.r);
      nrm = 1.0 / (det.r * det.r + det.i * det.i);

      /* set Ainv(j-1:j,j-1:j) */
      mm = Dinv_ia[j - 1];
#ifdef PRINT_CHECK
      if (Dinv_ja[mm] != j - 1) {
        mexPrintf("Dinv: mismatch (%d,%d)\n", Dinv_ja[mm] + 1, j);
        fflush(stdout);
      }
      if (Dinv_ja[mm + 1] != j) {
        mexPrintf("Dinv: mismatch (%d,%d)\n", Dinv_ja[mm + 1] + 1, j + 1);
        fflush(stdout);
      }
#endif
      Dinv_a[mm].r = (Djj.r * det.r + Djj.i * det.i) * nrm;
      Dinv_a[mm].i = (-Djj.r * det.i + Djj.i * det.r) * nrm;
      Dinv_a[mm + 1].r = -(Djjm1.r * det.r + Djjm1.i * det.i) * nrm;
      Dinv_a[mm + 1].i = -(-Djjm1.r * det.i + Djjm1.i * det.r) * nrm;

      mm = Dinv_ia[j];
#ifdef PRINT_CHECK
      if (Dinv_ja[mm] != j - 1) {
        mexPrintf("Dinv: mismatch (%d,%d)\n", Dinv_ja[mm] + 1, j);
        fflush(stdout);
      }
      if (Dinv_ja[mm + 1] != j) {
        mexPrintf("Dinv: mismatch (%d,%d)\n", Dinv_ja[mm + 1] + 1, j + 1);
        fflush(stdout);
      }
#endif
      Dinv_a[mm].r = -(Djm1j.r * det.r + Djm1j.i * det.i) * nrm;
      Dinv_a[mm].i = -(-Djm1j.r * det.i + Djm1j.i * det.r) * nrm;

      Dinv_a[mm + 1].r = (Djm1jm1.r * det.r + Djm1jm1.i * det.i) * nrm;
      Dinv_a[mm + 1].i = (-Djm1jm1.r * det.i + Djm1jm1.i * det.r) * nrm;

      /* update Dinv(j-1,j-1) */
      mm = Dinv_ia[j - 1];
#ifdef PRINT_CHECK
      if (UTinv_ja[UTinv_ia[j - 1]] == j - 1 ||
          UTinv_ja[UTinv_ia[j - 1] + 1] == j) {
        mexPrintf("UTinv: diagonal or sub-diagonal index mismatch at "
                  "(%d,%d),(%d,%d)\n",
                  UTinv_ja[UTinv_ia[j - 1]] + 1, j,
                  UTinv_ja[UTinv_ia[j - 1] + 1] + 1, j);
        fflush(stdout);
      }
#endif

      l = UTinv_ia[j - 1];
#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[j - 1]] != j - 1) {
        mexPrintf("UT: diagonal index mismatch at (%d,%d)\n",
                  UT_ja[UT_ia[j - 1]] + 1, j);
        fflush(stdout);
      }
#endif
      m = UT_ia[j - 1] + 1;
#ifdef PRINT_CHECK
      if (UT_ja[m] <= j) {
        mexPrintf("UT: UT(%d,%d)=%8.1le=0???\n", UT_ja[m] + 1, j,
                  UT_valuesR[m]);
        fflush(stdout);
      }
#endif
      if (UT_ja[m] <= j)
        m++;
      while (l < UTinv_ia[j] && m < UT_ia[j]) {
        /* UT(kk,j-1) */
        kk = UT_ja[m];
        /* UTinv(ll,j-1) */
        ll = UTinv_ja[l];
        if (kk < ll)
          m++;
        else if (ll < kk)
          l++;
        else { /* indices match */
          /* Dinv(j-1,j-1)=Dinv(j-1,j-1)-UT(kk,j-1)^T*UTinv(kk,j-1) */
          Dinv_a[mm].r -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].r - UT_valuesI[m] * UTinv_a[l].i
                  : UT_valuesR[m] * UTinv_a[l].r;
          Dinv_a[mm].i -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].i + UT_valuesI[m] * UTinv_a[l].r
                  : UT_valuesR[m] * UTinv_a[l].i;
          m++;
          l++;
        }
      } /* end while */

      /* update Dinv(j,j-1) */
      mm = Dinv_ia[j - 1] + 1;
      l = UTinv_ia[j - 1];
#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[j]] != j) {
        mexPrintf("UT: diagonal index mismatch at (%d,%d)\n",
                  UT_ja[UT_ia[j]] + 1, j + 1);
        fflush(stdout);
      }
#endif
      m = UT_ia[j] + 1;
      while (l < UTinv_ia[j] && m < UT_ia[j + 1]) {
        /* UT(kk,j) */
        kk = UT_ja[m];
        /* UTinv(ll,j-1) */
        ll = UTinv_ja[l];
        if (kk < ll)
          m++;
        else if (ll < kk)
          l++;
        else { /* indices match */
          /* Dinv(j,j-1)=Dinv(j,j-1)-UT(kk,j)^T*UTinv(kk,j-1) */
          Dinv_a[mm].r -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].r - UT_valuesI[m] * UTinv_a[l].i
                  : UT_valuesR[m] * UTinv_a[l].r;
          Dinv_a[mm].i -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].i + UT_valuesI[m] * UTinv_a[l].r
                  : UT_valuesR[m] * UTinv_a[l].i;
          m++;
          l++;
        }
      } /* end while */

      /* update Dinv(j-1,j) */
      mm = Dinv_ia[j];
      l = UTinv_ia[j];
      m = UT_ia[j - 1] + 1;
      while (l < UTinv_ia[j + 1] && m < UT_ia[j]) {
        /* UT(kk,j-1) */
        kk = UT_ja[m];
        /* UTinv(ll,j) */
        ll = UTinv_ja[l];
        if (kk < ll)
          m++;
        else if (ll < kk)
          l++;
        else { /* indices match */
          /* Dinv(j-1,j)=Dinv(j-1,j)-UT(kk,j-1)^T*UTinv(kk,j) */
          Dinv_a[mm].r -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].r - UT_valuesI[m] * UTinv_a[l].i
                  : UT_valuesR[m] * UTinv_a[l].r;
          Dinv_a[mm].i -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].i + UT_valuesI[m] * UTinv_a[l].r
                  : UT_valuesR[m] * UTinv_a[l].i;
          m++;
          l++;
        }
      } /* end while */

      /* update Dinv(j,j) */
      mm = Dinv_ia[j] + 1;
      l = UTinv_ia[j];
#ifdef PRINT_CHECK
      if (UT_ja[UT_ia[j]] != j) {
        mexPrintf("UT: diagonal index mismatch at (%d,%d)\n",
                  UT_ja[UT_ia[j]] + 1, j + 1);
        fflush(stdout);
      }
#endif
      m = UT_ia[j] + 1;
      while (l < UTinv_ia[j + 1] && m < UT_ia[j + 1]) {
        /* UT(kk,j) */
        kk = UT_ja[m];
        /* UTinv(ll,j) */
        ll = UTinv_ja[l];
        if (kk < ll)
          m++;
        else if (ll < kk)
          l++;
        else { /* indices match */
          /* Dinv(j,j)=Dinv(j,j)-UT(kk,j)^T*UTinv(kk,j) */
          Dinv_a[mm].r -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].r - UT_valuesI[m] * UTinv_a[l].i
                  : UT_valuesR[m] * UTinv_a[l].r;
          Dinv_a[mm].i -=
              (flag_piUT)
                  ? UT_valuesR[m] * UTinv_a[l].i + UT_valuesI[m] * UTinv_a[l].r
                  : UT_valuesR[m] * UTinv_a[l].i;
          m++;
          l++;
        }
      } /* end while */
        /* END Computation
         * Dinv(j-1:j,j-1:j)=D(j-1:j,j-1:j)^{-1}-UT(IT,j-1:j)^T*UTinv(IT,j-1:j) */

      j = j - 2;
    } /* end if-else */
  }   /* end while i */

#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: numerical values structure inserted\n");
  fflush(stdout);
  for (i = 0; i < n; i++) {
    mexPrintf("Linv: row indices column %d\n", i + 1);
    for (j = Linv_ia[i]; j < Linv_ia[i + 1]; j++) {
      mexPrintf("%8ld", Linv_ja[j] + 1);
    }
    mexPrintf("\n");
    for (j = Linv_ia[i]; j < Linv_ia[i + 1]; j++) {
      mexPrintf("%8.1le", Linv_a[j]);
    }
    mexPrintf("\n");
    mexPrintf("Dinv: row indices column %d\n", i + 1);
    for (j = Dinv_ia[i]; j < Dinv_ia[i + 1]; j++) {
      mexPrintf("%8ld", Dinv_ja[j] + 1);
    }
    mexPrintf("\n");
    for (j = Dinv_ia[i]; j < Dinv_ia[i + 1]; j++) {
      mexPrintf("%8.1le", Dinv_a[j]);
    }
    mexPrintf("\n");
    mexPrintf("UTinv: row indices column %d\n", i + 1);
    for (j = UTinv_ia[i]; j < UTinv_ia[i + 1]; j++) {
      mexPrintf("%8ld", UTinv_ja[j] + 1);
    }
    mexPrintf("\n");
    for (j = UTinv_ia[i]; j < UTinv_ia[i + 1]; j++) {
      mexPrintf("%8.1le", UTinv_a[j]);
    }
    mexPrintf("\n");
  }
#endif

  /* export Ainv */
  nnz = UTinv_ia[n] + Dinv_ia[n] + Linv_ia[n];
  plhs[0] = mxCreateSparse((mwSize)n, (mwSize)n, (mwSize)nnz, mxCOMPLEX);
  Ainv_ja = (mwIndex *)mxGetIr(plhs[0]);
  Ainv_ia = (mwIndex *)mxGetJc(plhs[0]);
  Ainv_valuesR = (double *)mxGetPr(plhs[0]);
  Ainv_valuesI = (double *)mxGetPi(plhs[0]);

/* permute the matrix back to the original shape when exporting the matrix to
 * MATLAB */
/* 1. step: find out the number of nonzeros per column of the permuted matrix.
 */
#ifdef PRINT_CHECK
  for (m = 0; m < n; m++) {
    if (ibuff[m] != 0)
      mexPrintf("ibuff dirty at position %d\n", m + 1);
    fflush(stdout);
  }
#endif
  for (k = 0; k < n; k++) {
    i = p[k];
    /* check UTinv(:,invp(k)) */
    ibuff[k] += UTinv_ia[i + 1] - UTinv_ia[i];
    /* check Dinv(:,invp(k)) */
    ibuff[k] += Dinv_ia[i + 1] - Dinv_ia[i];
  } /* end for k */
  for (k = 0; k < n; k++) {
    i = p[k];
    /* check Linv(:,invp(k))^T */
    for (l = Linv_ia[i]; l < Linv_ia[i + 1]; l++) {
      /* Linv(invp(j),invp(k)) */
      j = invp[Linv_ja[l]];
      ibuff[j]++;
    } /* end for l */
  }   /* end for k */
#ifdef PRINT_INFO0
  mexPrintf("nz of Ainv column by column\n");
  for (m = 0; m < n; m++)
    mexPrintf("%4d", m + 1);
  mexPrintf("\n");
  fflush(stdout);
  for (m = 0; m < n; m++)
    mexPrintf("%4d", ibuff[m]);
  mexPrintf("\n");
  fflush(stdout);
#endif

  /* set up pointer structure in advance. */
  Ainv_ia[0] = 0;
  for (k = 0; k < n; k++)
    Ainv_ia[k + 1] = Ainv_ia[k] + ibuff[k];

  for (k = 0; k < n; k++) {
    i = p[k];
/* check UTinv(:,i) */
#ifdef PRINT_INFO0
    mexPrintf("Ainv(:,%d): insert column %d of UTinv\n", k + 1, i + 1);
#endif
    for (l = UTinv_ia[i]; l < UTinv_ia[i + 1]; l++) {
      j = invp[UTinv_ja[l]];
      ibuff[k]--;
      m = Ainv_ia[k] + ibuff[k];
      Ainv_ja[m] = j;
#ifdef PRINT_INFO0
      mexPrintf("%4d", j + 1);
#endif
      Ainv_valuesR[m] = UTinv_a[l].r;
      Ainv_valuesI[m] = UTinv_a[l].i;
    } /* end for l */
#ifdef PRINT_INFO0
    mexPrintf("\n");
    mexPrintf("Ainv(:,%d): insert column %d of Dinv\n", k + 1, i + 1);
#endif
    /* check Dinv(:,i) */
    for (l = Dinv_ia[i]; l < Dinv_ia[i + 1]; l++) {
      j = invp[Dinv_ja[l]];
      ibuff[k]--;
      m = Ainv_ia[k] + ibuff[k];
      Ainv_ja[m] = j;
#ifdef PRINT_INFO0
      mexPrintf("%4d", j + 1);
#endif
      Ainv_valuesR[m] = Dinv_a[l].r;
      Ainv_valuesI[m] = Dinv_a[l].i;
    } /* end for l */
#ifdef PRINT_INFO0
    mexPrintf("\n");
    mexPrintf("Ainv: insert row  %d of Linv^T\n", i + 1);
#endif
    /* check Linv(:,i)^T */
    for (l = Linv_ia[i]; l < Linv_ia[i + 1]; l++) {
      j = invp[Linv_ja[l]];
      ibuff[j]--;
      m = Ainv_ia[j] + ibuff[j];
      Ainv_ja[m] = k;
#ifdef PRINT_INFO0
      mexPrintf("(%4d,%4d) ", k + 1, j + 1);
#endif
      Ainv_valuesR[m] = Linv_a[l].r;
      Ainv_valuesI[m] = Linv_a[l].i;
    } /* end for l */
#ifdef PRINT_INFO0
    mexPrintf("\n");
#endif
  } /* end for k */
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: exported matrix reordered\n");
  fflush(stdout);
  for (i = 0; i < n; i++) {
    mexPrintf("row indices column %d\n", i + 1);
    for (j = Ainv_ia[i]; j < Ainv_ia[i + 1]; j++) {
      mexPrintf("%8ld", Ainv_ja[j] + 1);
    }
    mexPrintf("\n");
    for (j = Ainv_ia[i]; j < Ainv_ia[i + 1]; j++) {
      mexPrintf("%8.1le", Ainv_valuesR[j]);
    }
    mexPrintf("\n");
  }
#endif

  /* rescale and sort columns */
  for (k = 0; k < n; k++) {
    for (l = Ainv_ia[k]; l < Ainv_ia[k + 1]; l++) {
      m = Ainv_ja[l];
      Ainv_valuesR[l] *= Deltar_valuesR[m] * Deltal_valuesR[k];
      Ainv_valuesI[l] *= Deltar_valuesR[m] * Deltal_valuesR[k];
    } /* end for l */
    /* sort column k */
    l = Ainv_ia[k];
    m = Ainv_ia[k + 1] - Ainv_ia[k];
    Dqsortpair((doubleprecision *)Ainv_valuesR + l,
               (doubleprecision *)Ainv_valuesI + l, (integer *)Ainv_ja + l,
               (integer *)ibuff, &m);
  } /* end for k */
#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: exported matrix rescaled and indices sorted\n");
  fflush(stdout);
  for (i = 0; i < n; i++) {
    mexPrintf("row indices column %d\n", i + 1);
    for (j = Ainv_ia[i]; j < Ainv_ia[i + 1]; j++) {
      mexPrintf("%8ld", Ainv_ja[j] + 1);
    }
    mexPrintf("\n");
    for (j = Ainv_ia[i]; j < Ainv_ia[i + 1]; j++) {
      mexPrintf("%8.1le", Ainv_valuesR[j]);
    }
    mexPrintf("\n");
  }
#endif

  /* release UTinv,Dinv,Linv matrices */
  free(UTinv_ia);
  free(UTinv_ja);
  free(UTinv_a);
  free(Dinv_ia);
  free(Dinv_ja);
  free(Dinv_a);
  free(Linv_ia);
  free(Linv_ja);
  free(Linv_a);
  free(ibuff);
  /* release permutation arrays */
  free(p);
  free(invp);

#ifdef PRINT_INFO
  mexPrintf("ZGNLselinv: memory released\n");
  fflush(stdout);
#endif

  return;
}
