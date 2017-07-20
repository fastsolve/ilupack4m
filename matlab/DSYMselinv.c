/* ========================================================================== */
/* === DSYMselinv mexFunction =============================================== */
/* ========================================================================== */

/*
    Usage:

    Return the structure 'options' and incomplete LDL^T preconditioner

    Example:

    % for initializing parameters
    Linv=DSYMselinv(L,D,p, Delta)


    Authors:

        Matthias Bollhoefer, TU Braunschweig

    Date:

        March 06, 2015. ILUPACK V2.5.

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
/* #define PRINT_INFO */

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
  mxArray *L_input, *D_input, *p_input, *Delta_input;
  integer i, j, k, l, m, kk, ll, mm, n, flag, nnz, *ia, *ja, *p, *invp, ierr,
      *ibuff;
  doubleprecision *ainv, Djm1jm1, Djjm1, Djj, det;
  size_t mrows, ncols;
  double *Ainv_valuesR, *L_valuesR, *D_valuesR, *p_valuesR, *Delta_valuesR;
  mwIndex *Ainv_ja, /* row indices of output matrix Ainv     */
      *Ainv_ia,     /* column pointers of output matrix Ainv */
      *L_ja,        /* row indices of input matrix L         */
      *L_ia,        /* column pointers of input matrix L     */
      *D_ja,        /* row indices of input matrix D         */
      *D_ia;        /* column pointers of input matrix D     */

  if (nrhs != 4)
    mexErrMsgTxt("Four input arguments required.");
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
#ifdef PRINT_INFO
  mexPrintf("DSYMselinv: input parameter L imported\n");
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
#ifdef PRINT_INFO
  mexPrintf("DSYMselinv: input parameter D imported\n");
  fflush(stdout);
#endif

  /* The third input must be a integer vector */
  p_input = (mxArray *)prhs[2];
  /* get size of input matrix D */
  mrows = mxGetM(p_input);
  ncols = mxGetN(p_input);
  if (n != ncols && mrows != n) {
    mexErrMsgTxt(
        "Third argument must be a vector of same size as the first matrix.");
  }
  if (mxIsSparse(p_input)) {
    mexErrMsgTxt("Third input vector must be in dense format.");
  }
  p_valuesR = (double *)mxGetPr(p_input);

  /* memory for scaling and permutation */
  p = (integer *)MAlloc((size_t)n * sizeof(integer), "DSYMselinv");
  invp = (integer *)MAlloc((size_t)n * sizeof(integer), "DSYMselinv");
  ibuff = (integer *)MAlloc((size_t)n * sizeof(integer), "DSYMselinv");
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
  mexPrintf("DSYMselinv: input parameter p imported\n");
  fflush(stdout);
#endif

  /* The fourth input must be a  vector */
  Delta_input = (mxArray *)prhs[3];
  /* get size of input matrix Delta */
  mrows = mxGetM(Delta_input);
  ncols = mxGetN(Delta_input);
  if (n != ncols && mrows != n) {
    mexErrMsgTxt(
        "Fourth argument must be a vector of same size as the first matrix.");
  }
  if (mxIsSparse(Delta_input)) {
    mexErrMsgTxt("Fourth input vector must be in dense format.");
  }
  Delta_valuesR = (double *)mxGetPr(Delta_input);
#ifdef PRINT_INFO
  mexPrintf("DSYMselinv: input parameter Delta imported\n");
  fflush(stdout);
#endif

#ifdef PRINT_INFO
  mexPrintf("DSYMselinv: input parameters imported\n");
  fflush(stdout);
#endif

  /* set up preliminary structure for every column of ainv */
  /* create memory for output matrix Ainv, here: ia */
  ia = (integer *)MAlloc((size_t)(n + 1) * sizeof(integer), "DSYMselinv");
  i = 0;
  ia[0] = 0;
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
      kk = ia[i];
      /* diagonal entry */
      kk++;
/* copy sub-diagonal indices of L(:,i) */
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
      ia[i + 1] = kk;

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

      kk = ia[i];
      /* diagonal entry */
      kk++;
      /* sub-diagonal entry */
      kk++;
#ifdef PRINT_CHECK
      if (L_ja[L_ia[i]] != i) {
        mexPrintf("L: diagonal entry missmatch at entry %d\n", i + 1);
        fflush(stdout);
      }
#endif
      /* copy sub-sub-diagonal indices of L(:,i) */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
#ifdef PRINT_CHECK
        if (k <= i && m > L_ia[i]) {
          mexPrintf("L: sub-sub-diagonal entry missmatch at entry %d\n", i + 1);
          fflush(stdout);
        }
#endif
        if (k > i + 1) {
          kk++;
          ibuff[k] = 1;
        }
      } /* end for i */
      ia[i + 1] = kk;

      /* diagonal entry */
      kk++;
#ifdef PRINT_CHECK
      if (L_ja[L_ia[i + 1]] != i + 1) {
        mexPrintf("L: diagonal entry missmatch at entry %d\n", i + 2);
        fflush(stdout);
      }
#endif
      /* copy sub-diagonal indices of L(:,i+1) */
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
      ia[i + 2] = kk;

      /* clear ibuff, column i */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
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
  mexPrintf("DSYMselinv: additional memory for blocked columns: %d\n", nnz);
  fflush(stdout);
#endif
  nnz += ia[n];

  ja = (integer *)MAlloc((size_t)nnz * sizeof(integer), "DSYMselinv:ja");
  ainv = (double *)MAlloc((size_t)nnz * sizeof(double), "DSYMselinv:ainv");
  for (i = 0; i < nnz; i++) {
    ainv[i] = 0.0;
  } /* end for i */
  /* set up completed structure for every column of ainv */
  i = 0; /* column counter */
  ia[0] = 0;
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
      kk = ia[i];
      /* diagonal entry */
      ja[kk++] = i;
      /* copy sub-diagonal indices of L(:,i) */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
        if (k > i)
          ja[kk++] = k;
      } /* end for i */
      ia[i + 1] = kk;

      i++;
    } else { /* 2x2 case */
      kk = ia[i];
      /* diagonal entry */
      ja[kk++] = i;
      /* sub-diagonal entry */
      ja[kk++] = i + 1;
      /* copy sub-sub-diagonal indices of L(:,i) */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
        if (k > i + 1) {
          ja[kk++] = k;
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
            ja[kk++] = k;
            flag |= 1;
          }
          /* use negative check mark to distinguish between those
             indices that have only been visited by column i and those ones
             that are visited by column i+1 */
          ibuff[k] = -1;
        }
      } /* end for i */
      ia[i + 1] = kk;

      /* diagonal entry  column i+1 */
      ja[kk++] = i + 1;
      /* copy sub-diagonal indices of L(:,i+1) */
      for (m = L_ia[i + 1]; m < L_ia[i + 2]; m++) {
        k = L_ja[m];
        if (k > i + 1) {
          ja[kk++] = k;
        }
      } /* end for i */
      /* clear ibuff w.r.t. column i and add fill-in to column i+1 */
      for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
        k = L_ja[m];
        if (k > i + 1) {
          /* did only column i visit this index? */
          if (ibuff[k] > 0) {
            /* column i+1 will have another fill-in */
            ja[kk++] = k;
            flag |= 2;
          }
          /* clear buff */
          ibuff[k] = 0;
        }
      } /* end for i */
      ia[i + 2] = kk;

      /* clear ibuff, column i+1 */
      for (m = L_ia[i + 1]; m < L_ia[i + 2]; m++) {
        k = L_ja[m];
        if (k > i + 1)
          ibuff[k] = 0;
      } /* end for i */

      /* sort column i */
      m = ia[i];
      l = ia[i + 1] - m;
      if (flag & 1) {
        qqsorti(ja + m, ibuff, &l);
        while (l)
          ibuff[--l] = 0;
      }
      /* sort column i */
      m = ia[i + 1];
      l = ia[i + 2] - m;
      if (flag & 2) {
        qqsorti(ja + m, ibuff, &l);
        while (l)
          ibuff[--l] = 0;
      }
      flag = 0;

      i += 2;
    }
  } /* end while i */
#ifdef PRINT_INFO
  mexPrintf("DSYMselinv: row index structure inserted\n");
  fflush(stdout);
  for (i = 0; i < n; i++) {
    mexPrintf("row indices column %d\n", i + 1);
    for (j = ia[i]; j < ia[i + 1]; j++) {
      mexPrintf("%8ld", ja[j] + 1);
    }
    mexPrintf("\n");
    /*
    for (j=ia[i]; j<ia[i+1]; j++) {
        mexPrintf("%8.1le", ainv[j]);
    }
    mexPrintf("\n");
    */
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
    mm = ia[j];
    ainv[mm] = 1.0 / D_valuesR[m];

    j--;
  } else {
    /* mexPrintf("j=%d:%d\n",j,j+1);fflush(stdout); */
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
    Djm1jm1 = D_valuesR[m];
    Djjm1 = D_valuesR[m + 1];
#ifdef PRINT_CHECK
    if (D_ja[D_ia[j]] != j - 1) {
      mexPrintf("D: super-diagonal mismatch (%d,%d)\n", D_ja[D_ia[j]] + 1, j);
      fflush(stdout);
    }
#endif
    m = D_ia[j] + 1;
#ifdef PRINT_CHECK
    if (D_ja[m] != j) {
      mexPrintf("D: diagonal mismatch (%d,%d)\n", D_ja[m] + 1, j + 1);
      fflush(stdout);
    }
#endif
    Djj = D_valuesR[m];
    /* determinant for 2x2 matrix inverse */
    det = 1.0 / (Djm1jm1 * Djj - Djjm1 * Djjm1);
    /* set Ainv(j-1:j,j-1:j) */
    mm = ia[j - 1];
#ifdef PRINT_CHECK
    if (ja[mm] != j - 1) {
      mexPrintf("ainv: mismatch (%d,%d)\n", ja[mm] + 1, j);
      fflush(stdout);
    }
#endif
#ifdef PRINT_CHECK
    if (ja[mm + 1] != j) {
      mexPrintf("ainv: mismatch (%d,%d)\n", ja[mm + 1] + 1, j + 1);
      fflush(stdout);
    }
#endif
    ainv[mm] = Djj * det;
    ainv[mm + 1] = -Djjm1 * det;
    mm = ia[j];
#ifdef PRINT_CHECK
    if (ja[mm] != j) {
      mexPrintf("ainv: mismatch (%d,%d)\n", ja[mm] + 1, j + 1);
      fflush(stdout);
    }
#endif
    ainv[mm] = Djm1jm1 * det;

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

/************ Compute  Ainv(I,j)=-Ainv(I,I)*L(I,j) ******************/
/* recall that the entries of ainv are already initialized with 0.0 */
/* starting position in L(j+1:n,j)*/
#ifdef PRINT_CHECK
      if (L_ja[L_ia[j]] != j) {
        mexPrintf("L: mismatch (%d,%d)\n", L_ja[L_ia[j]] + 1, j + 1);
        fflush(stdout);
      }
#endif
      for (m = L_ia[j] + 1; m < L_ia[j + 1]; m++) {
        /* column index k>j of L(k,j) */
        k = L_ja[m];
/* mexPrintf("(%d,%d)\n",k+1,j+1);fflush(stdout); */
/* update Ainv(I,j) with -Ainv(I,k)*L(k,j) */
/* 1. part: lower triangular and diagonal part of Ainv(I,k) */
/* starting positions column of Ainv(j+1:n,j) and k of Ainv(k:n,k) */
#ifdef PRINT_CHECK
        if (ja[ia[j]] != j) {
          mexPrintf("Ainv: mismatch (%d,%d)\n", ja[ia[j]] + 1, j + 1);
          fflush(stdout);
        }
#endif
        l = ia[j] + 1;
        mm = ia[k];
        flag = 0;
        while (l < ia[j + 1] && mm < ia[k + 1]) {
          /* Ainv(kk,j) */
          kk = ja[l];
          /* Ainv(ll,k) */
          ll = ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* store the location l of Ainv(k,j) and avoid duplicate update */
            if (kk == k)
              flag = l;
            else {
              /* Ainv(kk,j)=Ainv(kk,j)-Ainv(kk,k)*L(k,j) */
              ainv[l] -= ainv[mm] * L_valuesR[m];
              /* mexPrintf("[%d,%d,%d]\n",kk+1,k+1,j+1);fflush(stdout);
                 mexPrintf("{%8.1le,%8.1le,%8.1le}\n",ainv[l],ainv[mm],L_valuesR[m]);fflush(stdout);
              */
            }
            l++;
            mm++;
          }
        } /* end while */
#ifdef PRINT_CHECK
        if (!flag) {
          mexPrintf("Ainv: row index %d is missing in column %d\n", k + 1,
                    j + 1);
          fflush(stdout);
        }
#endif
/* 2. part: strict upper triangular of Ainv(I,k), which is not
   stored but contributes to Ainv(k,j). Here use Ainv(I,k)^T */
#ifdef PRINT_CHECK
        if (L_ja[L_ia[j]] != j) {
          mexPrintf("L: mismatch (%d,%d)\n", L_ja[L_ia[j]] + 1, j + 1);
          fflush(stdout);
        }
#endif
        l = L_ia[j] + 1;
        mm = ia[k];
        while (l < L_ia[j + 1] && mm < ia[k + 1]) {
          /* L(kk,j) */
          kk = L_ja[l];
          /* Ainv(ll,k) */
          ll = ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* Ainv(k,j)=Ainv(k,j)-Ainv(kk,k)^T*L(kk,j) */
            ainv[flag] -= ainv[mm] * L_valuesR[l];
            /* mexPrintf("[%d,%d,%d]\n",ja[flag]+1,kk+1,j+1);fflush(stdout);
               mexPrintf("{%8.1le,%8.1le,%8.1le}\n",ainv[flag],ainv[mm],L_valuesR[l]);fflush(stdout);
            */
            l++;
            mm++;
          }
        } /* end while */
      }   /* end for m */
          /******** END Computation  Ainv(I,j)=-Ainv(I,I)*L(I,j) **************/

      /*********** Compute Ainv(j,j)=1/D(j,j)-L(I,j)^T*Ainv(I,j) ***********/
      mm = ia[j];
      m = D_ia[j];
      ainv[mm] = 1.0 / D_valuesR[m];
#ifdef PRINT_CHECK
      if (ja[mm] != D_ja[m]) {
        mexPrintf("Ainv,D: index mismatch at (%d,%d)\n", ja[mm] + 1,
                  D_ja[m] + 1);
        fflush(stdout);
      }
#endif
#ifdef PRINT_CHECK
      if (L_ja[L_ia[j]] != j) {
        mexPrintf("L: mismatch (%d,%d)\n", L_ja[L_ia[j]] + 1, j + 1);
        fflush(stdout);
      }
#endif
#ifdef PRINT_CHECK
      if (ja[ia[j]] != j) {
        mexPrintf("Ainv: mismatch (%d,%d)\n", ja[ia[j]] + 1, j + 1);
        fflush(stdout);
      }
#endif
      l = ia[j] + 1;
      m = L_ia[j] + 1;
      while (l < ia[j + 1] && m < L_ia[j + 1]) {
        /* L(kk,j) */
        kk = L_ja[m];
        /* Ainv(ll,j) */
        ll = ja[l];
        if (kk < ll)
          m++;
        else if (ll < kk)
          l++;
        else { /* indices match */
          /* Ainv(j,j)=Ainv(j,j)-L(kk,j)^T*Ainv(kk,j) */
          ainv[mm] -= L_valuesR[m] * ainv[l];
          m++;
          l++;
        }
      } /* end while */
      /******* END Computation Ainv(j,j)=1/D(j,j)-L(I,j)^T*Ainv(I,j) *******/

      j = j - 1;
    } else { /* 2x2 case */
             /* mexPrintf("j=%d:%d\n",j,j+1);fflush(stdout); */
/******** Compute  Ainv(I,j-1:j)=-Ainv(I,I)*L(I,j-1:j) **************/
/* recall that the entries of ainv are already initialized with 0.0 */
/* starting position in column L(j+1:n,j-1) */
#ifdef PRINT_CHECK
      if (L_ja[L_ia[j - 1]] != j - 1) {
        mexPrintf("L: diagonal index mismatch at (%d,%d)\n",
                  L_ja[L_ia[j - 1]] + 1, j);
        fflush(stdout);
      }
#endif
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
/* update Ainv(I,j-1) with -Ainv(I,k)*L(k,j-1) */
/* 1. part: lower triangular and diagonal part of Ainv(I,k) */
/* note that the indices of Ainv(:,j-1) should start with j-1 and j */
#ifdef PRINT_CHECK
        if (ja[ia[j - 1]] != j - 1 || ja[ia[j - 1] + 1] != j) {
          mexPrintf("Ainv: diagonal or sub-diagonal index mismatch at "
                    "(%d,%d),(%d,%d)\n",
                    ja[ia[j - 1]] + 1, j, ja[ia[j - 1] + 1] + 1, j);
          fflush(stdout);
        }
#endif
        l = ia[j - 1] + 2;
        mm = ia[k];
        flag = 0;
        while (l < ia[j] && mm < ia[k + 1]) {
          /* Ainv(kk,j-1) */
          kk = ja[l];
          /* Ainv(ll,k) */
          ll = ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* store the location l of Ainv(k,j) and avoid duplicate update */
            if (kk == k)
              flag = l;
            else {
              /* Ainv(kk,j-1)=Ainv(kk,j-1)-Ainv(kk,k)*L(k,j-1) */
              ainv[l] -= ainv[mm] * L_valuesR[m];
            }
            l++;
            mm++;
          }
        } /* end while */
          /* 2. part: strict upper triangular of Ainv(I,k), which is not
             stored but contributes to Ainv(k,j-1). Here use Ainv(I,k)^T */
#ifdef PRINT_CHECK
        if (L_ja[L_ia[j - 1]] != j - 1) {
          mexPrintf("L: diagonal index mismatch at (%d,%d)\n",
                    L_ja[L_ia[j - 1]] + 1, j);
          fflush(stdout);
        }
#endif
        l = L_ia[j - 1] + 1;
/* skip L(j,j-1) if present (which should be zero) */
#ifdef PRINT_CHECK
        if (L_ja[l] <= j) {
          mexPrintf("L: L(%d,%d)=%8.1le=0??\n", L_ja[l] + 1, j, L_valuesR[l]);
          fflush(stdout);
        }
#endif
        if (L_ja[l] <= j)
          l++;
        mm = ia[k];
        while (l < L_ia[j] && mm < ia[k + 1]) {
          /* L(kk,j-1) */
          kk = L_ja[l];
          /* Ainv(ll,k) */
          ll = ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* Ainv(k,j-1)=Ainv(k,j-1)-Ainv(kk,k)^T*L(kk,j-1) */
            ainv[flag] -= ainv[mm] * L_valuesR[l];
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
      for (m = L_ia[j] + 1; m < L_ia[j + 1]; m++) {
        /* column index k of L(k,j) */
        k = L_ja[m];
/* update Ainv(I,j) with -Ainv(I,k)*L(k,j) */
/* 1. part: lower triangular and diagonal part of Ainv(I,k) */
/* note that the indices of Ainv(:,j) should start with j */
#ifdef PRINT_CHECK
        if (ja[ia[j]] != j) {
          mexPrintf("Ainv: diagonal index mismatch at (%d,%d)\n", ja[ia[j]] + 1,
                    j + 1);
          fflush(stdout);
        }
#endif
        l = ia[j] + 1;
        mm = ia[k];
        flag = 0;
        while (l < ia[j + 1] && mm < ia[k + 1]) {
          /* Ainv(kk,j) */
          kk = ja[l];
          /* Ainv(ll,k) */
          ll = ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
            /* store the location l of Ainv(k,j) and avoid duplicate update */
            if (kk == k)
              flag = l;
            else {
              /* Ainv(kk,j)=Ainv(kk,j)-Ainv(kk,k)*L(k,j) */
              ainv[l] -= ainv[mm] * L_valuesR[m];
            }
            l++;
            mm++;
          }
        } /* end while */
          /* 2. part: strict upper triangular of Ainv(I,k), which is not
             stored but contributes to Ainv(k,j). Here use Ainv(I,k)^T */
#ifdef PRINT_CHECK
        if (L_ja[L_ia[j]] != j) {
          mexPrintf("L: diagonal index mismatch at (%d,%d)\n",
                    L_ja[L_ia[j]] + 1, j + 1);
          fflush(stdout);
        }
#endif
        l = L_ia[j] + 1;
        mm = ia[k];
        while (l < L_ia[j + 1] && mm < ia[k + 1]) {
          /* L(kk,j) */
          kk = L_ja[l];
          /* Ainv(ll,k) */
          ll = ja[mm];
          if (kk < ll)
            l++;
          else if (ll < kk)
            mm++;
          else { /* indices match */
                 /* Ainv(k,j)=Ainv(k,j)-Ainv(kk,k)^T*L(kk,j) */
            ainv[flag] -= ainv[mm] * L_valuesR[l];
            l++;
            mm++;
          }
        } /* end while */
      }   /* end for m */
          /******** END Computation  Ainv(I,j)=-Ainv(I,I)*L(I,j) **************/

      /* Compute
       * Ainv(j-1:j,j-1:j)=D(j-1:j,j-1:j)^{-1}-L(I,j-1:j)^T*Ainv(I,j-1:j) */
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
      Djm1jm1 = D_valuesR[m];
      Djjm1 = D_valuesR[m + 1];
#ifdef PRINT_CHECK
      if (D_ja[D_ia[j]] != j - 1) {
        mexPrintf("D: super-diagonal mismatch (%d,%d)\n", D_ja[D_ia[j]] + 1, j);
        fflush(stdout);
      }
#endif
      m = D_ia[j] + 1;
#ifdef PRINT_CHECK
      if (D_ja[m] != j) {
        mexPrintf("D: diagonal mismatch (%d,%d)\n", D_ja[m] + 1, j + 1);
        fflush(stdout);
      }
#endif
      Djj = D_valuesR[m];
      /* determinant for 2x2 matrix inverse */
      det = 1.0 / (Djm1jm1 * Djj - Djjm1 * Djjm1);
      /* set Ainv(j-1:j,j-1:j) */
      mm = ia[j - 1];
#ifdef PRINT_CHECK
      if (ja[mm] != j - 1) {
        mexPrintf("ainv: mismatch (%d,%d)\n", ja[mm] + 1, j);
        fflush(stdout);
      }
#endif
#ifdef PRINT_CHECK
      if (ja[mm + 1] != j) {
        mexPrintf("ainv: mismatch (%d,%d)\n", ja[mm + 1] + 1, j + 1);
        fflush(stdout);
      }
#endif
      ainv[mm] = Djj * det;
      ainv[mm + 1] = -Djjm1 * det;
      mm = ia[j];
#ifdef PRINT_CHECK
      if (ja[mm] != j) {
        mexPrintf("ainv: mismatch (%d,%d)\n", ja[mm] + 1, j + 1);
        fflush(stdout);
      }
#endif
      ainv[mm] = Djm1jm1 * det;

      /* update Ainv(j-1,j-1) */
      /* note that the indices of Ainv(:,j-1) should start with j-1 and j */
      mm = ia[j - 1];
#ifdef PRINT_CHECK
      if (ja[ia[j - 1]] != j - 1 || ja[ia[j - 1] + 1] != j) {
        mexPrintf("Ainv: diagonal or sub-diagonal index mismatch at "
                  "(%d,%d),(%d,%d)\n",
                  ja[ia[j - 1]] + 1, j, ja[ia[j - 1] + 1] + 1, j);
        fflush(stdout);
      }
#endif
      l = ia[j - 1] + 2;
#ifdef PRINT_CHECK
      if (L_ja[L_ia[j - 1]] != j - 1) {
        mexPrintf("L: diagonal index mismatch at (%d,%d)\n",
                  L_ja[L_ia[j - 1]] + 1, j);
        fflush(stdout);
      }
#endif
      m = L_ia[j - 1] + 1;
#ifdef PRINT_CHECK
      if (L_ja[m] <= j) {
        mexPrintf("L: L(%d,%d)=%8.1le=0???\n", L_ja[m] + 1, j, L_valuesR[m]);
        fflush(stdout);
      }
#endif
      if (L_ja[m] <= j)
        m++;
      while (l < ia[j] && m < L_ia[j]) {
        /* L(kk,j-1) */
        kk = L_ja[m];
        /* Ainv(ll,j-1) */
        ll = ja[l];
        if (kk < ll)
          m++;
        else if (ll < kk)
          l++;
        else { /* indices match */
          /* Ainv(j-1,j-1)=Ainv(j-1,j-1)-L(kk,j-1)^T*Ainv(kk,j-1) */
          ainv[mm] -= L_valuesR[m] * ainv[l];
          m++;
          l++;
        }
      } /* end while */

      /* update Ainv(j,j-1) */
      mm = ia[j - 1] + 1;
#ifdef PRINT_CHECK
      if (ja[ia[j - 1]] != j - 1 || ja[ia[j - 1] + 1] != j) {
        mexPrintf("Ainv: diagonal or sub-diagonal index mismatch at "
                  "(%d,%d),(%d,%d)\n",
                  ja[ia[j - 1]] + 1, j, ja[ia[j - 1] + 1] + 1, j);
        fflush(stdout);
      }
#endif
      l = ia[j - 1] + 2;
#ifdef PRINT_CHECK
      if (L_ja[L_ia[j]] != j) {
        mexPrintf("L: diagonal index mismatch at (%d,%d)\n", L_ja[L_ia[j]] + 1,
                  j + 1);
        fflush(stdout);
      }
#endif
      m = L_ia[j] + 1;
      while (l < ia[j] && m < L_ia[j + 1]) {
        /* L(kk,j) */
        kk = L_ja[m];
        /* Ainv(ll,j-1) */
        ll = ja[l];
        if (kk < ll)
          m++;
        else if (ll < kk)
          l++;
        else { /* indices match */
          /* Ainv(j,j-1)=Ainv(j,j-1)-L(kk,j)^T*Ainv(kk,j-1) */
          ainv[mm] -= L_valuesR[m] * ainv[l];
          m++;
          l++;
        }
      } /* end while */

      /* update Ainv(j,j) */
      mm = ia[j];
#ifdef PRINT_CHECK
      if (ja[ia[j]] != j) {
        mexPrintf("Ainv: diagonal index mismatch at (%d,%d)\n", ja[ia[j]] + 1,
                  j);
        fflush(stdout);
      }
#endif
      l = ia[j] + 1;
#ifdef PRINT_CHECK
      if (L_ja[L_ia[j]] != j) {
        mexPrintf("L: diagonal index mismatch at (%d,%d)\n", L_ja[L_ia[j]] + 1,
                  j + 1);
        fflush(stdout);
      }
#endif
      m = L_ia[j] + 1;
      while (l < ia[j + 1] && m < L_ia[j + 1]) {
        /* L(kk,j) */
        kk = L_ja[m];
        /* Ainv(ll,j) */
        ll = ja[l];
        if (kk < ll)
          m++;
        else if (ll < kk)
          l++;
        else { /* indices match */
          /* Ainv(j,j)=Ainv(j,j)-L(kk,j)^T*Ainv(kk,j) */
          ainv[mm] -= L_valuesR[m] * ainv[l];
          m++;
          l++;
        }
      } /* end while */
        /* END Computation
         * Ainv(j-1:j,j-1:j)=D(j,j)^{-1}-L(I,j-1:j)^T*Ainv(I,j-1:j) */

      j = j - 2;
    } /* end if-else */
  }   /* end while i */
#ifdef PRINT_INFO
  mexPrintf("DSYMselinv: numerical values structure inserted\n");
  fflush(stdout);
  for (i = 0; i < n; i++) {
    mexPrintf("row indices column %d\n", i + 1);
    for (j = ia[i]; j < ia[i + 1]; j++) {
      mexPrintf("%8ld", ja[j] + 1);
    }
    mexPrintf("\n");
    for (j = ia[i]; j < ia[i + 1]; j++) {
      mexPrintf("%8.1le", ainv[j]);
    }
    mexPrintf("\n");
  }
#endif

  /* export Ainv */
  plhs[0] = mxCreateSparse((mwSize)n, (mwSize)n, (mwSize)nnz, mxREAL);
  Ainv_ja = (mwIndex *)mxGetIr(plhs[0]);
  Ainv_ia = (mwIndex *)mxGetJc(plhs[0]);
  Ainv_valuesR = (double *)mxGetPr(plhs[0]);

  /* permute the matrix back to the original shape when exporting the matrix to
   * MATLAB */
  /* 1. step: find out the number of nonzeros per column of the permuted matrix.
     To do so,
              we count the number of nonzeros in Ainv and check, which ones are
     in the lower
              triangular part after permutation and which would move to the
     strict upper
              triangular part. The latter contribute to the associated row
     rather than the
              column
  */
  for (k = 0; k < n; k++) {
    i = invp[k];
    /* check Ainv(:,i) */
    for (l = ia[i]; l < ia[i + 1]; l++) {
      j = p[ja[l]];
      /* this entry will stay in the lower triangular part of column k=p(i) */
      if (j >= k)
        ibuff[k]++;
      else /* this entry will transfer to the upper triangular part of column
              k=p(i), thus
              it will contribute to the strict lower triangular part of column j
              instead
           */
        ibuff[j]++;
    } /* end for l */
  }   /* end for k */

  /* set up pointer structure in advance. This includes the entries which move
     from one
     column to another one because of switching to the upper triangular part */
  Ainv_ia[0] = 0;
  for (k = 0; k < n; k++)
    Ainv_ia[k + 1] = Ainv_ia[k] + ibuff[k];

  for (k = 0; k < n; k++) {
    i = invp[k];
    /* check Ainv(:,i) */
    for (l = ia[i]; l < ia[i + 1]; l++) {
      j = p[ja[l]];
      /* this entry will stay in the lower triangular part of column k=p(i) */
      if (j >= k) {
        ibuff[k]--;
        m = Ainv_ia[k] + ibuff[k];
        Ainv_ja[m] = j;
        Ainv_valuesR[m] = ainv[l];
      } else { /* this entry will transfer to the upper triangular part of
                column k=p(i), thus
                it will contribute to the strict lower triangular part of column
                j instead
             */
        ibuff[j]--;
        m = Ainv_ia[j] + ibuff[j];
        Ainv_ja[m] = k;
        Ainv_valuesR[m] = ainv[l];
      }
    } /* end for l */
  }   /* end for k */
#ifdef PRINT_INFO
  mexPrintf("DSYMselinv: exported matrix reordered\n");
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
      Ainv_valuesR[l] *= Delta_valuesR[m] * Delta_valuesR[k];
    } /* end for l */
    /* sort column k */
    l = Ainv_ia[k];
    m = Ainv_ia[k + 1] - Ainv_ia[k];
    Dqsort((doubleprecision *)Ainv_valuesR + l, (integer *)Ainv_ja + l,
           (integer *)ibuff, &m);
  } /* end for k */
#ifdef PRINT_INFO
  mexPrintf("DSYMselinv: exported matrix rescaled and indices sorted\n");
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

  /* release ainv matrix */
  free(ia);
  free(ja);
  free(ainv);
  free(ibuff);
  /* release permutation arrays */
  free(p);
  free(invp);

#ifdef PRINT_INFO
  mexPrintf("DSYMselinv: memory released\n");
  fflush(stdout);
#endif

  return;
}
