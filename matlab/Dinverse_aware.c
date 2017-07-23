/* $Id: Dinverse_aware.c 797 2015-08-07 20:44:18Z bolle $ */
/* ========================================================================== */
/* === Dinverse_aware mexFunction =========================================== */
/* ========================================================================== */

/*
    Usage:

    Return unit lower triangular LL with a few artificial nonzeros

    Example:

    % for initializing parameters
    LL=Dinverse_aware(L,D,ndense)


    Authors:

        Matthias Bollhoefer, TU Braunschweig

    Date:

        August 05, 2015. ILUPACK V2.5.

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
#include <stdlib.h>
#include <string.h>
#define _DOUBLE_REAL_
#include <ilupackmacros.h>
#define ELBOW 5
#define RM -2.2251e-308
/* #define SORT_ENTRIES */

#define MAX_FIELDS 100
#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define MIN(A, B) (((A) >= (B)) ? (B) : (A))
/* #define PRINT_CHECK  */
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
  mxArray *L_input, *D_input, *ndense_input, *LL_output;
  integer i, j, k, l, m, n, nz, p, q, r, s, ndense, *ibuff, **LL, *LLi, *LLj,
      *nLL, *eLL;
  doubleprecision *pr, *dbuff;
  size_t mrows, ncols;
  double *L_valuesR, *LL_valuesR;
  mwIndex *L_ja, /* row indices of input matrix L         */
      *L_ia,     /* column pointers of input matrix L     */
      *LL_ja,    /* row indices of output matrix LL       */
      *LL_ia,    /* column pointers of output matrix LL   */
      *D_ja,     /* row indices of input matrix D         */
      *D_ia;     /* column pointers of input matrix D     */

  if (nrhs != 3)
    mexErrMsgTxt("Three input arguments are required.");
  else if (nlhs != 1)
    mexErrMsgTxt("wrong number of output arguments.");
  else if (!mxIsNumeric(prhs[0]))
    mexErrMsgTxt("First input must be a matrix.");
  else if (!mxIsNumeric(prhs[1]))
    mexErrMsgTxt("Second input must be a matrix.");
  else if (!mxIsNumeric(prhs[2]))
    mexErrMsgTxt("Third input must be a number.");

  /* The first input must be a square sparse unit lower triangular matrix.*/
  L_input = (mxArray *)prhs[0];
  /* get size of input matrix L */
  mrows = mxGetM(L_input);
  ncols = mxGetN(L_input);
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
  mexPrintf("Dinverse_aware: input parameter L imported\n");
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
#ifdef PRINT_INFO
  mexPrintf("DSYMldl2bldl: input parameter D imported\n");
  fflush(stdout);
#endif

  /* The third input must be a number */
  ndense_input = (mxArray *)prhs[2];
  /* get size of input ndense */
  mrows = mxGetM(ndense_input);
  ncols = mxGetN(ndense_input);
  if (mrows != 1 || ncols != 1) {
    mexErrMsgTxt("Third input must be a scalar.");
  }
  /* starting block with dense columns, convert to C-style */
  ndense = *mxGetPr(ndense_input) - 1;
#ifdef PRINT_INFO
  mexPrintf("Dinverse_aware: input parameter ndense imported\n");
  fflush(stdout);
#endif

  /* pointer to list of nonzero entries, column-by-column */
  LL = (integer **)MAlloc((size_t)n * sizeof(integer *), "Dinverse_aware:LL");
  nLL = (integer *)MAlloc((size_t)n * sizeof(integer), "Dinverse_aware:nLL");
  /* elbow buffer space */
  eLL = (integer *)MAlloc((size_t)n * sizeof(integer), "Dinverse_aware:nLL");
  /* integer buffer of size n */
  ibuff =
      (integer *)MAlloc((size_t)n * sizeof(integer), "Dinverse_aware:ibuff");
  /* initially use pattern of L */
  for (i = 0; i < n; i++) {
    /* start of column i, C-style */
    k = L_ia[i];
    /* number of nonzeros in L(:,i) */
    nLL[i] = L_ia[i + 1] - k;
    /* allocate slightly more memory to reduce the number of physical
     * reallocations */
    eLL[i] = MAX(nLL[i] + ELBOW, nLL[i] * 1.1);
    /* provide memory for nonzeros of column i */
    LL[i] = (integer *)MAlloc((size_t)eLL[i] * sizeof(integer),
                              "Dinverse_aware:LL[i]");
    /* copy indices */
    memcpy(LL[i], L_ja + k, (size_t)nLL[i] * sizeof(integer));
#ifdef SORT_ENTRIES
    /* sort indices of in increasing order */
    qqsorti(LL[i], ibuff, nLL + i);
#endif
  } /* end for k */
#ifdef PRINT_INFO
  for (i = 0; i < n; i++) {
    mexPrintf("col %3d:", i + 1);
    for (k = 0; k < nLL[i]; k++)
      mexPrintf("%4d", LL[i][k] + 1);
    mexPrintf("\n");
    fflush(stdout);
  }
#endif
  /* compression by advancing dense columns */
  i = ndense - 1;
  while (i >= 0) {
    if (nLL[i] == n - i) {
      ndense--;
      i--;
    } else
      i = -1;
  } /* end while */
#ifdef PRINT_INFO
  mexPrintf("ndense=%d\n", ndense + 1);
  fflush(stdout);
#endif

  /* scan pattern and augment it */
  i = 0;
  while (i < n) {
    /* 2x2 case */
    if (D_ia[i + 1] - D_ia[i] > 1) {
      /* initially make sure that column i and i+1 share the same nonzero
       * pattern */
      /* L(i:i+1,i:i+1)=I must be fulfilled! */
      r = 1; /* skip diagonal entry i   */
      s = 1; /* skip diagonal entry i+1 */
      m = 0; /* counter for the buffer */
      /* reference to the nonzero indices of column i */
      LLi = LL[i];
      /* safeguard the case that L(i+1,i)~=0, which must not be the case! */
      if (r < nLL[i])
        if (LLi[r] == i + 1)
          r = 2;
      /* reference to the nonzero indices of column i+1 */
      LLj = LL[i + 1];
      /* scan nonzero patterns of column i and i+1 exluding i,i+1 */
      while (r < nLL[i] && s < nLL[i + 1]) {
        p = LLi[r];
        q = LLj[s];
        if (p < q) {
          /* copy i's index */
          ibuff[m++] = p;
          r++;
        } else if (q < p) {
          /* copy i+1's index */
          ibuff[m++] = q;
          s++;
        } else {
          /* copy joint index */
          ibuff[m++] = q;
          r++;
          s++;
        }
      } /* end while */
      while (r < nLL[i]) {
        /* copy i-index */
        ibuff[m++] = LLi[r];
        r++;
      } /* end while */
      while (s < nLL[i + 1]) {
        /* copy i+1 index */
        ibuff[m++] = LLj[s];
        s++;
      } /* end while */
      /* safeguard the case that L(i+1,i)~=0, which must not be the case! */
      r = 1;
      if (r < nLL[i])
        if (LLi[r] == i + 1)
          r = 2;
      if (m + r > nLL[i]) {
        /* increase memory if necessary */
        eLL[i] = MAX(eLL[i], m + r);
        LL[i] = (integer *)ReAlloc(LL[i], (size_t)eLL[i] * sizeof(integer),
                                   "Dinverse_aware:LL[i]");
        /* new number of row indices in columns i */
        nLL[i] = m + r;
        /* copy merged index array back */
        memcpy(LL[i] + r, ibuff, (size_t)m * sizeof(integer));
      }
      if (m + 1 > nLL[i + 1]) {
        /* increase memory if necessary */
        eLL[i + 1] = MAX(eLL[i + 1], m + 1);
        LL[i + 1] =
            (integer *)ReAlloc(LL[i + 1], (size_t)eLL[i + 1] * sizeof(integer),
                               "Dinverse_aware:LL[i+1]");
        /* new number of row indices in columns i+1 */
        nLL[i + 1] = m + 1;
        /* copy merged index array back */
        memcpy(LL[i + 1] + 1, ibuff, (size_t)m * sizeof(integer));
      }
    } /* end 2x2 case */

#ifdef PRINT_INFO
    if (D_ia[i + 1] - D_ia[i] > 1) {
      mexPrintf("augmented patterns\n");
      fflush(stdout);
      mexPrintf("col %3d:", i + 1);
      for (k = 0; k < nLL[i]; k++)
        mexPrintf("%4d", LL[i][k] + 1);
      mexPrintf("\n");
      fflush(stdout);
      mexPrintf("col %3d:", i + 2);
      for (k = 0; k < nLL[i + 1]; k++)
        mexPrintf("%4d", LL[i + 1][k] + 1);
      mexPrintf("\n");
      fflush(stdout);
    } else {
      mexPrintf("augmented pattern\n");
      fflush(stdout);
      mexPrintf("col %3d:", i + 1);
      for (k = 0; k < nLL[i]; k++)
        mexPrintf("%4d", LL[i][k] + 1);
      mexPrintf("\n");
      fflush(stdout);
    }
#endif

    /* scan columns associated with nonzeros L(:,i), but skip
       i and stop as soon as ndense is reached, be aware of C-style!
    */
    LLi = LL[i];
    /* skip diagonal index i */
    k = 1;
    /* safeguard the 2x2 case that L(i+1,i)~=0, which must not be the case! */
    if (D_ia[i + 1] - D_ia[i] > 1)
      if (k < nLL[i])
        if (LLi[k] == i + 1)
          k = 2;
    while (k < nLL[i]) {
      j = LLi[k];
      /* dense lower triangular block reached */
      if (j >= ndense)
        k = nLL[i];
      else {       /* scan column j and check for additional fill */
        r = k + 1; /* we do not need to check L(j,i) */
        s = 1;     /* skip L(j,j) */
        m = 0;     /* counter for auxiliary buffer */
        /* reference to the nonzero indices of column j>i */
        LLj = LL[j];
        /* column j is not yet dense */
        if (nLL[j] < n - j) {
#ifdef PRINT_INFO
          mexPrintf("scanning column %d, nz=%d\n", j + 1, nLL[j]);
          fflush(stdout);
#endif
          while (r < nLL[i] && s < nLL[j]) {
            p = LLi[r];
            q = LLj[s];
            if (p < q) {
              /* copy fill-index */
              ibuff[m++] = p;
              r++;
            } else if (q < p) {
              /* copy original index */
              ibuff[m++] = q;
              s++;
            } else {
              /* copy joint index */
              ibuff[m++] = q;
              r++;
              s++;
            }
          } /* end while */
          while (r < nLL[i]) {
            /* copy fill-index */
            ibuff[m++] = LLi[r];
            r++;
          } /* end while */
          while (s < nLL[j]) {
            /* copy original index */
            ibuff[m++] = LLj[s];
            s++;
          } /* end while */
          /* did we encounter fill-in? */
          if (m + 1 > nLL[j]) {
            /* increase memory if necessary */
            eLL[j] = MAX(eLL[j], m + 1);
            LL[j] = (integer *)ReAlloc(LL[j], (size_t)eLL[j] * sizeof(integer),
                                       "Dinverse_aware:LL[j]");
            /* new number of row indices in column j */
            nLL[j] = m + 1;
            /* copy merged index array back */
            memcpy(LL[j] + 1, ibuff, (size_t)m * sizeof(integer));
          }
        } /* end if (not dense column j) */
        else {
#ifdef PRINT_INFO
          mexPrintf("dense column %d\n", j + 1);
          fflush(stdout);
#endif
        }

        /* start of dense columns detected? */
        if (nLL[j] == n - j && j == ndense - 1) {
          ndense--;
#ifdef PRINT_INFO
          mexPrintf("ndense=%d\n", ndense + 1);
          fflush(stdout);
#endif
        }
      } /* end else */
      k++;
    } /* end while */

    /* advance one column more in the 2x2 case */
    if (D_ia[i + 1] - D_ia[i] == 1)
      i++;
    else
      i += 2;
  } /* end for i */

#ifdef SORT_ENTRIES
  /* double buffer of size n */
  dbuff = (doubleprecision *)MAlloc((size_t)n * sizeof(doubleprecision),
                                    "Dinverse_aware:dbuff");
#endif

  /* export augmented L */
  /* first compute total number of nonzeros */
  nz = 0;
  for (i = 0; i < n; i++) {
    nz += nLL[i];
  } /* end for i */
  plhs[0] = mxCreateSparse((mwSize)n, (mwSize)n, (mwSize)nz, mxREAL);
  LL_output = (mxArray *)plhs[0];
  LL_ja = (mwIndex *)mxGetIr(LL_output);
  LL_ia = (mwIndex *)mxGetJc(LL_output);
  LL_valuesR = (double *)mxGetPr(LL_output);

  LL_ia[0] = 0;
  for (i = 0; i < n; i++) {
    j = L_ia[i];
#ifdef SORT_ENTRIES
    m = L_ia[i + 1] - j;
    memcpy(ibuff, L_ja + j, (size_t)m * sizeof(integer));
    memcpy(dbuff, L_valuesR + j, (size_t)m * sizeof(doubleprecision));
    /* sort indices of in increasing order */
    /* we do not need any further re-allocation for LL, use eLL as stack */
    Dqsort(dbuff, ibuff, eLL, &m);
    j = 0;
#endif
    k = 0;
    m = LL_ia[i];
    while (j < L_ia[i + 1]) {
      l = LL[i][k];
      LL_ja[m] = l;
/* indices match, original entry */
#ifdef SORT_ENTRIES
      if (l == ibuff[j]) {
        LL_valuesR[m] = dbuff[j];
        j++;
      }
#else
      if (l == L_ja[j]) {
        LL_valuesR[m] = L_valuesR[j];
        j++;
      }
#endif
      else { /* the row indices of L MUST be a subset of that of LL
                Thus this must be fill-in */
        LL_valuesR[m] = RM;
      }
      k++;
      m++;
    } /* end while */
    /* remaining fill-in */
    while (k < nLL[i]) {
      l = LL[i][k];
      LL_ja[m] = l;
      LL_valuesR[m] = RM;
      m++;
      k++;
    } /* end while */
    LL_ia[i + 1] = LL_ia[i] + nLL[i];

    /* give away index memory for column i */
    FRee(LL[i]);
  } /* end for i */

  /* release memory */
  FRee(ibuff);
  FRee(LL);
  FRee(nLL);
  FRee(eLL);
#ifdef SORT_ENTRIES
  FRee(dbuff);
#endif

#ifdef PRINT_INFO
  mexPrintf("Dinverse_aware: memory released\n");
  fflush(stdout);
#endif

  return;
}
