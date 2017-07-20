/* ========================================================================== */
/* 00=== symilupackcamd mexFunction ========================================= */
/* ========================================================================== */

/*
    Usage:

    Return constrained AMD reordering

    Example:

    % for initializing parameters
    p = symilupackcamd(A);



    Authors:

        Matthias Bollhoefer, TU Braunschweig

    Date:

        April 03, 2013. ILUPACK V2.5.

    Acknowledgements:

        This work was supported from 2002 to 2007 by the DFG research center
        MATHEON "Mathematics for key technologies"

    Notice:

        Copyright (c) 2013 by TU Braunschweig.  All Rights Reserved.

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

#define MAX_FIELDS 100

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
  Dmat A;
  DILUPACKparam param;
  mxArray *A_input;
  integer *p, *invq, nB = 0, *c;
  double *prowscale = NULL, *pcolscale = NULL;
  int ierr, i, j, k, l;
  size_t mrows, ncols;
  mwSize nnz;
  double *pr;
  mwIndex *A_ia, *A_ja;

  if (nrhs != 2)
    mexErrMsgTxt("Two input arguments are required.");
  else if (nlhs > 1)
    mexErrMsgTxt("Too many output arguments.");
  else if (!mxIsNumeric(prhs[0]))
    mexErrMsgTxt("First input must be a matrix.");

  /* The first input must be a square matrix.*/
  A_input = (mxArray *)prhs[0];
  mrows = mxGetM(A_input);
  ncols = mxGetN(A_input);
  nnz = mxGetNzmax(A_input);
  if (mrows != ncols) {
    mexErrMsgTxt("First input must be a square matrix.");
  }
  A_ja = (mwIndex *)mxGetIr(A_input);
  A_ia = (mwIndex *)mxGetJc(A_input);

  A.nc = A.nr = mrows;
  A.ia = (integer *)MAlloc((size_t)(A.nc + 1) * sizeof(integer),
                           "symilupackmetisn");
  A.ja = (integer *)MAlloc((size_t)nnz * sizeof(integer), "symilupackmetisn");
  A.a = NULL;
  A.ia[0] = 1;
  for (i = 0; i < ncols; i++) {
    A.ia[i + 1] = A.ia[i];
    for (j = A_ia[i]; j < A_ia[i + 1]; j++) {
      k = A_ja[j];
      if (k >= i) {
        l = A.ia[i + 1] - 1;
        A.ja[l] = k + 1;
        A.ia[i + 1] = l + 2;
      }
    }
  }

  /* The first input must be a square matrix.*/
  A_input = (mxArray *)prhs[1];
  mrows = mxGetM(A_input);
  ncols = mxGetN(A_input);
  pr = (double *)mxGetPr(A_input);
  c = (integer *)MAlloc((size_t)(mrows * ncols) * sizeof(integer),
                        "symilupackcamd");
  for (i = 0; i < mrows * ncols; i++) {
    c[i] = pr[i];
  }

  /* init parameters to their default values */
  DSPDAMGinit(&A, &param);
  param.ipar[7] = 0;
  param.ipar[8] = 0;
  param.indpartial = c;

  p = (integer *)MAlloc((size_t)A.nc * sizeof(integer), "symilupackcamd");
  invq = (integer *)MAlloc((size_t)A.nc * sizeof(integer), "symilupackcamd");
  ierr = DGNLperm_camd(A, prowscale, pcolscale, p, invq, &nB, &param);

  /* Create output vector */
  nlhs = 1;
  plhs[0] = mxCreateDoubleMatrix(1, mrows, mxREAL);
  pr = (double *)mxGetPr(plhs[0]);
  for (i = 0; i < mrows; i++)
    pr[i] = p[i];

  free(p);
  free(invq);
  free(A.ia);
  free(A.ja);
  free(c);
  return;
}
