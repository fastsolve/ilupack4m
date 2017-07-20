/* ========================================================================== */
/* === symmwmilupackrcmsp mexFunction ======================================= */
/* ========================================================================== */

/*
    $Id: symmwmilupackrcmsp.c 809 2016-02-17 10:48:31Z bolle $
    Usage:

    Return METIS multilevel reordering by nodes

    Example:

    % for initializing parameters
    p = symmwmilupackrcmsp(A);



    Authors:

        Matthias Bollhoefer, TU Braunschweig

    Date:

        March 09, 2008. ILUPACK V2.2.

    Acknowledgements:

        This work was supported from 2002 to 2007 by the DFG research center
        MATHEON "Mathematics for key technologies"

    Notice:

        Copyright (c) 2008 by TU Braunschweig.  All Rights Reserved.

        THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
        EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.

    Availability:

        This file is located at

        http://www.math.tu-berlin.de/ilupack/
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
  mxArray *A_input, *ind_input;
  integer *p, *invq, nB = 0;
  double *prowscale, *pcolscale;
  int ierr, i, j, k, l, lp, lm, m;
  size_t mrows, ncols;
  mwSize nnz;
  double *pr, *D, *A_a;
  mwIndex *A_ia, *A_ja;

  if (nrhs != 2)
    mexErrMsgTxt("Two input arguments are required.");
  else if (nlhs != 2)
    mexErrMsgTxt("Two output arguments are required.");
  else if (!mxIsNumeric(prhs[0]))
    mexErrMsgTxt("First input must be a matrix.");
  else if (!mxIsNumeric(prhs[1]))
    mexErrMsgTxt("Second input must be vector.");

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
  A_a = (double *)mxGetPr(A_input);

  A.nc = A.nr = mrows;
  A.ia = (integer *)MAlloc((size_t)(A.nc + 1) * sizeof(integer),
                           "symmwmilupackrcmsp");
  A.ja = (integer *)MAlloc((size_t)nnz * sizeof(integer), "symmwmilupackrcmsp");
  A.a = (double *)MAlloc((size_t)nnz * sizeof(double), "symmwmilupackrcmsp");
  A.ia[0] = 1;
  for (i = 0; i < ncols; i++) {
    A.ia[i + 1] = A.ia[i];
    for (j = A_ia[i]; j < A_ia[i + 1]; j++) {
      /* a_ik */
      k = A_ja[j];
      if (k >= i) {
        l = A.ia[i + 1] - 1;
        A.ja[l] = k + 1;
        A.a[l] = A_a[j];
        A.ia[i + 1] = l + 2;
      }
    }
  }

  /* The second input must be a vector.*/
  ind_input = (mxArray *)prhs[1];
  i = mxGetM(ind_input);
  j = mxGetN(ind_input);
  if (i * j != ncols) {
    mexErrMsgTxt("Second input must be a vector of same size as the matrix.");
  }

  /* init parameters to their default values */
  DSYMAMGinit(&A, &param);

  /* import ind */
  param.ind =
      (integer *)MAlloc((size_t)A.nc * sizeof(integer), "symmwmilupackrcmsp");
  pr = (double *)mxGetPr(ind_input);
  for (i = 0; i < A.nc; i++)
    param.ind[i] = pr[i];

  param.nindicator = 2 * A.nc;
  param.indicator =
      (integer *)MAlloc((size_t)param.nindicator * sizeof(integer),
                        "symmwmilupackrcmsp:param.indicator");
  j = 0;
  k = 0;
  lp = lm = 0;
  for (i = 0; i < A.nc; i++) {
    m = param.ind[i];
    param.indicator[i] = m;
    /* keep track on the first positive/negative entry */
    if (lp == 0 && m > 0)
      lp = m;
    if (lm == 0 && m < 0)
      lm = m;
    if (m < 0)
      j = -1;
    /* is there an underlying block structure? */
    if ((m > 0 && m != lp) || (m < 0 && m != lm))
      k = -1;
  }
  if (j) {
    param.flags |= SADDLE_POINT;
  }
  if (k) {
    param.flags |= BLOCK_STRUCTURE;
  }

  p = (integer *)MAlloc((size_t)A.nc * sizeof(integer), "symmwmilupackrcmsp");
  invq =
      (integer *)MAlloc((size_t)A.nc * sizeof(integer), "symmwmilupackrcmsp");
  prowscale =
      (double *)MAlloc((size_t)A.nc * sizeof(double), "symmwmilupackrcmsp");
  pcolscale = prowscale;
#ifdef _MC64_MATCHING_
  ierr = DSYMperm_mc64_rcm_sp(A, prowscale, pcolscale, p, invq, &nB, &param);
#elif defined _PARDISO_MATCHING_
  ierr = DSYMperm_mwm_rcm_sp(A, prowscale, pcolscale, p, invq, &nB, &param);
#else /* MUMPS matching */
  ierr =
      DSYMperm_matching_rcm_sp(A, prowscale, pcolscale, p, invq, &nB, &param);
#endif

  /* Create output vector */
  nlhs = 2;
  plhs[0] = mxCreateDoubleMatrix(1, mrows, mxREAL);
  pr = (double *)mxGetPr(plhs[0]);
  for (i = 0; i < mrows; i++)
    pr[i] = p[i];

  plhs[1] = mxCreateDoubleMatrix(mrows, 1, mxREAL);
  D = (double *)mxGetPr(plhs[1]);
  for (i = 0; i < mrows; i++)
    D[i] = prowscale[i];

  free(param.ind);
  free(param.indicator);
  free(p);
  free(invq);
  free(prowscale);
  free(A.ia);
  free(A.ja);
  return;
}
