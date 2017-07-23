/* ========================================================================== */
/* === partitionilupackmetisn mexFunction
 * ========================================= */
/* ========================================================================== */

/*
    Usage:

    Return METIS multilevel reordering by nodes

    Example:

    % for initializing parameters
    [p,dist] = partitionilupackmetisn(A,parts);



    Authors:

        Matthias Bollhoefer, TU Braunschweig

    Date:

        March 02, 2013. ILUPACK V2.4.

    Acknowledgements:

        This work was supported from 2002 to 2007 by the DFG research center
        MATHEON "Mathematics for key technologies"

    Notice:

        Copyright (c) 2013 by TU Braunschweig.  All Rights Reserved.

        THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
        EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.

    Availability:

        This file is located at
        http://ilupack.tu-bs.de
*/

/* ========================================================================== */
/* === Include files and prototypes ========================================= */
/* ========================================================================== */

#include "matrix.h"
#include "mex.h"
#include <ilupack.h>
#include <metis_defs.h>
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
  integer *p, *invq, nB = 0, nleaves, nleaves_backup, nthreads, *ddist,
                     ddistsize, *rangtab, *treetab, dimT;
  double *prowscale = NULL, *pcolscale = NULL;
  int ierr, i, j, k, l;
  size_t mrows, ncols;
  mwSize nnz;
  double *pr;
  mwIndex *A_ia, *A_ja;

  if (nrhs != 2)
    mexErrMsgTxt("Two input arguments required.");
  else if (nlhs != 3)
    mexErrMsgTxt("Three output arguments required.");
  else if (!mxIsNumeric(prhs[0]))
    mexErrMsgTxt("First input must be a matrix.");
  else if (!mxIsNumeric(prhs[1]))
    mexErrMsgTxt("Second input must be a number.");

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
                           "partitionilupackmetisn");
  A.ja = (integer *)MAlloc((size_t)nnz * sizeof(integer),
                           "partitionilupackmetisn");
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
  pr = mxGetPr(A_input);
  nleaves = *pr;
  nthreads = nleaves; /* nthreads=1; */

  /* take nearest power of 2 greater than or equal to nleaves */
  nleaves_backup = nleaves;
  i = 0;
  while (nleaves > 1) {
    nleaves >>= 1;
    i++;
  } /* end while */
  nleaves = 1 << i;
  if (nleaves < nleaves_backup)
    nleaves <<= 1;
  if (nleaves > METIS_MAXLEAVES)
    nleaves = METIS_MAXLEAVES;

  /* init parameters to their default values */
  DSPDAMGinit(&A, &param);
  param.ipar[7] = 0;
  param.ipar[8] = 0;

  ddistsize = 6 * nleaves;
  ddist = (integer *)CAlloc((size_t)ddistsize * 6, sizeof(integer),
                            "partitionilupackmetisn");
  p = (integer *)MAlloc((size_t)A.nc * sizeof(integer),
                        "partitionilupackmetisn");
  invq = (integer *)MAlloc((size_t)A.nc * sizeof(integer),
                           "partitionilupackmetisn");
  ierr = DGNLpartition_metis_n(A, p, invq, nleaves, nthreads, ddist, ddistsize,
                               &param);

  /* extract tree information from the Metis partitioning */
  dimT = nleaves * 2;
  /* rangtab describes the start of the j-th node in C-style (starting with
     zero;
     consistently rangtab[dimT-1] is n (mapping behind the partitioning)
  */
  rangtab = (integer *)CAlloc((size_t)dimT, sizeof(integer),
                              "partitionilupackmetisn:rangtab");
  /* treetab returns the parent information in C-style. There are 2*nleaves-1
     nodes in a binary tree. "-1" refers to the root node, which does not have a
     parent
  */
  treetab = (integer *)CAlloc((size_t)dimT, sizeof(integer),
                              "partitionilupackmetisn:treetab");
  ILUPACK_TreeComputation(nleaves, ddist, rangtab, treetab);

  /* Create output vector */
  nlhs = 3;
  plhs[0] = mxCreateDoubleMatrix(1, mrows, mxREAL);
  pr = (double *)mxGetPr(plhs[0]);
  for (i = 0; i < mrows; i++)
    pr[i] = p[i] + 1;

  plhs[1] = mxCreateDoubleMatrix(1, dimT, mxREAL);
  pr = (double *)mxGetPr(plhs[1]);
  for (i = 0; i < dimT; i++)
    pr[i] = rangtab[i] + 1;

  plhs[2] = mxCreateDoubleMatrix(1, dimT, mxREAL);
  pr = (double *)mxGetPr(plhs[2]);
  for (i = 0; i < dimT; i++)
    pr[i] = treetab[i] + 1;

  free(p);
  free(invq);
  free(ddist);
  free(treetab);
  free(rangtab);
  free(A.ia);
  free(A.ja);
  return;
}
