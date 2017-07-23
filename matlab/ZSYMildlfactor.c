/* $Id: ZSYMildlfactor.c 796 2015-08-06 20:27:07Z bolle $ */
/* ========================================================================== */
/* === AMGfactor mexFunction ================================================ */
/* ========================================================================== */

/*
    Usage:

    Return the structure 'options' and incomplete LDL^T preconditioner

    Example:

    % for initializing parameters
    [L,D,P,scal,options]=ZSYMildlfactor(A,options)


    Authors:

        Matthias Bollhoefer, TU Braunschweig

    Date:

        February 20, 2015. ILUPACK V2.5.

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
  Zmat A;
  ZILUPACKparam options;
  integer *ibuff, *jlu, posiwk;
  const char **fnames;
  const mwSize *dims;
  const mwSize mydims[] = {1, 1};
  mxClassID *classIDflags;
  mxArray *tmp, *fout, *A_input, *options_input, *options_output;
  char *pdata, *input_buf, *output_buf;
  mwSize nnz, ndim, buflen;
  mwIndex jstruct;
  int ifield, status, nfields, i, j, k, l, m;
  integer *p, *invq, nB = 0, ierr, PILUCparam;
  LONG_INT imem, myimem, mymaximem;
  doublecomplex *prowscale, *pcolscale, *alu, *dbuff;
  size_t mrows, ncols, sizebuf, mem;
  double dbuf, dbufr, dbufi, *A_valuesR, *A_valuesI, *pr, *pi;
  mwIndex *A_ja, /* row indices of input matrix A */
      *A_ia;     /* column pointers of input matrix A */

  if (nrhs != 2)
    mexErrMsgTxt("Two input arguments required.");
  else if (nlhs != 5)
    mexErrMsgTxt("wrong number of output arguments.");
  else if (!mxIsStruct(prhs[1]))
    mexErrMsgTxt("Second input must be a structure.");
  else if (!mxIsNumeric(prhs[0]))
    mexErrMsgTxt("First input must be a matrix.");

  /* The first input must be a square matrix.*/
  A_input = (mxArray *)prhs[0];
  /* get size of input matrix A */
  mrows = mxGetM(A_input);
  ncols = mxGetN(A_input);
  nnz = mxGetNzmax(A_input);
  if (mrows != ncols) {
    mexErrMsgTxt("First input must be a square matrix.");
  }
  if (!mxIsSparse(A_input)) {
    mexErrMsgTxt("ILUPACK: input matrix must be in sparse format.");
  }

  /* copy input matrix to sparse row format */
  A.nc = A.nr = mrows;
  A.ia = (integer *)MAlloc((size_t)(A.nc + 1) * sizeof(integer),
                           "ZSYMilupackfactor");
  A.ja = (integer *)MAlloc((size_t)nnz * sizeof(integer), "ZSYMilupackfactor");
  A.a = (doublecomplex *)MAlloc((size_t)nnz * sizeof(doublecomplex),
                                "ZSYMilupackfactor");

  A_ja = (mwIndex *)mxGetIr(A_input);
  A_ia = (mwIndex *)mxGetJc(A_input);
  A_valuesR = (double *)mxGetPr(A_input);
  A_valuesI = (double *)mxGetPi(A_input);

  /* -------------------------------------------------------------------- */
  /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
  /*     notation.                                                        */
  /* -------------------------------------------------------------------- */

  A.ia[0] = 1;
  for (i = 0; i < ncols; i++) {
    A.ia[i + 1] = A.ia[i];
    for (j = A_ia[i]; j < A_ia[i + 1]; j++) {
      k = A_ja[j];
      /* make sure that only the upper triangular part is imported */
      if (k >= i) {
        l = A.ia[i + 1] - 1;
        A.ja[l] = k + 1;
        A.a[l].r = A_valuesR[j];
        A.a[l].i = A_valuesI[j];
        A.ia[i + 1] = l + 2;
      }
    }
  }
  A.nnz = A.ia[A.nc] - 1;

  /* initialize ILUPACK options structure to its default options */
  ZSYMAMGinit(&A, &options);
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: sparse matrix imported\n");
  fflush(stdout);
#endif

  /* Get second input arguments */
  options_input = (mxArray *)prhs[1];
  nfields = mxGetNumberOfFields(options_input);

  /* Allocate memory  for storing classIDflags */
  classIDflags =
      (mxClassID *)mxCalloc((size_t)nfields, (size_t)sizeof(mxClassID));

  /* allocate memory  for storing pointers */
  fnames = mxCalloc((size_t)nfields, (size_t)sizeof(*fnames));
  /* Get field name pointers */
  for (ifield = 0; ifield < nfields; ifield++) {
    fnames[ifield] = mxGetFieldNameByNumber(options_input, ifield);
  }

  /* import data */
  for (ifield = 0; ifield < nfields; ifield++) {
    tmp = mxGetFieldByNumber(options_input, 0, ifield);
    classIDflags[ifield] = mxGetClassID(tmp);

    ndim = mxGetNumberOfDimensions(tmp);
    dims = mxGetDimensions(tmp);

    /* Create string/numeric array */
    if (classIDflags[ifield] == mxCHAR_CLASS) {
      /* Get the length of the input string. */
      buflen = (mxGetM(tmp) * mxGetN(tmp)) + 1;

      /* Allocate memory for input and output strings. */
      input_buf = (char *)mxCalloc((size_t)buflen, (size_t)sizeof(char));

      /* Copy the string data from tmp into a C string input_buf. */
      status = mxGetString(tmp, input_buf, buflen);

      /* ordering */
      if (!strcmp("ordering", fnames[ifield])) {
        if (strcmp(options.ordering, input_buf)) {
          options.ordering =
              (char *)MAlloc((size_t)buflen * sizeof(char), "ZSYMildlfactor");
          strcpy(options.ordering, input_buf);
        }
      } else {
        /* mexPrintf("%s ignored\n",fnames[ifield]);fflush(stdout); */
      }
    } else {
      if (!strcmp("matching", fnames[ifield])) {
        options.matching = *mxGetPr(tmp);
      } else if (!strcmp("droptol", fnames[ifield])) {
        options.droptol = *mxGetPr(tmp);
      } else if (!strcmp("diagcomp", fnames[ifield])) {
        options.maxit = *mxGetPr(tmp); /* abuse maxit as indicator */
      } else if (!strcmp("lfil", fnames[ifield])) {
        options.lfil = *mxGetPr(tmp);
      } else {
        /* mexPrintf("%s ignored\n",fnames[ifield]);fflush(stdout); */
      }
    }
  }
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: options structure imported\n");
  fflush(stdout);
#endif

  /* memory for scaling and permutation */
  p = (integer *)MAlloc((size_t)A.nc * sizeof(integer), "ZSYMildlfactor");
  invq = (integer *)MAlloc((size_t)A.nc * sizeof(integer), "ZSYMildlfactor");
  pcolscale = (doublecomplex *)MAlloc((size_t)A.nc * sizeof(doublecomplex),
                                      "ZSYMildlfactor");
  prowscale = pcolscale;
  nB = A.nc;
  if (options.matching) {
    if (!strcmp("metise", options.ordering)) {
#ifdef _MC64_MATCHING_
      ierr = ZSYMperm_mc64_metis_e(A, prowscale, pcolscale, p, invq, &nB,
                                   &options);
#else
      ierr = ZSYMperm_matching_metis_e(A, prowscale, pcolscale, p, invq, &nB,
                                       &options);
#endif
#ifdef PRINT_INFO
      mexPrintf("ZSYMildlfactor: mwm+metise performed\n");
      fflush(stdout);
#endif
    } else if (!strcmp("amd", options.ordering)) {
#ifdef _MC64_MATCHING_
      ierr = ZSYMperm_mc64_amd(A, prowscale, pcolscale, p, invq, &nB, &options);
#else
      ierr = ZSYMperm_matching_amd(A, prowscale, pcolscale, p, invq, &nB,
                                   &options);
#endif
#ifdef PRINT_INFO
      mexPrintf("ZSYMildlfactor: mwm+amd performed\n");
      fflush(stdout);
#endif
    } else if (!strcmp("rcm", options.ordering)) {
#ifdef _MC64_MATCHING_
      ierr = ZSYMperm_mc64_rcm(A, prowscale, pcolscale, p, invq, &nB, &options);
#else
      ierr = ZSYMperm_matching_rcm(A, prowscale, pcolscale, p, invq, &nB,
                                   &options);
#endif
#ifdef PRINT_INFO
      mexPrintf("ZSYMildlfactor: mwm+rcm performed\n");
      fflush(stdout);
#endif
    } else /* METIS nested dissection by nodes */
    /* if (!strcmp("metisn",options.ordering)) */ {
#ifdef _MC64_MATCHING_
      ierr = ZSYMperm_mc64_metis_n(A, prowscale, pcolscale, p, invq, &nB,
                                   &options);
#else
      ierr = ZSYMperm_matching_metis_n(A, prowscale, pcolscale, p, invq, &nB,
                                       &options);
#endif
#ifdef PRINT_INFO
      mexPrintf("ZSYMildlfactor: mwm+metisn performed\n");
      fflush(stdout);
#endif
    }
  } else {
    if (!strcmp("metisn", options.ordering)) {
      ierr = ZSYMperm_metis_n(A, prowscale, pcolscale, p, invq, &nB, &options);
#ifdef PRINT_INFO
      mexPrintf("ZSYMildlfactor: metisn performed\n");
      fflush(stdout);
#endif
    } else if (!strcmp("metise", options.ordering)) {
      ierr = ZSYMperm_metis_e(A, prowscale, pcolscale, p, invq, &nB, &options);
#ifdef PRINT_INFO
      mexPrintf("ZSYMildlfactor: metise performed\n");
      fflush(stdout);
#endif
    } else if (!strcmp("amd", options.ordering)) {
      ierr = ZSYMperm_amd(A, prowscale, pcolscale, p, invq, &nB, &options);
#ifdef PRINT_INFO
      mexPrintf("ZSYMildlfactor: amd performed\n");
      fflush(stdout);
#endif
    } else if (!strcmp("rcm", options.ordering)) {
      ierr = ZSYMperm_rcm(A, prowscale, pcolscale, p, invq, &nB, &options);
#ifdef PRINT_INFO
      mexPrintf("ZSYMildlfactor: rcm performed\n");
      fflush(stdout);
#endif
    } else { /* none */
      ierr = ZSYMperm_null(A, prowscale, pcolscale, p, invq, &nB, &options);
#ifdef PRINT_INFO
      mexPrintf("ZSYMildlfactor: only scaling performed\n");
      fflush(stdout);
#endif
    }
  }
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: scaling and reordering computed, ierr=%d\n", ierr);
  fflush(stdout);
#endif

#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: matrix rescaled\n");
  fflush(stdout);
#endif
#ifdef PRINT_INFO2
  mexPrintf("p\n");
  for (i = 0; i < A.nr; i++) {
    mexPrintf("%8d", p[i]);
  }
  mexPrintf("\n");
  mexPrintf("colscal\n");
  for (i = 0; i < A.nr; i++) {
    mexPrintf("%8.1e", pcolscale[i]);
  }
  mexPrintf("\n");
#endif
#ifdef PRINT_INFO2
  mexPrintf("matrix\n");
  for (i = 0; i < A.nr; i++) {
    mexPrintf("row %d\n", i + 1);
    for (j = A.ia[i] - 1; j < A.ia[i + 1] - 1; j++) {
      mexPrintf("%8d", A.ja[j]);
    }
    mexPrintf("\n");
    for (j = A.ia[i] - 1; j < A.ia[i + 1] - 1; j++) {
      mexPrintf("%8.1le", A.a[j].r);
    }
    mexPrintf("\n");
    for (j = A.ia[i] - 1; j < A.ia[i + 1] - 1; j++) {
      mexPrintf("%8.1le", A.a[j].i);
    }
    mexPrintf("\n");
  }
#endif

  PILUCparam = 0;
  /* bit 0: simple dual threshold dropping strategy */
  /* bit 1: a zero pivot at step k is replaced by a small number */
  PILUCparam |= 2;
  /* bit 2: simple Schur complement */
  /* bit 3: ILU is computed for the first time */
  /* bit 4: simple inverse-based dropping would have been used if it applies */
  /* bit 5  diagonal compensation is possibly done */
  if (options.maxit)
    PILUCparam |= 32;

  /* auxiliary buffers */
  ibuff =
      (integer *)MAlloc((size_t)14 * A.nc * sizeof(integer), "ZSYMildlfactor");
  dbuff = (doublecomplex *)MAlloc((size_t)4 * A.nc * sizeof(doublecomplex),
                                  "ZSYMildlfactor");

  /* ilu buffers */
  /* just a simple guess at least 2n+1! */
  mem = ELBOW * A.ia[A.nr] + 1;
  jlu = (integer *)MAlloc((size_t)mem * sizeof(integer), "ZSYMildlfactor");
  alu = (doublecomplex *)MAlloc((size_t)mem * sizeof(doublecomplex),
                                "ZSYMildlfactor");

  posiwk = 0;
  ierr = 0;
  do {
    imem = (LONG_INT)mem;
#ifdef PRINT_INFO
    printf("call SYMILUC, imem=%ld, lfil=%d, droptol=%8.1le\n", imem,
           options.lfil, options.droptol);
    fflush(stdout);
#endif
    ZSYMiluc(&(A.nc), A.a, A.ja, A.ia, &(options.lfil), &(options.droptol),
             &PILUCparam, p, invq, alu, jlu, &imem, dbuff, ibuff, &posiwk,
             &ierr);

    if (posiwk > 0) {
      /* total amount of memory requested by the parameters */
      myimem = ELBOW * (size_t)A.ia[A.nr];
      mem += myimem;
#ifdef PRINT_INFO
      printf("posiwk=%d, re-allocate memory using %ld\n", posiwk, mem);
      fflush(stdout);
#endif
      jlu =
          (integer *)ReAlloc(jlu, mem * sizeof(integer), "ZSYMildlfactor:jlu");
      alu = (doublecomplex *)ReAlloc(alu, mem * sizeof(doublecomplex),
                                     "ZSYMildlfactor:alu");
#ifdef PRINT_INFO
      printf("new memory %ld\n", mem);
      fflush(stdout);
#endif
    } /* end if posiwk>0 */
  } while (posiwk > 0);
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: matrix factorized, ierr=%d\n", ierr);
  fflush(stdout);
#endif

#ifdef PRINT_INFO2
  i = 0;
  while (i < A.nc) {
    /* 1x1 pivot */
    if (jlu[A.nc + 1 + i] == 0) {
      mexPrintf("row %d\n", i + 1);
      mexPrintf("%8d ", i + 1);
      for (j = jlu[i]; j < jlu[i + 1]; j++)
        mexPrintf("%8d", jlu[j - 1]);
      mexPrintf("\n");
      mexPrintf("%8.1le ", alu[i].r);
      for (j = jlu[i]; j < jlu[i + 1]; j++)
        mexPrintf("%8.1le", alu[j - 1].r);
      mexPrintf("\n");
      mexPrintf("%8.1le ", alu[i].i);
      for (j = jlu[i]; j < jlu[i + 1]; j++)
        mexPrintf("%8.1le", alu[j - 1].i);
      mexPrintf("\n");

      i++;
    } else { /* 2x2 pivot */
      mexPrintf("rows %d:%d\n", i + 1, i + 2);

      mexPrintf("row %d\n", i + 1);
      mexPrintf("%8d %8d ", i + 1, i + 2);
      for (j = jlu[i]; j < jlu[i + 1]; j++)
        mexPrintf("%8d", jlu[j - 1]);
      mexPrintf("\n");
      mexPrintf("%8.1le %8.1le ", alu[i].r, alu[A.nc + 1 + i].r);
      m = jlu[i];
      for (j = jlu[i]; j < jlu[i + 1]; j++) {
        mexPrintf("%8.1le", alu[m - 1].r);
        m += 2;
      }
      mexPrintf("\n");
      mexPrintf("%8.1le %8.1le ", alu[i].i, alu[A.nc + 1 + i].i);
      m = jlu[i];
      for (j = jlu[i]; j < jlu[i + 1]; j++) {
        mexPrintf("%8.1le", alu[m - 1].i);
        m += 2;
      }
      mexPrintf("\n");

      mexPrintf("row %d\n", i + 2);
      mexPrintf("%8d %8d ", i + 1, i + 2);
      for (j = jlu[i]; j < jlu[i + 1]; j++)
        mexPrintf("%8d", jlu[j - 1]);
      mexPrintf("\n");
      mexPrintf("%8.1le %8.1le ", alu[A.nc + 1 + i].r, alu[i + 1].r);
      m = jlu[i];
      for (j = jlu[i]; j < jlu[i + 1]; j++) {
        mexPrintf("%8.1le", alu[m].r);
        m += 2;
      }
      mexPrintf("\n");
      mexPrintf("%8.1le %8.1le ", alu[A.nc + 1 + i].i, alu[i + 1].i);
      m = jlu[i];
      for (j = jlu[i]; j < jlu[i + 1]; j++) {
        mexPrintf("%8.1le", alu[m].i);
        m += 2;
      }
      mexPrintf("\n");

      i += 2;
    }
  }
#endif

  /* export L */
  /* get memory requirement */
  i = 0;
  nnz = 0;
  while (i < A.nc) {
    /* 1x1 pivot */
    if (jlu[A.nc + 1 + i] == 0) {
      nnz += jlu[i + 1] - jlu[i] + 1;
      i++;
    } else { /* 2x2 pivot */
      nnz += 2 * (jlu[i + 1] - jlu[i] + 1);
      i += 2;
    }
  }
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: L requires %d nnz\n", nnz);
  fflush(stdout);
#endif
  plhs[0] = mxCreateSparse((mwSize)A.nr, (mwSize)A.nc, (mwSize)nnz, mxCOMPLEX);
  A_ja = (mwIndex *)mxGetIr(plhs[0]);
  A_ia = (mwIndex *)mxGetJc(plhs[0]);
  A_valuesR = (double *)mxGetPr(plhs[0]);
  A_valuesI = (double *)mxGetPi(plhs[0]);
  i = 0;
  /* pointer */
  k = 0;
  A_ia[0] = 0;
  while (i < A.nc) {
    /* 1x1 pivot */
    if (jlu[A.nc + 1 + i] == 0) {
#ifdef PRINT_INFO2
      mexPrintf("ZSYMildlfactor: row %d\n", i + 1);
      fflush(stdout);
#endif
      /* diagonal entry */
      /* index */
      A_ja[k] = i;
      /* values */
      A_valuesR[k] = 1.0;
      A_valuesI[k++] = 0.0;
      /* off-diagonal entries */
      for (j = jlu[i]; j < jlu[i + 1]; j++) {
        l = jlu[j - 1] - 1;
        /* index */
        A_ja[k] = l;
        /* values */
        /* mexPrintf("(%d,%d): %12.4le+i%12.4le\n",i+1,l+1,alu[m].r,alu[m].i);
         */
        A_valuesR[k] = alu[j - 1].r * alu[i].r - alu[j - 1].i * alu[i].i;
        A_valuesI[k++] = alu[j - 1].i * alu[i].r + alu[j - 1].r * alu[i].i;
      } /* end for j */

      /* increase memory appropriately*/
      A_ia[i + 1] = k;

      i++;
    } else { /* 2x2 pivot */
#ifdef PRINT_INFO2
      mexPrintf("ZSYMildlfactor: rows %d:%d\n", i + 1, i + 2);
      fflush(stdout);
#endif
      /* mexPrintf("(%d,%d): %12.4le+i%12.4le\n",i+1,i+1,alu[i].r,alu[i].i);
         mexPrintf("(%d,%d):
         %12.4le+i%12.4le\n",i+1,i+2,alu[A.nc+1+i].r,alu[A.nc+1+i].i);
         mexPrintf("(%d,%d):
         %12.4le+i%12.4le\n",i+2,i+1,alu[A.nc+1+i].r,alu[A.nc+1+i].i);
         mexPrintf("(%d,%d): %12.4le+i%12.4le\n",i+2,i+2,alu[i+1].r,alu[i+1].i);
      */

      /* diagonal entry column i */
      /* index */
      A_ja[k] = i;
      /* values */
      A_valuesR[k] = 1.0;
      A_valuesI[k++] = 0.0;
      /* off-diagonal entries column i */
      m = jlu[i];
      for (j = jlu[i]; j < jlu[i + 1]; j++) {
        l = jlu[j - 1] - 1;
        /* index */
        A_ja[k] = l;
        /* values */
        /* mexPrintf("(%d,%d):
         * %12.4le+i%12.4le\n",i+1,l+1,alu[m-1].r,-alu[m-1].i); */
        A_valuesR[k] = alu[m - 1].r * alu[i].r - alu[m - 1].i * alu[i].i +
                       alu[m].r * alu[A.nc + 1 + i].r -
                       alu[m].i * alu[A.nc + 1 + i].i;
        A_valuesI[k++] = alu[m - 1].r * alu[i].i + alu[m - 1].i * alu[i].r +
                         alu[m].r * alu[A.nc + 1 + i].i +
                         alu[m].i * alu[A.nc + 1 + i].r;
        m += 2;
      } /* end for j */

      /* increase pointer appropriately */
      A_ia[i + 1] = k;

      /* diagonal entry column i+1 */
      /* index */
      A_ja[k] = i + 1;
      /* values */
      A_valuesR[k] = 1.0;
      A_valuesI[k++] = 0.0;
      /* off-diagonal entries column i+1 */
      m = jlu[i];
      for (j = jlu[i]; j < jlu[i + 1]; j++) {
        l = jlu[j - 1] - 1;
        /* index */
        A_ja[k] = l;
        /* values */
        /* mexPrintf("(%d,%d): %12.4le+i%12.4le\n",i+2,l+1,alu[m].r,alu[m].i);
         */
        A_valuesR[k] = alu[m - 1].r * alu[A.nc + 1 + i].r -
                       alu[m - 1].i * alu[A.nc + 1 + i].i +
                       alu[m].r * alu[i + 1].r - alu[m].i * alu[i + 1].i;
        A_valuesI[k++] = alu[m - 1].r * alu[A.nc + 1 + i].i +
                         alu[m - 1].i * alu[A.nc + 1 + i].r +
                         alu[m].r * alu[i + 1].i + alu[m].i * alu[i + 1].r;
        m += 2;
      } /* end for j */

      /* increase memory appropriately*/
      A_ia[i + 2] = k;

      i += 2;
    }
  }
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: nnz consumed %d\n", k);
  fflush(stdout);
#endif
#ifdef PRINT_INFO2
  for (i = 0; i < A.nc; i++) {
    mexPrintf("row %d\n", i + 1);
    for (j = A_ia[i]; j < A_ia[i + 1]; j++) {
      mexPrintf("%8d", A_ja[j] + 1);
    }
    mexPrintf("\n");
    for (j = A_ia[i]; j < A_ia[i + 1]; j++) {
      mexPrintf("%8.1le", A_valuesR[j]);
    }
    mexPrintf("\n");
    for (j = A_ia[i]; j < A_ia[i + 1]; j++) {
      mexPrintf("%8.1le", A_valuesI[j]);
    }
    mexPrintf("\n");
  }
#endif
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: L exported\n");
  fflush(stdout);
#endif

  /* export D */
  /* get memory requirement */
  i = 0;
  nnz = 0;
  while (i < A.nc) {
    /* 1x1 pivot */
    if (jlu[A.nc + 1 + i] == 0) {
      nnz++;
      i++;
    } else { /* 2x2 pivot */
      nnz += 4;
      i += 2;
    }
  }
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: D requires %d\n", nnz);
  fflush(stdout);
#endif
  plhs[1] = mxCreateSparse((mwSize)A.nr, (mwSize)A.nc, (mwSize)nnz, mxCOMPLEX);
  A_ja = (mwIndex *)mxGetIr(plhs[1]);
  A_ia = (mwIndex *)mxGetJc(plhs[1]);
  A_valuesR = (double *)mxGetPr(plhs[1]);
  A_valuesI = (double *)mxGetPi(plhs[1]);
  i = 0;
  k = 0;
  /* pointer */
  A_ia[0] = 0;
  while (i < A.nc) {
    /* 1x1 pivot */
    if (jlu[A.nc + 1 + i] == 0) {
      /* index */
      A_ja[k] = i;
      /* values */
      dbuf = alu[i].r * alu[i].r + alu[i].i * alu[i].i;
      A_valuesR[k] = alu[i].r / dbuf;
      A_valuesI[k] = -alu[i].i / dbuf;

      /* increase memory by 1 */
      A_ia[i + 1] = ++k;

      i++;
    } else { /* 2x2 pivot */
      /* determinant */
      dbufr = alu[i].r * alu[i + 1].r - alu[i].i * alu[i + 1].i -
              (alu[A.nc + 1 + i].r * alu[A.nc + 1 + i].r -
               alu[A.nc + 1 + i].i * alu[A.nc + 1 + i].i);
      dbufi = alu[i].r * alu[i + 1].i + alu[i].i * alu[i + 1].r -
              (alu[A.nc + 1 + i].r * alu[A.nc + 1 + i].i +
               alu[A.nc + 1 + i].i * alu[A.nc + 1 + i].r);
      dbuf = dbufr * dbufr + dbufi * dbufi;
      /* index */
      A_ja[k] = i;
      A_ja[k + 1] = i + 1;
      /* values */
      A_valuesR[k] = (alu[i + 1].r * dbufr + alu[i + 1].i * dbufi) / dbuf;
      A_valuesI[k] = (-alu[i + 1].r * dbufi + alu[i + 1].i * dbufr) / dbuf;
      A_valuesR[k + 1] =
          -(alu[A.nc + 1 + i].r * dbufr + alu[A.nc + 1 + i].i * dbufi) / dbuf;
      A_valuesI[k + 1] =
          -(-alu[A.nc + 1 + i].r * dbufi + alu[A.nc + 1 + i].i * dbufr) / dbuf;

      /* increase memory by 2 */
      k += 2;
      A_ia[i + 1] = k;

      /* index */
      A_ja[k] = i;
      A_ja[k + 1] = i + 1;
      /* values */
      A_valuesR[k] =
          -(alu[A.nc + 1 + i].r * dbufr + alu[A.nc + 1 + i].i * dbufi) / dbuf;
      A_valuesI[k] =
          -(-alu[A.nc + 1 + i].r * dbufi + alu[A.nc + 1 + i].i * dbufr) / dbuf;
      A_valuesR[k + 1] = (alu[i].r * dbufr + alu[i].i * dbufi) / dbuf;
      A_valuesI[k + 1] = (-alu[i].r * dbufi + alu[i].i * dbufr) / dbuf;

      /* increase memory by 2 */
      k += 2;
      A_ia[i + 2] = k;

      i += 2;
    }
  }
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: D consumed %d\n", k);
  fflush(stdout);
#endif
#ifdef PRINT_INFO2
  for (i = 0; i < A.nc; i++) {
    mexPrintf("row %d\n", i + 1);
    for (j = A_ia[i]; j < A_ia[i + 1]; j++) {
      mexPrintf("%8d", A_ja[j] + 1);
    }
    mexPrintf("\n");
    for (j = A_ia[i]; j < A_ia[i + 1]; j++) {
      mexPrintf("%8.1le", A_valuesR[j]);
    }
    mexPrintf("\n");
    for (j = A_ia[i]; j < A_ia[i + 1]; j++) {
      mexPrintf("%8.1le", A_valuesI[j]);
    }
    mexPrintf("\n");
  }
#endif
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: D exported\n");
  fflush(stdout);
#endif

  /* export P */
  plhs[2] = mxCreateSparse((mwSize)A.nr, (mwSize)A.nc, (mwSize)A.nc, mxREAL);
  A_ja = (mwIndex *)mxGetIr(plhs[2]);
  A_ia = (mwIndex *)mxGetJc(plhs[2]);
  A_valuesR = (double *)mxGetPr(plhs[2]);
  for (i = 0; i < A.nc; i++) {
    /* pointer */
    A_ia[i] = i;
    /* index */
    A_ja[i] = p[i] - 1;
    /* values */
    A_valuesR[i] = 1.0;
  }
  A_ia[A.nc] = A.nc;
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: P exported\n");
  fflush(stdout);
#endif

  /* export scal */
  plhs[3] = mxCreateSparse((mwSize)A.nr, (mwSize)A.nc, (mwSize)A.nc, mxREAL);
  A_ja = (mwIndex *)mxGetIr(plhs[3]);
  A_ia = (mwIndex *)mxGetJc(plhs[3]);
  A_valuesR = (double *)mxGetPr(plhs[3]);
  for (i = 0; i < A.nc; i++) {
    /* pointer */
    A_ia[i] = i;
    /* index */
    A_ja[i] = i;
    /* values */
    A_valuesR[i] = pcolscale[i].r;
  }
  A_ia[A.nc] = A.nc;
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: S exported\n");
  fflush(stdout);
#endif

  /* export options */
  plhs[4] = mxCreateStructMatrix((mwSize)1, (mwSize)1, nfields, fnames);
  options_output = plhs[4];

  /* export data */
  for (ifield = 0; ifield < nfields; ifield++) {
    tmp = mxGetFieldByNumber(options_input, 0, ifield);
    classIDflags[ifield] = mxGetClassID(tmp);

    ndim = mxGetNumberOfDimensions(tmp);
    dims = mxGetDimensions(tmp);

    /* Create string/numeric array */
    if (classIDflags[ifield] == mxCHAR_CLASS) {
      if (!strcmp("ordering", fnames[ifield])) {
        output_buf = (char *)mxCalloc((size_t)strlen(options.ordering) + 1,
                                      (size_t)sizeof(char));
        strcpy(output_buf, options.ordering);
        fout = mxCreateString(output_buf);
      } else {
        /* mexPrintf("%s ignored\n",fnames[ifield]);fflush(stdout); */
      }
    } else {
      fout = mxCreateNumericArray((mwSize)ndim, dims, classIDflags[ifield],
                                  mxREAL);
      pdata = mxGetData(fout);

      sizebuf = mxGetElementSize(tmp);
      if (!strcmp("matching", fnames[ifield])) {
        dbuf = options.matching;
        memcpy(pdata, &dbuf, (size_t)sizebuf);
      } else if (!strcmp("droptol", fnames[ifield])) {
        dbuf = options.droptol;
        memcpy(pdata, &dbuf, sizebuf);
      } else if (!strcmp("diagcomp", fnames[ifield])) {
        dbuf = options.maxit; /* abuse maxit as indicator */
        memcpy(pdata, &dbuf, (size_t)sizebuf);
      } else if (!strcmp("lfil", fnames[ifield])) {
        dbuf = options.lfil;
        memcpy(pdata, &dbuf, (size_t)sizebuf);
      } else {
        memcpy(pdata, mxGetData(tmp), (size_t)sizebuf);
      }
    }

    /* Set each field in output structure */
    mxSetFieldByNumber(options_output, (mwIndex)0, ifield, fout);
  } /* end for */
#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: options structure exported\n");
  fflush(stdout);
#endif

  switch (ierr) {
  case 0: /* perfect! */
    break;
  case -1: /* Error. input matrix may be wrong.
              (The elimination process has generated a
              row in L or U whose length is .gt.  n.) */
    mexErrMsgTxt("ILUPACK error, data may be wrong.");
    break;
  case -3: /* The matrix U overflows the array alu */
    mexErrMsgTxt("memory overflow");
    break;
  case -4: /* Illegal value for lfil */
    mexErrMsgTxt("Illegal value for `options.lfil'\n");
    break;
  default: /* zero pivot encountered at step number ierr */
    mexErrMsgTxt("zero pivot encountered, please reduce `options.droptol'\n");
    break;
  } /* end switch */

  /* release copy of input matrix */
  free(A.ia);
  free(A.ja);
  free(A.a);

  /* release permutation & scaling arrays */
  free(p);
  free(invq);
  free(pcolscale);

  /* release ILU memory */
  free(ibuff);
  free(dbuff);
  free(jlu);
  free(alu);

  /* options auxiliary fields */
  if (options.ibuff != NULL)
    free(options.ibuff);
  if (options.dbuff != NULL)
    free(options.dbuff);
  if (options.iaux != NULL)
    free(options.iaux);
  if (options.daux != NULL)
    free(options.daux);

  /* release temporary mex interface memory */
  mxFree(fnames);
  mxFree(classIDflags);

#ifdef PRINT_INFO
  mexPrintf("ZSYMildlfactor: memory released\n");
  fflush(stdout);
#endif

  return;
}
