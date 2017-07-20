/* $Id: ZSYMldl2bldl.c 795 2015-08-06 20:20:21Z bolle $ */
/* ========================================================================== */
/* === ZSYMldl2bldl mexFunction ============================================= */
/* ========================================================================== */

/*
    Usage:

    Return cell arrays BL and BD that refer to a block triangular factorization

    Example:

    % for initializing parameters
    [BL,BD]=ZSYMldl2bldl(L,D,threshold,maxsize,tol)


    Authors:

        Matthias Bollhoefer, TU Braunschweig

    Date:

        March 14, 2015. ILUPACK V2.5.

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
#include <blas.h>
#include <ilupack.h>
#include <ilupackmacros.h>
#include <lapack.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FIELDS 100
#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define MIN(A, B) (((A) >= (B)) ? (B) : (A))
#define ELBOW MAX(4.0, 2.0)
/* #define PRINT_CHECK  */
/* #define PRINT_INFO   */

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
  mwSize dims[1];
  const char *BLnames[] = {"J", "I", "L", "D"};
  const char *BDnames[] = {"J", "D"};
  mxArray *L_input, *D_input, *threshold_input, *maxsize_input, *tol_input,
      *block_column, *D_matrix, *L_matrix, *block_index, *BL, *BD;
  integer i, j, k, l, m, kk, n, flag, nnz, p, cnt, cnti, cntj, cntij,
      BLDsize = 10, *ia, *ja, *idxpos, *idxlst, maxsize, flag_piL, flag_piD;
  doubleprecision tol, threshold, *prL, *prD, *piL, *piD, mx, valr, vali, nrm,
      *pr, *pi;
  doublecomplex *zbuff, *BLDbuff, *pz;
  size_t mrows, ncols;
  double *L_valuesR, *D_valuesR, *L_valuesI, *D_valuesI;
  mwIndex *L_ja, /* row indices of input matrix L         */
      *L_ia,     /* column pointers of input matrix L     */
      *D_ja,     /* row indices of input matrix D         */
      *D_ia;     /* column pointers of input matrix D     */

  if (nrhs != 5)
    mexErrMsgTxt("Five input arguments required.");
  else if (nlhs != 2)
    mexErrMsgTxt("wrong number of output arguments.");
  else if (!mxIsNumeric(prhs[0]))
    mexErrMsgTxt("First input must be a matrix.");
  else if (!mxIsNumeric(prhs[1]))
    mexErrMsgTxt("Second input must be a matrix.");
  else if (!mxIsNumeric(prhs[2]))
    mexErrMsgTxt("Third input must be a number.");
  else if (!mxIsNumeric(prhs[3]))
    mexErrMsgTxt("Fourth input must be a number.");
  else if (!mxIsNumeric(prhs[4]))
    mexErrMsgTxt("Fifth input must be a number.");

  /* The first input must be a square matrix.*/
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
  L_valuesI = (double *)mxGetPi(L_input);
  flag_piL = (L_valuesI != NULL);
#ifdef PRINT_INFO
  mexPrintf("ZSYMldl2bldl: input parameter L imported\n");
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
  mexPrintf("ZSYMldl2bldl: input parameter D imported\n");
  fflush(stdout);
#endif

  /* The third input must a double number */
  threshold_input = (mxArray *)prhs[2];
  /* get size of input matrix D */
  mrows = mxGetM(threshold_input);
  ncols = mxGetN(threshold_input);
  if (1 != ncols || mrows != 1 || !mxIsNumeric(prhs[2])) {
    mexErrMsgTxt("Third argument must be scalar number.");
  }
  threshold = *mxGetPr(threshold_input);

#ifdef PRINT_INFO
  mexPrintf("ZSYMldl2bldl: input parameter threshold imported\n");
  fflush(stdout);
#endif

  /* The fourth input must be a  number */
  maxsize_input = (mxArray *)prhs[3];
  /* get size of input matrix Delta */
  mrows = mxGetM(maxsize_input);
  ncols = mxGetN(maxsize_input);
  if (1 != ncols || mrows != 1 || !mxIsNumeric(prhs[3])) {
    mexErrMsgTxt("Fourth argument must be number.");
  }
  maxsize = *mxGetPr(maxsize_input);
#ifdef PRINT_INFO
  mexPrintf("ZSYMldl2bldl: input parameter maxsize imported\n");
  fflush(stdout);
#endif

  /* The fifth input must be a scalar */
  tol_input = (mxArray *)prhs[4];
  /* get size of input matrix Delta */
  mrows = mxGetM(tol_input);
  ncols = mxGetN(tol_input);
  if (1 != ncols || mrows != 1 || !mxIsNumeric(prhs[4])) {
    mexErrMsgTxt("Fourth argument must be number.");
  }
  tol = *mxGetPr(tol_input);
#ifdef PRINT_INFO
  mexPrintf("ZSYMldl2bldl: input parameter tol imported\n");
  fflush(stdout);
#endif

#ifdef PRINT_INFO
  mexPrintf("ZSYMldl2bldl: input parameters imported\n");
  fflush(stdout);
#endif

  idxlst =
      (integer *)MAlloc((size_t)n * sizeof(integer), "ZSYMldl2bldl:idxlst");
  idxpos = (integer *)CAlloc((size_t)n, sizeof(integer), "ZSYMldl2bldl:idxpos");
  zbuff = (doublecomplex *)MAlloc((size_t)n * sizeof(doublecomplex),
                                  "ZSYMldl2bldl:zbuff");
  BLDbuff = (doublecomplex *)MAlloc(((size_t)BLDsize) * BLDsize *
                                        sizeof(doublecomplex),
                                    "ZSYMldl2bldl:BLDbuff");

  /* temporary output cell arrays of size n */
  dims[0] = n;
  BL = mxCreateCellArray((mwSize)1, dims);
  BD = mxCreateCellArray((mwSize)1, dims);

  i = 0;
  k = 0;
  while (i < n) {
#ifdef PRINT_CHECK
    mexPrintf("ZSYMldl2bldl: analyze columns starting with i=%d\n", i + 1);
    fflush(stdout);
#endif
    /* lower triangular part in column i */
    /* compute max_p>=i |l_{pi}| */
    mx = 0.0;
    for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
      valr = L_valuesR[m];
      vali = (flag_piL) ? L_valuesI[m] : 0.0;
      nrm = sqrt(valr * valr + vali * vali);
      if (nrm > mx)
        mx = nrm;
    } /* end for i */
    mx *= tol;
    /* extract essential nonzero subdiagonal pattern */
    cnt = 0;
    for (m = L_ia[i]; m < L_ia[i + 1]; m++) {
      p = L_ja[m];
      valr = L_valuesR[m];
      vali = (flag_piL) ? L_valuesI[m] : 0.0;
      nrm = sqrt(valr * valr + vali * vali);
      if (p > i && nrm >= tol) {
        idxlst[cnt] = p;
        idxpos[p] = ++cnt;
#ifdef PRINT_CHECK
        mexPrintf("ZSYMldl2bldl: store %d at position %d\n", p + 1, cnt);
        fflush(stdout);
#endif
      } /* end if */
    }   /* end for m */
    cnti = cnt;

    /* scan column j */
    j = i + 1;
    flag = -1;
    while (j < n && flag) {
#ifdef PRINT_CHECK
      mexPrintf("ZSYMldl2bldl: analyze column j=%d\n", j + 1);
      fflush(stdout);
#endif
      /* maximum number of columns exceeded? */
      if (j - i + 1 > maxsize) {
        flag = 0;
        j = j - 1;
      } else {
        /* strict lower triangular part in column j */
        /* compute max_p |l_{pj}| */
        mx = 0.0;
        for (m = L_ia[j]; m < L_ia[j + 1]; m++) {
          valr = L_valuesR[m];
          vali = (flag_piL) ? L_valuesI[m] : 0.0;
          nrm = sqrt(valr * valr + vali * vali);
          if (nrm > mx)
            mx = nrm;
        } /* end for m */
        mx *= tol;

        /* extract essential nonzero subdiagonal pattern */
        cntj = 0;
        cntij = 0;
        for (m = L_ia[j]; m < L_ia[j + 1]; m++) {
          p = L_ja[m];
          valr = L_valuesR[m];
          vali = (flag_piL) ? L_valuesI[m] : 0.0;
          nrm = sqrt(valr * valr + vali * vali);
          if (p > j && nrm >= tol) {
            /* do we meet an already existing nonzero entry? */
            if (idxpos[p])
              cntij++;
            else {
              cntj++;
              idxlst[cnt] = p;
              idxpos[p] = ++cnt;
#ifdef PRINT_CHECK
              mexPrintf("ZSYMldl2bldl: store %d at position %d\n", p + 1, cnt);
              fflush(stdout);
#endif
            } /* end if-else */
          }   /* end if */
        }     /* end for m */

        /* now cntij refers to the intersection of indices,
           cnti-cntij refers to the entries not shared by column j,
           cntj refers to the entries only existing in column j
           => we have cntij common entries and in total
              cnti+cntj entries
           there is a little exception concerning position j,  which
           possibly has to be excluded when the columns merge
        */
        l = (idxpos[j]) ? -1 : 0;
        if (cntij < threshold * (cnti + cntj + l)) {
#ifdef PRINT_CHECK
          mexPrintf("ZSYMldl2bldl: intersection: %d, union: %d, do not merge\n",
                    cntij, cnti + cntj + l);
          fflush(stdout);
#endif
          /* remove additional entries from column j */
          for (m = 0; m < cntj; m++) {
            /* additional index from column j */
            l = idxlst[cnti + m];
            idxpos[l] = 0;
          } /* end for m */
          cnt -= cntj;
#ifdef PRINT_INFO
          mexPrintf("ZSYMldl2bldl: additional entries removed\n");
          fflush(stdout);
#endif

          flag = 0;
          j = j - 1;
        } else {
/* remove index j from the list */
#ifdef PRINT_CHECK
          mexPrintf("ZSYMldl2bldl: intersection: %d, union: %d, merge\n", cntij,
                    cnti + cntj + l);
          fflush(stdout);
#endif
          if (l) {
            /* shuffle last entry to the former position of j */
            /* reduce number of indices */
            cnt--;
            /* position of j inside idxlst */
            l = idxpos[j] - 1;
            /* last nonzero index */
            m = idxlst[cnt];
            /* overwrite j */
            idxlst[l] = m;
            /* new position of m, shifted by one */
            idxpos[m] = l + 1;
            /* index j is now removed */
            idxpos[j] = 0;
          } /* end if l */
          /* update current number of nonzeros */
          cnti = cnt;

          j = j + 1;
        } /* end if-else */
      }   /* end if j-i+1>maxsize */
    }     /* end while j<n && flag */

    /* unite columns i:j */
    /* determine final column j */
    if (flag)
      j = n - 1;
    /* 2x2 diagonal block while truncating? */
    if (j < n - 1) {
      m = 0;
      /* m is the index of the second nonzero entry in column j */
      if (D_ia[j + 1] - D_ia[j] > 1)
        m = D_ja[D_ia[j] + 1];
      if (m > j) {
#ifdef PRINT_CHECK
        mexPrintf("ZSYMldl2bldl: also add column j=%d\n", j + 1);
        fflush(stdout);
#endif
        /* For simplicity also add column j+1 */
        j = j + 1;
        /* strict lower triangular part in column j */
        /* compute max_p |l_{pj}| */
        mx = 0.0;
        for (m = L_ia[j]; m < L_ia[j + 1]; m++) {
          valr = L_valuesR[m];
          vali = (flag_piL) ? L_valuesI[m] : 0.0;
          nrm = sqrt(valr * valr + vali * vali);
          if (nrm > mx)
            mx = nrm;
        } /* end for m */
        mx *= tol;

        /* extract essential nonzero subdiagonal pattern */
        for (m = L_ia[j]; m < L_ia[j + 1]; m++) {
          p = L_ja[m];
          valr = L_valuesR[m];
          vali = (flag_piL) ? L_valuesI[m] : 0.0;
          nrm = sqrt(valr * valr + vali * vali);
          if (p > j && nrm >= tol) {
            /* only consider the case of additional fill */
            if (!idxpos[p]) {
              idxlst[cnt] = p;
              idxpos[p] = ++cnt;
            } /* end if */
          }   /* end if */
        }     /* end for m */

        /* remove index j from the list */
        l = (idxpos[j]) ? -1 : 0;
        if (l) {
          /* shuffle last entry to the former position of j */
          /* reduce number of indices */
          cnt--;
          /* position of j inside idxlst */
          l = idxpos[j] - 1;
          /* last nonzero index */
          m = idxlst[cnt];
          /* overwrite j */
          idxlst[l] = m;
          /* new position of m, shifted by one */
          idxpos[m] = l + 1;
          /* index j is now removed */
          idxpos[j] = 0;
        } /* end if l */
      }   /* end if 2x2 case */
    }     /* end if j<n-1 */

#ifdef PRINT_INFO
    mexPrintf("ZSYMldl2bldl: create output structures for BL\n");
    fflush(stdout);
#endif

    /* set up new block column with four elements J, I, L, D */
    block_column = mxCreateStructMatrix((mwSize)1, (mwSize)1, 4, BLnames);

    /* structure element 0:  J */
    block_index = mxCreateDoubleMatrix((mwSize)1, (mwSize)(j - i + 1), mxREAL);
    pr = mxGetPr(block_index);
#ifdef PRINT_CHECK
    mexPrintf("ZSYMldl2bldl: store column indices\n");
    fflush(stdout);
#endif
    for (m = 0; m <= j - i; m++) {
      pr[m] = i + m + 1;
#ifdef PRINT_CHECK
      mexPrintf("%6d", i + m + 1);
#endif
    } /* end for m */
#ifdef PRINT_CHECK
    mexPrintf("\n");
    fflush(stdout);
#endif
    /* set each field in output structure */
    mxSetFieldByNumber(block_column, (mwIndex)0, 0, block_index);
#ifdef PRINT_INFO
    mexPrintf("ZSYMldl2bldl: block_column.J set\n");
    fflush(stdout);
#endif

    /* structure element 1:  I */
    block_index = mxCreateDoubleMatrix((mwSize)1, (mwSize)cnt, mxREAL);
    pr = mxGetPr(block_index);
    /* remove check marks */
    for (m = 0; m < cnt; m++) {
      l = idxlst[m];
      idxpos[l] = 0;
    } /* end for m */
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (idxpos[m]) {
        mexPrintf("ZSYMldl2bldl: idxpos[%d]=%d !=0 !!!\n", m + 1, idxpos[m]);
        fflush(stdout);
      }
    }
#endif
    /* sort indices of "idxlst" in increasing order */
    qqsorti(idxlst, idxpos, &cnt);
    /* clear buffer "idxpos" */
    for (m = 0; m < cnt; m++)
      idxpos[m] = 0;
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (idxpos[m]) {
        mexPrintf("ZSYMldl2bldl: idxpos[%d]=%d !=0 !!!!\n", m + 1, idxpos[m]);
        fflush(stdout);
      }
    }
#endif
/* transfer sorted indices and store location */
#ifdef PRINT_CHECK
    mexPrintf("ZSYMldl2bldl: store row indices\n");
    fflush(stdout);
#endif
    for (m = 0; m < cnt; m++) {
      l = idxlst[m];
      pr[m] = l + 1;
      idxpos[l] = m + 1;
#ifdef PRINT_CHECK
      mexPrintf("%6d", l + 1);
#endif
    } /* end for m */
#ifdef PRINT_CHECK
    mexPrintf("\n");
    fflush(stdout);
#endif
    /* set each field in output structure */
    mxSetFieldByNumber(block_column, (mwIndex)0, 1, block_index);
#ifdef PRINT_INFO
    mexPrintf("ZSYMldl2bldl: block_column.I set\n");
    fflush(stdout);
#endif

    /* structure element 2:  L */
    L_matrix =
        mxCreateDoubleMatrix((mwSize)cnt, (mwSize)(j - i + 1), mxCOMPLEX);
    prL = mxGetPr(L_matrix);
    piL = mxGetPi(L_matrix);
    /* structure element 3:  D */
    D_matrix = mxCreateDoubleMatrix((mwSize)(j - i + 1), (mwSize)(j - i + 1),
                                    mxCOMPLEX);
    prD = mxGetPr(D_matrix);
    piD = mxGetPi(D_matrix);
    /* init with zeros */
    for (m = 0; m < cnt * (j - i + 1); m++)
      prL[m] = piL[m] = 0.0;
    for (m = 0; m < (j - i + 1) * (j - i + 1); m++)
      prD[m] = piD[m] = 0.0;
    /* extract nonzeros from columns i:j */
    for (m = i; m <= j;
         m++, prL += cnt, prD += j - i + 1, piL += cnt, piD += j - i + 1) {
      for (l = L_ia[m]; l < L_ia[m + 1]; l++) {
        /* index p of L_{p,m} */
        p = L_ja[l];
        /* diagonal index */
        if (p == m) {
          prD[p - i] = 1.0;
          piD[p - i] = 0.0;
        }
        /* index p is located inside the strict lower triangular part
           of the diagonal block L_{i:j,i:j}
        */
        else if (m < p && p <= j) {
          prD[p - i] = L_valuesR[l];
          piD[p - i] = (flag_piL) ? L_valuesI[l] : 0.0;
        }
        /* index p must be part of L_{j+1:n,i:j} */
        else if (p > j) {
          /* is the index in the output structure present? */
          kk = idxpos[p];
          /* mexPrintf("index %d located position %d,
           * value=%8.1le\n",p+1,kk,L_valuesR[l]);fflush(stdout); */
          if (kk) {
            /* kk-1 is the position of the row index */
            prL[kk - 1] = L_valuesR[l];
            piL[kk - 1] = (flag_piL) ? L_valuesI[l] : 0.0;
          } /* end if */
        }   /* end if-elseif-elseif */
      }     /* end for l */
    }       /* end for m */
    /* clear positions from "idxpos" */
    for (m = 0; m < cnt; m++) {
      l = idxlst[m];
      idxpos[l] = 0;
    } /* end for m */
#ifdef PRINT_CHECK
    mexPrintf("ZSYMldl2bldl: lower triangular diagonal block\n");
    fflush(stdout);
    prD = mxGetPr(D_matrix);
    piD = mxGetPi(D_matrix);
    for (m = 0; m <= j - i; m++) {
      for (l = 0; l <= j - i; l++) {
        mexPrintf("%8.1le+%8.1lei", prD[m + l * (j - i + 1)],
                  piD[m + l * (j - i + 1)]);
      }
      mexPrintf("\n");
      fflush(stdout);
    }
#endif
#ifdef PRINT_CHECK
    mexPrintf("ZSYMldl2bldl: sub-diagonal block\n");
    fflush(stdout);
    prL = mxGetPr(L_matrix);
    piL = mxGetPi(L_matrix);
    for (m = 0; m < cnt; m++) {
      for (l = 0; l <= j - i; l++) {
        mexPrintf("%8.1le+%8.1lei", prL[m + l * cnt], piL[m + l * cnt]);
      }
      mexPrintf("\n");
      fflush(stdout);
    }
#endif
#ifdef PRINT_CHECK
    for (m = 0; m < n; m++) {
      if (idxpos[m]) {
        mexPrintf("ZSYMldl2bldl: idxpos[%d]=%d !=0 !!!\n", m + 1, idxpos[m]);
        fflush(stdout);
      }
    }
#endif
    /* build L_{i:j,i:j}^{-T}L_{j+1:n,i:j}^T using BLAS function DTRSV */
    if (cnt && j > i) {
      prL = mxGetPr(L_matrix);
      piL = mxGetPi(L_matrix);
      prD = mxGetPr(D_matrix);
      piD = mxGetPi(D_matrix);
      m = j - i + 1;
      /* prD is lower triangular, we have to solve with the transpose and it
         has unit diagonal part:
         -> "L", "T", "U"
         Furthermore, its size and its leading dimension is j-i+1
         prL is a cnt x (j-i+1) matrix, but we have to use its transpose which
         requires an index jump of cnt rather than 1
      */
      /* attention! MATLAB uses different storage scheme for complex numbers! */
      /* buffer for triangular matrix is too small */
      if (BLDsize < m) {
        BLDsize = m;
        BLDbuff = (doublecomplex *)ReAlloc(
            BLDbuff, ((size_t)BLDsize) * BLDsize * sizeof(doublecomplex),
            "ZSYMldl2bldl:zbuff");
      }
      /* copy triangular matrix */
      pz = BLDbuff;
      for (l = 0; l < m * m; l++) {
        pz->r = *prD++;
        pz->i = *piD++;
        pz++;
      }
      for (l = 0; l < cnt; l++) {
        /* copy prL/piL into zbuff */
        for (p = 0; p <= j - i; p++) {
          zbuff[p].r = prL[l + p * cnt];
          zbuff[p].i = (flag_piL) ? piL[l + p * cnt] : 0.0;
        } /* end for p */
        p = 1;
        ztrsv_("L", "T", "U", &m, BLDbuff, &m, zbuff, &p, 1, 1, 1);
        /* copy zbuff back to prL/piL */
        for (p = 0; p <= j - i; p++) {
          prL[l + p * cnt] = zbuff[p].r;
          if (flag_piL)
            piL[l + p * cnt] = zbuff[p].i;
        } /* end for p */
      }
#ifdef PRINT_CHECK
      mexPrintf("ZSYMldl2bldl: sub-diagonal block after DTRSV\n");
      fflush(stdout);
      prL = mxGetPr(L_matrix);
      piL = mxGetPi(L_matrix);
      for (m = 0; m < cnt; m++) {
        for (l = 0; l <= j - i; l++) {
          mexPrintf("%8.1le+%8.1lei", prL[m + l * cnt], piL[m + l * cnt]);
        }
        mexPrintf("\n");
        fflush(stdout);
      }
#endif
    } /* end if cnt & j>i */

    /* set each field in output structure */
    mxSetFieldByNumber(block_column, (mwIndex)0, 2, L_matrix);
    mxSetFieldByNumber(block_column, (mwIndex)0, 3, D_matrix);
#ifdef PRINT_INFO
    mexPrintf("ZSYMldl2bldl: block_column.L/D set\n");
    fflush(stdout);
#endif

    /* assign block column to cell array BL */
    mxSetCell(BL, (mwIndex)k, block_column);
#ifdef PRINT_INFO
    mexPrintf("ZSYMldl2bldl: BL{%d} set\n", k + 1);
    fflush(stdout);
#endif

#ifdef PRINT_INFO
    mexPrintf("ZSYMldl2bldl: create output structures for BD\n");
    fflush(stdout);
#endif

    /* set up new block column with two elements J, D */
    block_column = mxCreateStructMatrix((mwSize)1, (mwSize)1, 2, BDnames);

    /* structure element 0:  J */
    block_index = mxCreateDoubleMatrix((mwSize)1, (mwSize)(j - i + 1), mxREAL);
    pr = mxGetPr(block_index);
#ifdef PRINT_CHECK
    mexPrintf("ZSYMldl2bldl: store column indices\n");
    fflush(stdout);
#endif
    for (m = 0; m <= j - i; m++) {
      pr[m] = i + m + 1;
#ifdef PRINT_CHECK
      mexPrintf("%6d", i + m + 1);
#endif
    } /* end for m */
#ifdef PRINT_CHECK
    mexPrintf("\n");
    fflush(stdout);
#endif
    /* set each field in output structure */
    mxSetFieldByNumber(block_column, (mwIndex)0, 0, block_index);
#ifdef PRINT_INFO
    mexPrintf("ZSYMldl2bldl: block_column.J set\n");
    fflush(stdout);
#endif

    /* structure element 1:  D */
    nnz = D_ia[j + 1] - D_ia[i];
    D_matrix = mxCreateSparse((mwSize)(j - i + 1), (mwSize)(j - i + 1),
                              (mwSize)nnz, mxCOMPLEX);
    ia = (mwIndex *)mxGetJc(D_matrix);
    ja = (mwIndex *)mxGetIr(D_matrix);
    prD = (double *)mxGetPr(D_matrix);
    piD = (double *)mxGetPi(D_matrix);
    kk = 0;
    for (m = 0; m <= j - i; m++) {
      ia[m] = kk;
      for (l = D_ia[m + i]; l < D_ia[m + i + 1]; l++) {
        ja[kk] = D_ja[l] - i;
        prD[kk] = D_valuesR[l];
        piD[kk++] = (flag_piD) ? D_valuesI[l] : 0.0;
      } /* end for l */
    }   /* end for m */
    ia[m] = kk;
    /* set each field in output structure */
    mxSetFieldByNumber(block_column, (mwIndex)0, 1, D_matrix);
#ifdef PRINT_INFO
    mexPrintf("ZSYMldl2bldl: block_column.L/D set\n");
    fflush(stdout);
#endif

    /* assign block column to cell array BD */
    mxSetCell(BD, (mwIndex)k, block_column);
#ifdef PRINT_INFO
    mexPrintf("ZSYMldl2bldl: BD{%d}.D set\n", k + 1);
    fflush(stdout);
#endif

    k = k + 1;
    i = j + 1;

  } /* end while i<n */

  dims[0] = k;
  plhs[0] = mxCreateCellArray((mwSize)1, dims);
  plhs[1] = mxCreateCellArray((mwSize)1, dims);
  for (m = 0; m < k; m++) {
    block_column = mxGetCell(BL, (mwIndex)m);
    mxSetCell(plhs[0], (mwIndex)m, block_column);
    mxSetCell(BL, (mwIndex)m, NULL);
    block_column = mxGetCell(BD, (mwIndex)m);
    mxSetCell(plhs[1], (mwIndex)m, block_column);
    mxSetCell(BD, (mwIndex)m, NULL);
  } /* end for m */

  /* release memory */
  mxDestroyArray(BL);
  mxDestroyArray(BD);
  free(idxlst);
  free(idxpos);
  free(zbuff);
  free(BLDbuff);

#ifdef PRINT_INFO
  mexPrintf("ZSYMldl2bldl: memory released\n");
  fflush(stdout);
#endif

  return;
}
