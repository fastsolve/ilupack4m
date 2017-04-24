/* $Id: DSYMLDLsol.c 799 2015-08-12 14:52:16Z bolle $ */
/* ========================================================================== */
/* === DSYMLDLsol mexFunction =========================================== */
/* ========================================================================== */

/*
    Usage:

    Solve S^{-1}P^T (LDL^T) PS^{-1}z=y, where S is a diagonal scaling matrix,
    P is a permutation, and LDL^T is a triangular factorization with 1x1 and 2x2 pivots
    
    Example:

    % for initializing parameters
    z=DSYMLDLsol(y,Pvec,Svec,L,iD)


    Authors:

	Matthias Bollhoefer, TU Braunschweig

    Date:

	August 12, 2015. ILUPACK V2.5.  

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

#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <stdlib.h>
#include <ilupack.h>
#define _DOUBLE_REAL_
#include <ilupackmacros.h>

#define MAX_FIELDS 100
#define MAX(A,B) (((A)>=(B))?(A):(B))
#define MIN(A,B) (((A)>=(B))?(B):(A))
/* #define PRINT_CHECK  */
/* #define PRINT_INFO   */

/* ========================================================================== */
/* === mexFunction ========================================================== */
/* ========================================================================== */

void mexFunction
(
    /* === Parameters ======================================================= */

    int nlhs,			/* number of left-hand sides */
    mxArray *plhs [],		/* left-hand side matrices */
    int nrhs,			/* number of right--hand sides */
    const mxArray *prhs []	/* right-hand side matrices */
)
{
    mxArray         *L_input, *iD_input, *Pvec_input, *Svec_input, *y_input, *z_output;
    integer         i,j,k,l,n;
    doubleprecision  *pPvecr, *pSvecr, *pzr, *pyr, *dbuff, x,y;
    size_t          mrows, ncols;
    double          *L_valuesR, *iD_valuesR;
    mwIndex         *L_ja,                 /* row indices of input matrix L         */
                    *L_ia,                 /* column pointers of input matrix L     */
                    *iD_ja,                /* row indices of input matrix iD        */
                    *iD_ia;                /* column pointers of input matrix iD    */
    

    if (nrhs!=5)
       mexErrMsgTxt("Five input arguments are required.");
    else if (nlhs!=1)
       mexErrMsgTxt("wrong number of output arguments.");
    else if (!mxIsNumeric(prhs[0]))
       mexErrMsgTxt("First input must be a vector.");
    else if (!mxIsNumeric(prhs[1]))
       mexErrMsgTxt("Second input must be a vector.");
    else if (!mxIsNumeric(prhs[2]))
       mexErrMsgTxt("Third input must be a vector.");
    else if (!mxIsNumeric(prhs[3]))
       mexErrMsgTxt("Fourth input must be a matrix.");
    else if (!mxIsNumeric(prhs[4]))
       mexErrMsgTxt("Fifth input must be a matrix.");

    
    /* The first input must be a dense vector */
    y_input=(mxArray *)prhs[0];
    /* get size of input vector y */
    mrows=mxGetM(y_input);
    ncols=mxGetN(y_input);
    if (ncols!=1) {
       mexErrMsgTxt("First input must be a vector");
    }
    if (mxIsSparse (y_input)) {
        mexErrMsgTxt ("First input vector must be in dense format.") ;
    }
    n=mrows;
    pyr=(double *)mxGetPr(y_input);
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: input parameter y imported\n");fflush(stdout);
#endif
    

    /* The second input must be a dense vector */
    Pvec_input=(mxArray *)prhs[1];
    /* get size of input vector y */
    mrows=mxGetM(Pvec_input);
    ncols=mxGetN(Pvec_input);
    if (ncols!=1 || mrows!=n) {
       mexErrMsgTxt("Second input must be a vector of same size as first");
    }
    if (mxIsSparse (Pvec_input)) {
        mexErrMsgTxt ("Second input vector must be in dense format.") ;
    }
    pPvecr=(double *)mxGetPr(Pvec_input);
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: input parameter Pvec imported\n");fflush(stdout);
#endif
    

    /* The third input must be a dense vector */
    Svec_input=(mxArray *)prhs[2];
    /* get size of input vector y */
    mrows=mxGetM(Svec_input);
    ncols=mxGetN(Svec_input);
    if (ncols!=1 || mrows!=n) {
       mexErrMsgTxt("Third input must be a vector of same size as first");
    }
    if (mxIsSparse (Svec_input)) {
        mexErrMsgTxt ("Third input vector must be in dense format.") ;
    }
    pSvecr=(double *)mxGetPr(Svec_input);
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: input parameter Svec imported\n");fflush(stdout);
#endif

    
    /* The fourth input must be a square sparse unit lower triangular matrix.*/
    L_input=(mxArray *)prhs[3];
    /* get size of input matrix L */
    mrows=mxGetM(L_input);
    ncols=mxGetN(L_input);
    if (mrows!=ncols || ncols!=n) {
       mexErrMsgTxt("Fourth input must be a square matrix of the same size as the other inputs.");
    }
    if (!mxIsSparse (L_input)) {
        mexErrMsgTxt ("Fourth input matrix must be in sparse format.") ;
    }
    L_ja     =(mwIndex *)mxGetIr(L_input);
    L_ia     =(mwIndex *)mxGetJc(L_input);
    L_valuesR=(double *) mxGetPr(L_input);
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: input parameter L imported\n");fflush(stdout);
#endif

    
    /* The fifth input must be a square block diagonal matrix.*/
    iD_input=(mxArray *)prhs[4];
    /* get size of input matrix iD */
    mrows=mxGetM(iD_input);
    ncols=mxGetN(iD_input);
    if (mrows!=ncols || mrows!=n) {
       mexErrMsgTxt("Second input must be a square matrix of same size as the first matrix.");
    }
    if (!mxIsSparse (iD_input)) {
       mexErrMsgTxt ("Second input matrix must be in sparse format.") ;
    }
    iD_ja     =(mwIndex *)mxGetIr(iD_input);
    iD_ia     =(mwIndex *)mxGetJc(iD_input);
    iD_valuesR=(double *) mxGetPr(iD_input);
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: input parameter iD imported\n");fflush(stdout);
#endif

    /* buffer vector */
    dbuff=(double *)mxCalloc((size_t)n, (size_t)sizeof(double));

    /* create output vector */
    z_output=mxCreateDoubleMatrix((mwSize)n,(mwSize)1, mxREAL);
    plhs[0]=z_output;
    pzr=(double *)mxGetPr(z_output);
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: output vector and buffer allocated\n");fflush(stdout);
#endif

    /* permutation + diagonal scaling */
    for (i=0; i<n; i++) {
        /* the input vector counts permutations from 1,...,n */
        j=pPvecr[i]-1;
        dbuff[i]=pSvecr[i]*pyr[j];
    } /* end for i */
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: permutation + diagonal scaling done\n");fflush(stdout);
#endif

    
    /* forward substitution with unit lower triangular matrix */
    for (i=0; i<n; i++) {
        /* downdate with strict lower triangular part excluding the diagonal part which is 1 */
#ifdef PRINT_CHECK
        if (L_ia[i]>=L_ia[i+1]) {
	   mexPrintf("empty column %d\n",i+1);
	   fflush(stdout);
	}
	else {
	   if (L_ja[L_ia[i]]!=i) {
	      mexPrintf("column %d does not start with diagonal entry\n",i+1);
	      fflush(stdout);
	   }
	}
#endif
        for (k=L_ia[i]+1; k<L_ia[i+1]; k++) {
	    /* row index */
	    j=L_ja[k];
	    dbuff[j]-=L_valuesR[k]*dbuff[i];
	} /* end for k */
    } /* end for i */
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: forward substitution done\n");fflush(stdout);
#endif

    
    /* multiplication with a block diagonal matrix with 1x1 and 2x2 pivots */
    i=0;
    j=iD_ia[0];
    while (i<n) {
          k=iD_ia[i+1];
          /* 2x2 pivot */
          if (k-j>1) {
	     l=iD_ia[i+2];
	     if (l-k>1) { /* 2x2 pivot with four nonzero elements */
	        x=iD_valuesR[j]  *dbuff[i]+iD_valuesR[k]  *dbuff[i+1];
		y=iD_valuesR[j+1]*dbuff[i]+iD_valuesR[k+1]*dbuff[i+1];
	     }
	     else { /* 2x2 pivot with three nonzero elements [ a_{ii}   a_{i,i+1};
		                                              a_{i+1,i}    0     ] */
	        x=iD_valuesR[j]  *dbuff[i]+iD_valuesR[k]  *dbuff[i+1];
		y=iD_valuesR[j+1]*dbuff[i];
	     }
	     dbuff[i]  =x;
	     dbuff[i+1]=y;
	     i+=2;
	     j=iD_ia[i];
	  }
	  else {
	     if (iD_ja[j]!=i) { /* 2x2 pivot with a_{ii}=0 */
	        l=iD_ia[i+2];
		if (l-k>1) { /* 2x2 pivot with three nonzero elements [   0      a_{i,i+1};
		                                                       a_{i+1,i} a_{i+1,i+1}] */
		   x=                         iD_valuesR[k]  *dbuff[i+1];
		   y=iD_valuesR[j  ]*dbuff[i]+iD_valuesR[k+1]*dbuff[i+1];
		}
		else { /* 2x2 pivot with two nonzero elements [   0      a_{i,i+1};
		                                               a_{i+1,i}    0     ] */
		   x=                         iD_valuesR[k]  *dbuff[i+1];
		   y=iD_valuesR[j  ]*dbuff[i];
		}
		dbuff[i]  =x;
		dbuff[i+1]=y;
		i+=2;
		j=iD_ia[i];
             }
	     else { /* 1x1 pivot */
	        dbuff[i]*=iD_valuesR[j];
		i++;
		j=k;
	     }
	  }
    } /* end while i */
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: multiplication by block diagonal matrix done\n");fflush(stdout);
#endif


    /* back substitution with unit upper triangular matrix */
    for (i=n-1; i>=0; i--) {
        /* scalar product with strict upper triangular part excluding the diagonal part which is 1 */
        for (k=L_ia[i]+1; k<L_ia[i+1]; k++) {
	    /* row index */
	    j=L_ja[k];
	    dbuff[i]-=L_valuesR[k]*dbuff[j];
	} /* end for k */
    } /* end for i */
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: backward substitution done\n");fflush(stdout);
#endif

    
    /* permutation + diagonal scaling */
    for (i=0; i<n; i++) {
        /* the input vector counts permutations from 1,...,n */
        j=pPvecr[i]-1;
        pzr[j]=pSvecr[i]*dbuff[i];
    } /* end for i */
#ifdef PRINT_INFO
    mexPrintf("DSYMLDLsol: permutation + diagonal scaling done\n");fflush(stdout);
#endif

    mxFree(dbuff);
    
    return;
}
