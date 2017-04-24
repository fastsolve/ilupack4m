/* $Id: ZGNLLDUsol.c 799 2015-08-12 14:52:16Z bolle $ */
/* ========================================================================== */
/* === ZGNLLDUhsol mexFunction =========================================== */
/* ========================================================================== */

/*
    Usage:

    Solve (Sl^{-1}P^T (LDU^T) PSr^{-1})^* z=y, where Sl,Sr are diagonal scaling matrice,
    P is a permutation, and LDU^T is a triangular factorization with 1x1 and 2x2 pivots
    
    Example:

    % for initializing parameters
    z=ZGNLLDUhsol(y,Pvec,Slvec,Srvec,L,iD,UT)


    Authors:

	Matthias Bollhoefer, TU Braunschweig

    Date:

	October 16, 2015. ILUPACK V2.5.  

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
    mxArray         *L_input, *iD_input, *UT_input, *Pvec_input, *Slvec_input, *Srvec_input,
                    *y_input, *z_output;
    integer         i,j,k,l,n;
    doubleprecision  *pPvecr, *pSlvecr, *pSrvecr, *pzr, *pyr, *dbuffr, xr, yr, 
                     *pPveci, *pSlveci, *pSrveci, *pzi, *pyi, *dbuffi, xi, yi;
    size_t          mrows, ncols;
    double          *L_valuesR, *iD_valuesR, *UT_valuesR, *L_valuesI, *iD_valuesI, *UT_valuesI;
    mwIndex         *L_ja, *UT_ja,         /* row indices of input matrix L,U^T     */
                    *L_ia, *UT_ia,         /* column pointers of input matrix L,U^T */
                    *iD_ja,                /* row indices of input matrix iD        */
                    *iD_ia;                /* column pointers of input matrix iD    */
    

    if (nrhs!=7)
       mexErrMsgTxt("Seven input arguments are required.");
    else if (nlhs!=1)
       mexErrMsgTxt("wrong number of output arguments.");
    else if (!mxIsNumeric(prhs[0]))
       mexErrMsgTxt("First input must be a vector.");
    else if (!mxIsNumeric(prhs[1]))
       mexErrMsgTxt("Second input must be a vector.");
    else if (!mxIsNumeric(prhs[2]))
       mexErrMsgTxt("Third input must be a vector.");
    else if (!mxIsNumeric(prhs[3]))
       mexErrMsgTxt("Fourth input must be a vector.");
    else if (!mxIsNumeric(prhs[4]))
       mexErrMsgTxt("Fifth input must be a matrix.");
    else if (!mxIsNumeric(prhs[5]))
       mexErrMsgTxt("Sixth input must be a matrix.");
    else if (!mxIsNumeric(prhs[6]))
       mexErrMsgTxt("Seventh input must be a matrix.");

    
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
    pyi=(double *)mxGetPi(y_input);
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: input parameter y imported\n");fflush(stdout);
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
    pPveci=(double *)mxGetPi(Pvec_input);
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: input parameter Pvec imported\n");fflush(stdout);
#endif
    

    /* The third input must be a dense vector */
    Slvec_input=(mxArray *)prhs[2];
    /* get size of input vector y */
    mrows=mxGetM(Slvec_input);
    ncols=mxGetN(Slvec_input);
    if (ncols!=1 || mrows!=n) {
       mexErrMsgTxt("Third input must be a vector of same size as first");
    }
    if (mxIsSparse (Slvec_input)) {
        mexErrMsgTxt ("Third input vector must be in dense format.") ;
    }
    pSlvecr=(double *)mxGetPr(Slvec_input);
    pSlveci=(double *)mxGetPi(Slvec_input);
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: input parameter Slvec imported\n");fflush(stdout);
#endif

    /* The fourth input must be a dense vector */
    Srvec_input=(mxArray *)prhs[3];
    /* get size of input vector y */
    mrows=mxGetM(Srvec_input);
    ncols=mxGetN(Srvec_input);
    if (ncols!=1 || mrows!=n) {
       mexErrMsgTxt("Fourth input must be a vector of same size as first");
    }
    if (mxIsSparse (Srvec_input)) {
        mexErrMsgTxt ("Fourth input vector must be in dense format.") ;
    }
    pSrvecr=(double *)mxGetPr(Srvec_input);
    pSrveci=(double *)mxGetPi(Srvec_input);
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: input parameter Srvec imported\n");fflush(stdout);
#endif

    
    /* The fifth input must be a square sparse unit lower triangular matrix.*/
    L_input=(mxArray *)prhs[4];
    /* get size of input matrix L */
    mrows=mxGetM(L_input);
    ncols=mxGetN(L_input);
    if (mrows!=ncols || ncols!=n) {
       mexErrMsgTxt("Fifth input must be a square matrix of the same size as the other inputs.");
    }
    if (!mxIsSparse (L_input)) {
        mexErrMsgTxt ("Fifth input matrix must be in sparse format.") ;
    }
    L_ja     =(mwIndex *)mxGetIr(L_input);
    L_ia     =(mwIndex *)mxGetJc(L_input);
    L_valuesR=(double *) mxGetPr(L_input);
    L_valuesI=(double *) mxGetPi(L_input);
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: input parameter L imported\n");fflush(stdout);
#endif

    
    /* The sixth input must be a square block diagonal matrix.*/
    iD_input=(mxArray *)prhs[5];
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
    iD_valuesI=(double *) mxGetPi(iD_input);
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: input parameter iD imported\n");fflush(stdout);
#endif


    /* The seventh input must be a square sparse transposed unit upper triangular matrix.*/
    UT_input=(mxArray *)prhs[6];
    /* get size of input matrix U^T */
    mrows=mxGetM(UT_input);
    ncols=mxGetN(UT_input);
    if (mrows!=ncols || ncols!=n) {
       mexErrMsgTxt("Seventh input must be a square matrix of the same size as the other inputs.");
    }
    if (!mxIsSparse (UT_input)) {
        mexErrMsgTxt ("Seventh input matrix must be in sparse format.") ;
    }
    UT_ja     =(mwIndex *)mxGetIr(UT_input);
    UT_ia     =(mwIndex *)mxGetJc(UT_input);
    UT_valuesR=(double *) mxGetPr(UT_input);
    UT_valuesI=(double *) mxGetPi(UT_input);
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: input parameter UT imported\n");fflush(stdout);
#endif

    
    /* buffer vector */
    dbuffr=(double *)mxCalloc((size_t)n, (size_t)sizeof(double));
    dbuffi=(double *)mxCalloc((size_t)n, (size_t)sizeof(double));

    /* create output vector */
    z_output=mxCreateDoubleMatrix((mwSize)n,(mwSize)1, mxCOMPLEX);
    plhs[0]=z_output;
    pzr=(double *)mxGetPr(z_output);
    pzi=(double *)mxGetPi(z_output);
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: output vector and buffer allocated\n");fflush(stdout);
#endif

    /* permutation + transposed left diagonal scaling */
    if (pyi!=NULL) {
       for (i=0; i<n; i++) {
	   /* the input vector counts permutations from 1,...,n */
	   j=pPvecr[i]-1;
	   dbuffr[i]=pSrvecr[i]*pyr[j];
	   dbuffi[i]=pSrvecr[i]*pyi[j];
       } /* end for i */
    }
    else {
       for (i=0; i<n; i++) {
	   /* the input vector counts permutations from 1,...,n */
	   j=pPvecr[i]-1;
	   dbuffr[i]=pSrvecr[i]*pyr[j];
	   dbuffi[i]=0.0;
       } /* end for i */
    }
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: permutation + diagonal scaling done\n");fflush(stdout);
#endif

    
    /* forward substitution with conjugate transposed unit upper triangular matrix conj(UT)=U^* */
    if (UT_valuesI!=NULL) {
       for (i=0; i<n; i++) {
           /* downdate with strict lower triangular part excluding the diagonal part which is 1 */
#ifdef PRINT_CHECK
           if (UT_ia[i]>=UT_ia[i+1]) {
              mexPrintf("empty column %d\n",i+1);
	      fflush(stdout);
           }
	   else {
	      if (UT_ja[UT_ia[i]]!=i) {
	         mexPrintf("column %d does not start with diagonal entry\n",i+1);
	         fflush(stdout);
	      }
	   }
#endif
	   for (k=UT_ia[i]+1; k<UT_ia[i+1]; k++) {
	       /* row index */
	       j=UT_ja[k];
	       dbuffr[j]-=UT_valuesR[k]*dbuffr[i]+UT_valuesI[k]*dbuffi[i]; /* conjugate! */
	       dbuffi[j]-=UT_valuesR[k]*dbuffi[i]-UT_valuesI[k]*dbuffr[i]; /* conjugate! */
	   } /* end for k */
       } /* end for i */
    }
    else {
       for (i=0; i<n; i++) {
           /* downdate with strict lower triangular part excluding the diagonal part which is 1 */
#ifdef PRINT_CHECK
           if (UT_ia[i]>=UT_ia[i+1]) {
	      mexPrintf("empty column %d\n",i+1);
	      fflush(stdout);
           }
	   else {
	      if (UT_ja[UT_ia[i]]!=i) {
	         mexPrintf("column %d does not start with diagonal entry\n",i+1);
	         fflush(stdout);
	      }
	   }
#endif
	   for (k=UT_ia[i]+1; k<UT_ia[i+1]; k++) {
	       /* row index */
	       j=UT_ja[k];
	       dbuffr[j]-=UT_valuesR[k]*dbuffr[i];
	       dbuffi[j]-=UT_valuesR[k]*dbuffi[i];
	   } /* end for k */
       } /* end for i */
    }
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: forward substitution done\n");fflush(stdout);
#endif

    
    /* multiplication with a conjugate transposed block diagonal matrix with 1x1 and 2x2 pivots */
    if (iD_valuesI!=NULL) {
       i=0;
       j=iD_ia[0];
       while (i<n) {
	     k=iD_ia[i+1];
	     /* 2x2 pivot */
	     if (k-j>1) {
	        l=iD_ia[i+2];
		/* transpose 2x2 block! */
		if (l-k>1) { /* 2x2 pivot with four nonzero elements */
		   xr=iD_valuesR[j]  *dbuffr[i]  +iD_valuesI[j]  *dbuffi[i]     /* conjugate! */
		     +iD_valuesR[j+1]*dbuffr[i+1]+iD_valuesI[j+1]*dbuffi[i+1];  /* conjugate! */
		   yr=iD_valuesR[k]  *dbuffr[i]  +iD_valuesI[k]  *dbuffi[i]     /* conjugate! */
		     +iD_valuesR[k+1]*dbuffr[i+1]+iD_valuesI[k+1]*dbuffi[i+1];  /* conjugate! */
		   xi=iD_valuesR[j]  *dbuffi[i]  -iD_valuesI[j]  *dbuffr[i]     /* conjugate! */
		     +iD_valuesR[j+1]*dbuffi[i+1]-iD_valuesI[j+1]*dbuffr[i+1];  /* conjugate! */
		   yi=iD_valuesR[k]  *dbuffi[i]  -iD_valuesI[k]  *dbuffr[i]     /* conjugate! */
		     +iD_valuesR[k+1]*dbuffi[i+1]-iD_valuesI[k+1]*dbuffr[i+1];  /* conjugate! */
		}
		else { /* 2x2 pivot with three nonzero elements [ a_{ii}   a_{i,i+1};
		                                                  a_{i+1,i}    0     ] */
		   xr=iD_valuesR[j]  *dbuffr[i]  +iD_valuesI[j]  *dbuffi[i]      /* conjugate! */
		     +iD_valuesR[j+1]*dbuffr[i+1]+iD_valuesI[j+1]*dbuffi[i+1];   /* conjugate! */
		   yr=iD_valuesR[k]  *dbuffr[i]  +iD_valuesI[k]  *dbuffi[i];     /* conjugate! */
		   xi=iD_valuesR[j]  *dbuffi[i]  -iD_valuesI[j]  *dbuffr[i]      /* conjugate! */
		     +iD_valuesR[j+1]*dbuffi[i+1]-iD_valuesI[j+1]*dbuffr[i+1];   /* conjugate! */
		   yi=iD_valuesR[k]  *dbuffi[i]  -iD_valuesI[k]  *dbuffr[i];     /* conjugate! */
		}
		dbuffr[i]  =xr;
		dbuffr[i+1]=yr;
		dbuffi[i]  =xi;
		dbuffi[i+1]=yi;
		i+=2;
		j=iD_ia[i];
	     }
	     else {
	        if (iD_ja[j]!=i) { /* 2x2 pivot with a_{ii}=0 */
		   l=iD_ia[i+2];
		   if (l-k>1) { /* 2x2 pivot with three nonzero elements [   0      a_{i,i+1};
		                                                          a_{i+1,i} a_{i+1,i+1}] */
		      xr=
			 iD_valuesR[j]  *dbuffr[i+1]+iD_valuesI[j]  *dbuffi[i+1]; /* conjugate! */
		      yr=iD_valuesR[k]  *dbuffr[i]  +iD_valuesI[k]  *dbuffi[i]    /* conjugate! */
			+iD_valuesR[k+1]*dbuffr[i+1]+iD_valuesI[k+1]*dbuffi[i+1]; /* conjugate! */
		      xi=
			 iD_valuesR[j]  *dbuffi[i+1]-iD_valuesI[j]  *dbuffr[i+1]; /* conjugate! */
		      yi=iD_valuesR[k]  *dbuffi[i]  -iD_valuesI[k]  *dbuffr[i]    /* conjugate! */
			+iD_valuesR[k+1]*dbuffi[i+1]-iD_valuesI[k+1]*dbuffr[i+1]; /* conjugate! */
		   }
		   else { /* 2x2 pivot with two nonzero elements [   0      a_{i,i+1};
		                                                  a_{i+1,i}    0     ] */
		      xr=
			 iD_valuesR[j]  *dbuffr[i+1]+iD_valuesI[j]  *dbuffi[i+1];  /* conjugate! */
		      yr=iD_valuesR[k]  *dbuffr[i]  +iD_valuesI[k]  *dbuffi[i];    /* conjugate! */
		      xi=
			 iD_valuesR[j]  *dbuffi[i+1]-iD_valuesI[j]  *dbuffr[i+1];  /* conjugate! */
		      yi=iD_valuesR[k]  *dbuffi[i]  -iD_valuesI[k]  *dbuffr[i];    /* conjugate! */
		   }
		   dbuffr[i]  =xr;
		   dbuffr[i+1]=yr;
		   dbuffi[i]  =xi;
		   dbuffi[i+1]=yi;
		   i+=2;
		   j=iD_ia[i];
		}
		else { /* 1x1 pivot */
	           xr=iD_valuesR[j]*dbuffr[i]+iD_valuesI[j]*dbuffi[i]; /* conjugate! */
		   xi=iD_valuesR[j]*dbuffi[i]-iD_valuesI[j]*dbuffr[i]; /* conjugate! */
		   dbuffr[i]=xr;
		   dbuffi[i]=xi;
		   i++;
		   j=k;
		}
	     }
       } /* end while i */
    }
    else {
       i=0;
       j=iD_ia[0];
       while (i<n) {
	     k=iD_ia[i+1];
	     /* 2x2 pivot */
	     if (k-j>1) {
	        l=iD_ia[i+2];
		if (l-k>1) { /* 2x2 pivot with four nonzero elements */
		   xr=iD_valuesR[j]  *dbuffr[i]  
		     +iD_valuesR[j+1]*dbuffr[i+1];
		   yr=iD_valuesR[k]  *dbuffr[i]  
		     +iD_valuesR[k+1]*dbuffr[i+1];
		   xi=iD_valuesR[j]  *dbuffi[i]  
		     +iD_valuesR[j+1]*dbuffi[i+1];
		   yi=iD_valuesR[k]  *dbuffi[i]  
		     +iD_valuesR[k+1]*dbuffi[i+1];
		}
		else { /* 2x2 pivot with three nonzero elements [ a_{ii}   a_{i,i+1};
		                                                  a_{i+1,i}    0     ] */
		   xr=iD_valuesR[j]  *dbuffr[i]  
		     +iD_valuesR[j+1]*dbuffr[i+1];
		   yr=iD_valuesR[k]  *dbuffr[i]  ;
		   xi=iD_valuesR[j]  *dbuffi[i]  
		     +iD_valuesR[j+1]*dbuffi[i+1];
		   yi=iD_valuesR[k]  *dbuffi[i]  ;
		}
		dbuffr[i]  =xr;
		dbuffr[i+1]=yr;
		dbuffi[i]  =xi;
		dbuffi[i+1]=yi;
		i+=2;
		j=iD_ia[i];
	     }
	     else {
	        if (iD_ja[j]!=i) { /* 2x2 pivot with a_{ii}=0 */
		   l=iD_ia[i+2];
		   if (l-k>1) { /* 2x2 pivot with three nonzero elements [   0      a_{i,i+1};
		                                                          a_{i+1,i} a_{i+1,i+1}] */
		      xr=
			 iD_valuesR[j]  *dbuffr[i+1];
		      yr=iD_valuesR[k  ]*dbuffr[i]  
			+iD_valuesR[k+1]*dbuffr[i+1];
		      xi=
			 iD_valuesR[j]  *dbuffi[i+1];
		      yi=iD_valuesR[k]  *dbuffi[i]  
			+iD_valuesR[k+1]*dbuffi[i+1];
		   }
		   else { /* 2x2 pivot with two nonzero elements [   0      a_{i,i+1};
		                                                  a_{i+1,i}    0     ] */
		      xr=
			 iD_valuesR[j]  *dbuffr[i+1];
		      yr=iD_valuesR[k]  *dbuffr[i]  ;
		      xi=
			 iD_valuesR[j]  *dbuffi[i+1];
		      yi=iD_valuesR[k]  *dbuffi[i]  ;
		   }
		   dbuffr[i]  =xr;
		   dbuffr[i+1]=yr;
		   dbuffi[i]  =xi;
		   dbuffi[i+1]=yi;
		   i+=2;
		   j=iD_ia[i];
		}
		else { /* 1x1 pivot */
	           xr=iD_valuesR[j]*dbuffr[i];
		   xi=iD_valuesR[j]*dbuffi[i];
		   dbuffr[i]=xr;
		   dbuffi[i]=xi;
		   i++;
		   j=k;
		}
	     }
       } /* end while i */
    }
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: multiplication by block diagonal matrix done\n");fflush(stdout);
#endif


    /* back substitution with conjugate transposed unit lower triangular matrix L^* */
    if (L_valuesI!=NULL) {
       for (i=n-1; i>=0; i--) {
           /* scalar product with strict upper triangular part excluding the diagonal part which is 1 */
	   for (k=L_ia[i]+1; k<L_ia[i+1]; k++) {
	       /* row index */
	       j=L_ja[k];
	       dbuffr[i]-=L_valuesR[k]*dbuffr[j]+L_valuesI[k]*dbuffi[j];  /* conjugate! */
	       dbuffi[i]-=L_valuesR[k]*dbuffi[j]-L_valuesI[k]*dbuffr[j];  /* conjugate! */
	   } /* end for k */
       } /* end for i */
    }
    else {
       for (i=n-1; i>=0; i--) {
           /* scalar product with strict upper triangular part excluding the diagonal part which is 1 */
	   for (k=L_ia[i]+1; k<L_ia[i+1]; k++) {
	       /* row index */
	       j=L_ja[k];
	       dbuffr[i]-=L_valuesR[k]*dbuffr[j];
	       dbuffi[i]-=L_valuesR[k]*dbuffi[j];
	   } /* end for k */
       } /* end for i */
    }
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: backward substitution done\n");fflush(stdout);
#endif

    
    /* permutation + transposed right diagonal scaling */
    for (i=0; i<n; i++) {
        /* the input vector counts permutations from 1,...,n */
        j=pPvecr[i]-1;
        pzr[j]=pSlvecr[i]*dbuffr[i];
        pzi[j]=pSlvecr[i]*dbuffi[i];
    } /* end for i */
#ifdef PRINT_INFO
    mexPrintf("ZGNLLDUhsol: permutation + diagonal scaling done\n");fflush(stdout);
#endif

    mxFree(dbuffr);
    mxFree(dbuffi);
    
    return;
}
