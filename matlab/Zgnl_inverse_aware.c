/* $Id: Zgnl_inverse_aware.c 802 2015-10-19 16:57:31Z bolle $ */
/* ========================================================================== */
/* === Zgnl_inverse_aware mexFunction =========================================== */
/* ========================================================================== */

/*
    Usage:

    Return unit lower triangular LL,UUT with a few artificial nonzeros
    
    Example:

    % for initializing parameters
    [LL,UUT]=Zgnl_inverse_aware(L,D,U,ndense)


    Authors:

	Matthias Bollhoefer, TU Braunschweig

    Date:

	October 06, 2015. ILUPACK V2.5.  

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
#define ELBOW 5
#define RM    -2.2251e-308
/* #define SORT_ENTRIES */

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
    mxArray         *L_input, *D_input, *U_input, *ndense_input, *LL_output, *UUT_output,
                    *transpose_input;
    integer         i,j,k,l,m,n,nz,p,q,r,s,ndense, *ibuff,
                    **LL, *LLi, *LLj, *nLL, *eLL, **UUT, *UUTi, *UUTj, *nUUT, *eUUT;
    char            *transposename;
    doubleprecision *pr, *dbuff;
    size_t          mrows, ncols;
    double          *L_valuesR, *LL_valuesR, *U_valuesR, *UT_valuesR, *UUT_valuesR,
                    *L_valuesI, *LL_valuesI, *U_valuesI, *UT_valuesI, *UUT_valuesI;
    mwIndex         *L_ja, *U_ja, *UT_ja,  /* row indices of input matrix L,U^T       */
                    *L_ia, *U_ia, *UT_ia,  /* column pointers of input matrix L,U^T   */
                    *LL_ja, *UUT_ja,       /* row indices of output matrix LL,UUT     */
                    *LL_ia, *UUT_ia,       /* column pointers of output matrix LL,UUT */
                    *D_ja,                 /* row indices of input matrix D           */
                    *D_ia;                 /* column pointers of input matrix D       */
    

    if (nrhs!=5)
       mexErrMsgTxt("Four input arguments are required.");
    else if (nlhs!=2)
       mexErrMsgTxt("wrong number of output arguments.");
    else if (!mxIsNumeric(prhs[0]))
       mexErrMsgTxt("First input must be a matrix.");
    else if (!mxIsNumeric(prhs[1]))
       mexErrMsgTxt("Second input must be a matrix.");
    else if (!mxIsNumeric(prhs[2]))
       mexErrMsgTxt("Third input must be a number.");
    else if (!mxIsNumeric(prhs[3]))
       mexErrMsgTxt("Fourth input must be a number.");
    else if (mxIsNumeric(prhs[4]))
       mexErrMsgTxt("Fifth input must be a character.");

    /* The first input must be a square sparse unit lower triangular matrix.*/
    L_input=(mxArray *)prhs[0];
    /* get size of input matrix L */
    mrows=mxGetM(L_input);
    ncols=mxGetN(L_input);
    if (mrows!=ncols) {
       mexErrMsgTxt("First input must be a square matrix.");
    }
    if (!mxIsSparse (L_input)) {
        mexErrMsgTxt ("First input matrix must be in sparse format.") ;
    }
    n=mrows;
    L_ja     =(mwIndex *)mxGetIr(L_input);
    L_ia     =(mwIndex *)mxGetJc(L_input);
    L_valuesR=(double *) mxGetPr(L_input);
    L_valuesI=(double *) mxGetPi(L_input);
#ifdef PRINT_INFO
    mexPrintf("Zgnl_inverse_aware: input parameter L imported\n");fflush(stdout);
#endif

    /* The second input must be a square matrix.*/
    D_input=(mxArray *)prhs[1];
    /* get size of input matrix D */
    mrows=mxGetM(D_input);
    ncols=mxGetN(D_input);
    if (mrows!=ncols || mrows!=n) {
       mexErrMsgTxt("Second input must be a square matrix of same size as the first matrix.");
    }
    if (!mxIsSparse (D_input)) {
       mexErrMsgTxt ("Second input matrix must be in sparse format.") ;
    }
    D_ja     =(mwIndex *)mxGetIr(D_input);
    D_ia     =(mwIndex *)mxGetJc(D_input);
#ifdef PRINT_INFO
    mexPrintf("Zgnl_inverse_aware: input parameter D imported\n");fflush(stdout);
#endif

    /* The fifth input must be a number */
    transpose_input=(mxArray *)prhs[4];
    /* get size of input ndense */
    mrows=mxGetM(transpose_input);
    ncols=mxGetN(transpose_input);
    /* Allocate memory for input and output strings. */
    transposename = (char *)mxCalloc((size_t)(mrows*ncols+1), (size_t)sizeof(char));

    /* Copy the string data from tmp into a C string pdata */
    i=mxGetString(transpose_input, transposename, (mrows*ncols+1));
#ifdef PRINT_INFO
    mexPrintf("Zgnl_inverse_aware: input parameter transpose imported\n");fflush(stdout);
#endif
    
    /* The third input must be a square sparse unit upper triangular matrix.*/
    U_input=(mxArray *)prhs[2];
    /* get size of input matrix U */
    mrows=mxGetM(U_input);
    ncols=mxGetN(U_input);
    if (mrows!=ncols || mrows!=n) {
       mexErrMsgTxt("Third input must be a square matrix.");
    }
    if (!mxIsSparse (U_input)) {
        mexErrMsgTxt ("Third input matrix must be in sparse format.") ;
    }
    n=mrows;
    U_ja     =(mwIndex *)mxGetIr(U_input);
    U_ia     =(mwIndex *)mxGetJc(U_input);
    U_valuesR=(double *) mxGetPr(U_input);
    U_valuesI=(double *) mxGetPi(U_input);
#ifdef PRINT_INFO
    mexPrintf("Zgnl_inverse_aware: input parameter U imported\n");fflush(stdout);
#endif


    ibuff=(integer *) MAlloc((size_t)n*sizeof(integer),  "Zgnl_inverse_aware:ibuff");
    for (i=0; i<n; i++) {
        ibuff[i]=0;
    }

    if (transposename[0]=='t' || transposename[0]=='T') {
       UT_ia=U_ia;
       UT_ja=U_ja;
       UT_valuesR=U_valuesR;
       UT_valuesI=U_valuesI;
    }
    else {
       /* compute transposed matrix U^T */
       UT_ia=(integer *)MAlloc((size_t)(n+1)*sizeof(integer *),"Zgnl_inverse_aware:UT_ia");
       nz=U_ia[n];
       UT_ja=(integer *)MAlloc((size_t)nz*sizeof(integer *),"Zgnl_inverse_aware:UT_ja");
       UT_valuesR=(double *)MAlloc((size_t)nz*sizeof(double *),"Zgnl_inverse_aware:UT_valuesR");
       UT_valuesI=(double *)MAlloc((size_t)nz*sizeof(double *),"Zgnl_inverse_aware:UT_valuesI");
       /* integer buffer of size n */
       for (i=0; i<n; i++) {
           for (j=U_ia[i]; j<U_ia[i+1]; j++) {
  	       /* row index k of (k,i) */
	       k=U_ja[j];
	       /* increase number of nz in row k */
	       ibuff[k]++;
	   }
       }
#ifdef PRINT_INFO
       mexPrintf("nz of each column of U^T computed\n");fflush(stdout);
#endif
       /* create logical pointer structure for U^T */
       UT_ia[0]=0;
       for (i=0; i<n; i++) {
	   UT_ia[i+1]=UT_ia[i]+ibuff[i];
	   ibuff[i]=0;
       }
#ifdef PRINT_INFO
       mexPrintf("logical pointer structure of U^T set up\n");fflush(stdout);
#endif
       for (i=0; i<n; i++) {
	   for (j=U_ia[i]; j<U_ia[i+1]; j++) {
	       /* row index k of (k,i) */
	       k=U_ja[j];
	       /* number of nz in U^T(:,k) that have already been inserted */
	       l=ibuff[k];
	       /* address in column k of U^T where to insert the next entry */
	       m=UT_ia[k]+l;
	       /* row index i of (i,k) w.r.t. U^T */
	       UT_ja[m]=i;
	       /* associated numerical value */
	       UT_valuesR[m]=U_valuesR[j];
	       UT_valuesI[m]=U_valuesI[j];
	       /* increment nz of U^T(:,k) */
	       ibuff[k]++;
	   }
       }
#ifdef PRINT_INFO
       mexPrintf("U^T computation completed\n");fflush(stdout);
#endif
    }
    
    /* The fourth input must be a number */
    ndense_input=(mxArray *)prhs[3];
    /* get size of input ndense */
    mrows=mxGetM(ndense_input);
    ncols=mxGetN(ndense_input);
    if (mrows!=1 || ncols!=1) {
       mexErrMsgTxt("Third input must be a scalar.");
    }
    /* starting block with dense columns, convert to C-style */
    ndense=*mxGetPr(ndense_input)-1;
#ifdef PRINT_INFO
    mexPrintf("Zgnl_inverse_aware: input parameter ndense imported\n");fflush(stdout);
#endif
    
    /* pointer to list of nonzero entries of L, column-by-column */
    LL   =(integer **)MAlloc((size_t)n*sizeof(integer *),"Zgnl_inverse_aware:LL");
    nLL  =(integer *) MAlloc((size_t)n*sizeof(integer),  "Zgnl_inverse_aware:nLL");
    /* elbow buffer space */ 
    eLL  =(integer *) MAlloc((size_t)n*sizeof(integer),  "Zgnl_inverse_aware:nLL");
    /* initially use pattern of L */
    for (i=0; i<n; i++) {
        /* start of column i, C-style */
        k=L_ia[i];
        /* number of nonzeros in L(:,i) */
        nLL[i]=L_ia[i+1]-k;
	/* allocate slightly more memory to reduce the number of physical reallocations */
	eLL[i]=MAX(nLL[i]+ELBOW,nLL[i]*1.1);
	/* provide memory for nonzeros of column i */
	LL[i]=(integer *) MAlloc((size_t)eLL[i]*sizeof(integer),"Zgnl_inverse_aware:LL[i]");
        /* copy indices */
	memcpy(LL[i], L_ja+k, (size_t)nLL[i]*sizeof(integer));
#ifdef SORT_ENTRIES
	/* sort indices of in increasing order */
	qqsorti(LL[i],ibuff,nLL+i);
#endif
    } /* end for k */
#ifdef PRINT_INFO
    for (i=0; i<n; i++) {
        mexPrintf("col %3d:",i+1);
        for (k=0; k<nLL[i]; k++) 
	    mexPrintf("%4d",LL[i][k]+1);
	mexPrintf("\n");
	fflush(stdout);
    }
#endif

    /* pointer to list of nonzero entries of U^T, column-by-column */
    UUT =(integer **)MAlloc((size_t)n*sizeof(integer *),"Zgnl_inverse_aware:UUT");
    nUUT=(integer *) MAlloc((size_t)n*sizeof(integer),  "Zgnl_inverse_aware:nUUT");
    /* elbow buffer space */ 
    eUUT=(integer *) MAlloc((size_t)n*sizeof(integer),  "Zgnl_inverse_aware:eUUT");
    /* initially use pattern of U^T */
    for (i=0; i<n; i++) {
        /* start of column i, C-style */
        k=UT_ia[i];
        /* number of nonzeros in U^T(:,i) */
        nUUT[i]=UT_ia[i+1]-k;
	/* allocate slightly more memory to reduce the number of physical reallocations */
	eUUT[i]=MAX(nUUT[i]+ELBOW,nUUT[i]*1.1);
	/* provide memory for nonzeros of column i */
	UUT[i]=(integer *) MAlloc((size_t)eUUT[i]*sizeof(integer),"Zgnl_inverse_aware:UUT[i]");
        /* copy indices */
	memcpy(UUT[i], UT_ja+k, (size_t)nUUT[i]*sizeof(integer));
#ifdef SORT_ENTRIES
	/* sort indices of in increasing order */
	qqsorti(UUT[i],ibuff,nUUT+i);
#endif
    } /* end for k */
#ifdef PRINT_INFO
    for (i=0; i<n; i++) {
        mexPrintf("col %3d:",i+1);
        for (k=0; k<nUUT[i]; k++) 
	    mexPrintf("%4d",UUT[i][k]+1);
	mexPrintf("\n");
	fflush(stdout);
    }
#endif
    /* compression by advancing dense columns */
    i=ndense-1;
    while (i>=0) {
          if (nLL[i]==n-i && nUUT[i]==n-i) {
	     ndense--;
	     i--;
	  }
	  else
	     i=-1;
    } /* end while */
#ifdef PRINT_INFO
    mexPrintf("ndense=%d\n",ndense+1);
    fflush(stdout);
#endif

    
    /* scan pattern and augment it */
    i=0;
    while (i<n) {
          /* 2x2 case */
          if (D_ia[i+1]-D_ia[i]>1) {
	     /* initially make sure that column i and i+1 share the same nonzero pattern */
	     /* L(i:i+1,i:i+1)=I must be fulfilled! */
	     r=1; /* skip diagonal entry i   */
	     s=1; /* skip diagonal entry i+1 */
	     m=0; /* counter for the buffer */
	     /* reference to the nonzero indices of column i */
	     LLi=LL[i];
	     /* safeguard the case that L(i+1,i)~=0, which must not be the case! */
	     if (r<nLL[i])
	        if (LLi[r]==i+1)
		   r=2;
	     /* reference to the nonzero indices of column i+1 */
	     LLj=LL[i+1];
	     /* scan nonzero patterns of column i and i+1 exluding i,i+1 */
	     while (r<nLL[i] && s<nLL[i+1]) {
	           p=LLi[r];
		   q=LLj[s];
		   if (p<q) {
		      /* copy i's index */
		      ibuff[m++]=p;
		      r++;
		   }
		   else if (q<p) {
		      /* copy i+1's index */
		      ibuff[m++]=q;
		      s++;
		   }
		   else {
		      /* copy joint index */
		      ibuff[m++]=q;
		      r++;
		      s++;
		   }
	     } /* end while */
	     while (r<nLL[i]) {
		   /* copy i-index */
	           ibuff[m++]=LLi[r];
		   r++;
	     } /* end while */
	     while (s<nLL[i+1]) {
		   /* copy i+1 index */
		   ibuff[m++]=LLj[s];
		   s++;
	     } /* end while */
	     /* safeguard the case that L(i+1,i)~=0, which must not be the case! */
	     r=1;
	     if (r<nLL[i])
	        if (LLi[r]==i+1)
		   r=2;
	     if (m+r>nLL[i]) {
	        /* increase memory if necessary */
	        eLL[i]  =MAX(eLL[i],  m+r);
		LL[i]   =(integer *)ReAlloc(LL[i],  (size_t)eLL[i]  *sizeof(integer),
					    "Zgnl_inverse_aware:LL[i]");
		/* new number of row indices in columns i */
		nLL[i]=m+r;
		/* copy merged index array back */
		memcpy(LL[i]+r,   ibuff, (size_t)m*sizeof(integer));
	     }
	     if (m+1>nLL[i+1]) {
	        /* increase memory if necessary */
		eLL[i+1]=MAX(eLL[i+1],m+1);
		LL[i+1] =(integer *)ReAlloc(LL[i+1],(size_t)eLL[i+1]*sizeof(integer),
					    "Zgnl_inverse_aware:LL[i+1]");
		/* new number of row indices in columns i+1 */
		nLL[i+1]=m+1;
		/* copy merged index array back */
		memcpy(LL[i+1]+1, ibuff, (size_t)m*sizeof(integer));
	     }

	     /* initially make sure that column i and i+1 share the same nonzero pattern */
	     /* UT(i:i+1,i:i+1)=I must be fulfilled! */
	     r=1; /* skip diagonal entry i   */
	     s=1; /* skip diagonal entry i+1 */
	     m=0; /* counter for the buffer */
	     /* reference to the nonzero indices of column i */
	     UUTi=UUT[i];
	     /* safeguard the case that UT(i+1,i)~=0, which must not be the case! */
	     if (r<nUUT[i])
	        if (UUTi[r]==i+1)
		   r=2;
	     /* reference to the nonzero indices of column i+1 */
	     UUTj=UUT[i+1];
	     /* scan nonzero patterns of column i and i+1 exluding i,i+1 */
	     while (r<nUUT[i] && s<nUUT[i+1]) {
	           p=UUTi[r];
		   q=UUTj[s];
		   if (p<q) {
		      /* copy i's index */
		      ibuff[m++]=p;
		      r++;
		   }
		   else if (q<p) {
		      /* copy i+1's index */
		      ibuff[m++]=q;
		      s++;
		   }
		   else {
		      /* copy joint index */
		      ibuff[m++]=q;
		      r++;
		      s++;
		   }
	     } /* end while */
	     while (r<nUUT[i]) {
		   /* copy i-index */
	           ibuff[m++]=UUTi[r];
		   r++;
	     } /* end while */
	     while (s<nUUT[i+1]) {
		   /* copy i+1 index */
		   ibuff[m++]=UUTj[s];
		   s++;
	     } /* end while */
	     /* safeguard the case that UT(i+1,i)~=0, which must not be the case! */
	     r=1;
	     if (r<nUUT[i])
	        if (UUTi[r]==i+1)
		   r=2;
	     if (m+r>nUUT[i]) {
	        /* increase memory if necessary */
	        eUUT[i]  =MAX(eUUT[i],  m+r);
		UUT[i]   =(integer *)ReAlloc(UUT[i],  (size_t)eUUT[i]  *sizeof(integer),
					    "Zgnl_inverse_aware:UUT[i]");
		/* new number of row indices in columns i */
		nUUT[i]=m+r;
		/* copy merged index array back */
		memcpy(UUT[i]+r,   ibuff, (size_t)m*sizeof(integer));
	     }
	     if (m+1>nUUT[i+1]) {
	        /* increase memory if necessary */
		eUUT[i+1]=MAX(eUUT[i+1],m+1);
		UUT[i+1] =(integer *)ReAlloc(UUT[i+1],(size_t)eUUT[i+1]*sizeof(integer),
					    "Zgnl_inverse_aware:UUT[i+1]");
		/* new number of row indices in columns i+1 */
		nUUT[i+1]=m+1;
		/* copy merged index array back */
		memcpy(UUT[i+1]+1, ibuff, (size_t)m*sizeof(integer));
	     }
	  } /* end 2x2 case */

#ifdef PRINT_INFO
          if (D_ia[i+1]-D_ia[i]>1) {
	     mexPrintf("augmented patterns\n");
	     fflush(stdout);
	     mexPrintf("LL, col %3d:",i+1);
	     for (k=0; k<nLL[i]; k++) 
	         mexPrintf("%4d",LL[i][k]+1);
	     mexPrintf("\n");
	     fflush(stdout);
	     mexPrintf("LL, col %3d:",i+2);
	     for (k=0; k<nLL[i+1]; k++) 
	         mexPrintf("%4d",LL[i+1][k]+1);
	     mexPrintf("\n");
	     fflush(stdout);
	     mexPrintf("UUT, col %3d:",i+1);
	     for (k=0; k<nUUT[i]; k++) 
	         mexPrintf("%4d",UUT[i][k]+1);
	     mexPrintf("\n");
	     fflush(stdout);
	     mexPrintf("col %3d:",i+2);
	     for (k=0; k<nUUT[i+1]; k++) 
	         mexPrintf("%4d",UUT[i+1][k]+1);
	     mexPrintf("\n");
	     fflush(stdout);
	  }
	  else {
	     mexPrintf("augmented pattern\n");
	     fflush(stdout);
	     mexPrintf("LL, col %3d:",i+1);
	     for (k=0; k<nLL[i]; k++) 
	         mexPrintf("%4d",LL[i][k]+1);
	     mexPrintf("\n");
	     fflush(stdout);
	     mexPrintf("UUT, col %3d:",i+1);
	     for (k=0; k<nUUT[i]; k++) 
	         mexPrintf("%4d",UUT[i][k]+1);
	     mexPrintf("\n");
	     fflush(stdout);
	  }
#endif
	  
	  /* scan columns associated with nonzeros UT(:,i), but skip
	     i and stop as soon as ndense is reached, be aware of C-style! 
	  */
	  UUTi=UUT[i];
	  /* skip diagonal index i */
	  k=1;
	  /* safeguard the 2x2 case that UT(i+1,i)~=0, which must not be the case! */
          if (D_ia[i+1]-D_ia[i]>1)
	     if (k<nUUT[i])
	        if (UUTi[k]==i+1)
		   k=2;
	  while (k<nUUT[i]) {
	        j=UUTi[k];
		/* dense lower triangular block reached */
		if (j>=ndense)
		   k=nUUT[i];
		else { /* scan column j and check for additional fill */
		   r=k+1; /* we do not need to check U^T(j,i) */
		   s=1;   /* skip L(j,j) */
		   m=0;   /* counter for auxiliary buffer */
		   /* reference to the nonzero indices of column j>i */
		   LLj=LL[j];
		   /* column j is not yet dense */
		   if (nLL[j]<n-j) {
#ifdef PRINT_INFO
		      mexPrintf("scanning column %d, nz=%d\n",j+1,nLL[j]);
		      fflush(stdout);
#endif
		      while (r<nUUT[i] && s<nLL[j]) {
			    p=UUTi[r];
			    q=LLj[s];
			    if (p<q) {
			       /* copy fill-index */
			       ibuff[m++]=p;
			       r++;
			    }
			    else if (q<p) {
			       /* copy original index */
			       ibuff[m++]=q;
			       s++;
			    }
			    else {
			       /* copy joint index */
			       ibuff[m++]=q;
			       r++;
			       s++;
			    }
		      } /* end while */
		      while (r<nUUT[i]) {
			    /* copy fill-index */
			    ibuff[m++]=UUTi[r];
			    r++;
		      } /* end while */
		      while (s<nLL[j]) {
			    /* copy original index */
			    ibuff[m++]=LLj[s];
			    s++;
		      } /* end while */
		      /* did we encounter fill-in? */
		      if (m+1>nLL[j]) {
			 /* increase memory if necessary */
			 eLL[j]=MAX(eLL[j],m+1);
			 LL[j]=(integer *)ReAlloc(LL[j],(size_t)eLL[j]*sizeof(integer),
						  "Zgnl_inverse_aware:LL[j]");
			 /* new number of row indices in column j */
			 nLL[j]=m+1;
			 /* copy merged index array back */
			 memcpy(LL[j]+1, ibuff, (size_t)m*sizeof(integer));
		      }
		   } /* end if (not dense column j) */
		   else {
#ifdef PRINT_INFO
		      mexPrintf("dense column %d\n",j+1);
		      fflush(stdout);
#endif
		   }

		   /* start of dense columns detected? */
		   if (nLL[j]==n-j && nUUT[j]==n-j && j==ndense-1) {
		      ndense--;
#ifdef PRINT_INFO
		      mexPrintf("ndense=%d\n",ndense+1);
		      fflush(stdout);
#endif
		   }		     
		} /* end else */
		k++;
	  } /* end while */

	  
	  /* scan columns associated with nonzeros L(:,i), but skip
	     i and stop as soon as ndense is reached, be aware of C-style! 
	  */
	  LLi=LL[i];
	  /* skip diagonal index i */
	  k=1;
	  /* safeguard the 2x2 case that L(i+1,i)~=0, which must not be the case! */
          if (D_ia[i+1]-D_ia[i]>1)
	     if (k<nLL[i])
	        if (LLi[k]==i+1)
		   k=2;
	  while (k<nLL[i]) {
	        j=LLi[k];
		/* dense lower triangular block reached */
		if (j>=ndense)
		   k=nLL[i];
		else { /* scan column j and check for additional fill */
		   r=k+1; /* we do not need to check L(j,i) */
		   s=1;   /* skip UT(j,j) */
		   m=0;   /* counter for auxiliary buffer */
		   /* reference to the nonzero indices of column j>i */
		   UUTj=UUT[j];
		   /* column j is not yet dense */
		   if (nUUT[j]<n-j) {
#ifdef PRINT_INFO
		      mexPrintf("scanning column %d, nz=%d\n",j+1,nUUT[j]);
		      fflush(stdout);
#endif
		      while (r<nLL[i] && s<nUUT[j]) {
			    p=LLi[r];
			    q=UUTj[s];
			    if (p<q) {
			       /* copy fill-index */
			       ibuff[m++]=p;
			       r++;
			    }
			    else if (q<p) {
			       /* copy original index */
			       ibuff[m++]=q;
			       s++;
			    }
			    else {
			       /* copy joint index */
			       ibuff[m++]=q;
			       r++;
			       s++;
			    }
		      } /* end while */
		      while (r<nLL[i]) {
			    /* copy fill-index */
			    ibuff[m++]=LLi[r];
			    r++;
		      } /* end while */
		      while (s<nUUT[j]) {
			    /* copy original index */
			    ibuff[m++]=UUTj[s];
			    s++;
		      } /* end while */
		      /* did we encounter fill-in? */
		      if (m+1>nUUT[j]) {
			 /* increase memory if necessary */
			 eUUT[j]=MAX(eUUT[j],m+1);
			 UUT[j]=(integer *)ReAlloc(UUT[j],(size_t)eUUT[j]*sizeof(integer),
						  "Zgnl_inverse_aware:UUT[j]");
			 /* new number of row indices in column j */
			 nUUT[j]=m+1;
			 /* copy merged index array back */
			 memcpy(UUT[j]+1, ibuff, (size_t)m*sizeof(integer));
		      }
		   } /* end if (not dense column j) */
		   else {
#ifdef PRINT_INFO
		      mexPrintf("dense column %d\n",j+1);
		      fflush(stdout);
#endif
		   }

		   /* start of dense columns detected? */
		   if (nUUT[j]==n-j && nLL[j]==n-j && j==ndense-1) {
		      ndense--;
#ifdef PRINT_INFO
		      mexPrintf("ndense=%d\n",ndense+1);
		      fflush(stdout);
#endif
		   }		     
		} /* end else */
		k++;
	  } /* end while */

	  
	  /* advance one column more in the 2x2 case */
	  if (D_ia[i+1]-D_ia[i]==1)
	     i++;
	  else
	     i+=2;
    } /* end for i */

    
#ifdef SORT_ENTRIES
    /* double buffer of size n */
    dbuff=(doubleprecision *) MAlloc((size_t)n*sizeof(doubleprecision),
				     "Zgnl_inverse_aware:dbuff");
#endif
    
    /* export augmented L */
    /* first compute total number of nonzeros */
    nz=0;
    for (i=0; i<n; i++) {
        nz+=nLL[i];
    } /* end for i */
    plhs[0]=mxCreateSparse((mwSize)n,(mwSize)n, (mwSize)nz, mxCOMPLEX);
    LL_output =(mxArray *)plhs[0];
    LL_ja     =(mwIndex *)mxGetIr(LL_output);
    LL_ia     =(mwIndex *)mxGetJc(LL_output);
    LL_valuesR=(double *) mxGetPr(LL_output);
    LL_valuesI=(double *) mxGetPi(LL_output);

    LL_ia[0]=0;
    for (i=0; i<n; i++) {
        j=L_ia[i];
#ifdef SORT_ENTRIES
	m=L_ia[i+1]-j;
	memcpy(ibuff,L_ja+j,     (size_t)m*sizeof(integer));
	memcpy(dbuff,L_valuesR+j,(size_t)m*sizeof(doubleprecision));
	/* sort indices of in increasing order */
	/* we do not need any further re-allocation for LL, use eLL as stack */
	Dqsort(dbuff,ibuff,eLL,&m);
	j=0;
#endif
        k=0;
	m=LL_ia[i];
	while (j<L_ia[i+1]) {
	      l=LL[i][k];
	      LL_ja[m]=l;
	      /* indices match, original entry */
#ifdef SORT_ENTRIES
	      if (l==ibuff[j]) {
		 LL_valuesR[m]=dbuff[j];
		 j++;
	      }
#else
	      if (l==L_ja[j]) {
		 LL_valuesR[m]=L_valuesR[j];
		 LL_valuesI[m]=L_valuesI[j];
		 j++;
	      }
#endif
	      else { /* the row indices of L MUST be a subset of that of LL 
		        Thus this must be fill-in */
		 LL_valuesR[m]=RM;
		 LL_valuesI[m]=0.0;
	      }
	      k++;
	      m++;
        } /* end while */
	/* remaining fill-in */
	while (k<nLL[i]) {
	      l=LL[i][k];
	      LL_ja[m]=l;
	      LL_valuesR[m]=RM;
	      LL_valuesI[m]=0.0;
	      m++;
	      k++;
        } /* end while */
	LL_ia[i+1]=LL_ia[i]+nLL[i];

	/* give away index memory for column i */
	FRee(LL[i]);
    } /* end for i */



    /* export augmented UT~U^T */
    /* first compute total number of nonzeros */
    nz=0;
    for (i=0; i<n; i++) {
        nz+=nUUT[i];
    } /* end for i */
    plhs[1]=mxCreateSparse((mwSize)n,(mwSize)n, (mwSize)nz, mxCOMPLEX);
    UUT_output =(mxArray *)plhs[1];
    UUT_ja     =(mwIndex *)mxGetIr(UUT_output);
    UUT_ia     =(mwIndex *)mxGetJc(UUT_output);
    UUT_valuesR=(double *) mxGetPr(UUT_output);
    UUT_valuesI=(double *) mxGetPi(UUT_output);

    UUT_ia[0]=0;
    for (i=0; i<n; i++) {
        j=UT_ia[i];
        k=0;
	m=UUT_ia[i];
	while (j<UT_ia[i+1]) {
	      l=UUT[i][k];
	      UUT_ja[m]=l;
	      if (l==UT_ja[j]) {
		 UUT_valuesR[m]=UT_valuesR[j];
		 UUT_valuesI[m]=UT_valuesI[j];
		 j++;
	      }
	      else { /* the row indices of UT MUST be a subset of that of UUT
		        Thus this must be fill-in */
		 UUT_valuesR[m]=RM;
		 UUT_valuesI[m]=0.0;
	      }
	      k++;
	      m++;
        } /* end while */
	/* remaining fill-in */
	while (k<nUUT[i]) {
	      l=UUT[i][k];
	      UUT_ja[m]=l;
	      UUT_valuesR[m]=RM;
	      UUT_valuesI[m]=0.0;
	      m++;
	      k++;
        } /* end while */
	UUT_ia[i+1]=UUT_ia[i]+nUUT[i];

	/* give away index memory for column i */
	FRee(UUT[i]);
    } /* end for i */
    if (transposename[0]!='t' && transposename[0]!='T') {
       free(UT_ia);
       free(UT_ja);
       free(UT_valuesR);
       free(UT_valuesI);
    }

    
    /* release memory */
    FRee(ibuff);
    FRee(LL);
    FRee(nLL);
    FRee(eLL);
    FRee(UUT);
    FRee(nUUT);
    FRee(eUUT);
#ifdef SORT_ENTRIES
    FRee(dbuff);
#endif    

#ifdef PRINT_INFO
    mexPrintf("Zgnl_inverse_aware: memory released\n");fflush(stdout);
#endif
    
    return;
}
