/* $Id: DSYMselbinv.c 791 2015-08-05 08:51:07Z bolle $ */
/* ========================================================================== */
/* === DSYMselbinv mexFunction =============================================== */
/* ========================================================================== */

/*
    Usage:

    Return the structure 'options' and incomplete LDL^T preconditioner

    Example:

    % for initializing parameters
    [D, BLinv]=DSYMselbinv(BL,BD,perm, Delta)


    Authors:

	Matthias Bollhoefer, TU Braunschweig

    Date:

	March 25, 2015. ILUPACK V2.5.

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
#include <lapack.h>
#include <blas.h>

#define MAX_FIELDS 100
#define MAX(A,B) (((A)>=(B))?(A):(B))
#define MIN(A,B) (((A)>=(B))?(B):(A))
#define ELBOW    MAX(4.0,2.0)
/* #define PRINT_CHECK */
/* #define PRINT_INFO  */

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
    mwSize       dims[1];
    const char   *BLnames[]={"J","I", "L","D"};
    mxArray      *BL,
                    *BL_block,
                       *BL_blockJ, *BL_blockI, *BL_blockL, *BL_blockD,
                 *BD,
                    *BD_block,
                       *BD_blockD,
                 *BLinv,
                    *BLinv_block,
                       *BLinv_blockJ, *BLinv_blockI, *BLinv_blockL, *BLinv_blockD,
                    *BLinv_blocki,
                       *BLinv_blockJi, *BLinv_blockIi, *BLinv_blockLi, *BLinv_blockDi,
                 *Delta, *perm;

    integer      i,j,k,l,m,n,p,q,r,s,t,
                 ii,jj,flag, *ipiv, size_gemm_buff, sumn,
                 level3_BLAS, copy_cnt, *block, nblocks,
                 n_size, m_size, ni_size, mi_size,
                 i_first, j_first, k_first,
                 Ji_cont, Ik_cont, Ii_cont;
    double       val, alpha, beta, *Dbuff, *work, *gemm_buff,
                 *pr, *pr2, *pr3, *pr4,
                 *prBLJ,     *prBLI,     *prBLL,     *prBLD,
                                                     *prBDD,
                 *prBLinvJ,  *prBLinvI,  *prBLinvL,  *prBLinvD,
                 *prBLinvJi, *prBLinvIi, *prBLinvLi, *prBLinvDi;
    mwIndex      *ja, *ia;


    if (nrhs!=4)
       mexErrMsgTxt("Four input arguments required.");
    else if (nlhs!=2)
       mexErrMsgTxt("wrong number of output arguments.");


    /* The first input must be a cell array.*/
    BL=(mxArray *)prhs[0];
    /* get size of input cell array BL */
    nblocks=MAX(mxGetM(BL),mxGetN(BL));
#ifdef PRINT_CHECK
    if (mxGetM(BL)!=1 && mxGetN(BL)!=1) {
       mexPrintf("BL must be a 1-dim. cell array!\n");
       fflush(stdout);
    }
#endif
    if (!mxIsCell(BL)) {
       mexErrMsgTxt ("First input matrix must be a cell array.") ;
    }
#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: input parameter BL imported\n");fflush(stdout);
#endif


    /* The second input must be a cell array as well.*/
    BD=(mxArray *)prhs[1];
    if (!mxIsCell(BD)) {
       mexErrMsgTxt ("Second input matrix must be a cell array.") ;
    }
    /* get size of input matrix BD */
    if (MAX(mxGetM(BD),mxGetN(BD))!=nblocks) {
       mexErrMsgTxt("Second input must be a cell array of same size as the first input.");
    }
#ifdef PRINT_CHECK
    if (mxGetM(BD)!=1 && mxGetN(BD)!=1) {
       mexPrintf("BD must be a 1-dim. cell array!\n");
       fflush(stdout);
    }
#endif
#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: input parameter BD imported\n");fflush(stdout);
#endif


    /* The third input must be an "integer" vector */
    perm=(mxArray *)prhs[2];
    if (!mxIsNumeric(perm)) {
       mexErrMsgTxt ("Third input vector must be in dense format.");
    }
    /* get size of input vector */
    n=mxGetM(perm)*mxGetN(perm);
#ifdef PRINT_CHECK
    if (mxGetM(perm)!=1 && mxGetN(perm)!=1) {
       mexPrintf("perm must be a 1-dim. array!\n");
       fflush(stdout);
    }
#endif
#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: input parameter perm imported\n");fflush(stdout);
#endif


    /* The fourth input must be dense a vector */
    Delta=(mxArray *)prhs[3];
    if (!mxIsNumeric(Delta)) {
       mexErrMsgTxt("Fourth input vector must be in dense format.");
    }
    /* get size of input matrix Delta */
    if (MAX(mxGetM(Delta),mxGetN(Delta))!=n) {
       mexErrMsgTxt("Fourth argument must be a vector of same size as the third one.");
    }
#ifdef PRINT_CHECK
    if (mxGetM(Delta)!=1 && mxGetN(Delta)!=1) {
       mexPrintf("Delta must be a 1-dim. array!\n");
       fflush(stdout);
    }
#endif
#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: input parameter Delta imported\n");fflush(stdout);
#endif

#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: input parameters imported\n");fflush(stdout);
#endif



    /* create output cell array BLinv of length "nblocks" */
    dims[0]=nblocks;
    plhs[1]=mxCreateCellArray((mwSize)1, dims);
    BLinv=plhs[1];


    /* auxiliary arrays for inverting diagonal blocks using dsytri_ */
    ipiv =(integer *)        MAlloc((size_t)n*sizeof(integer),        "DSYMselbinv:ipiv");
    work =(doubleprecision *)MAlloc((size_t)n*sizeof(doubleprecision),"DSYMselbinv:work");
    /* auxiliary buff for output D */
    Dbuff=(doubleprecision *)MAlloc((size_t)n*sizeof(doubleprecision),"DSYMselbinv:Dbuff");
    /* auxiliary buffer for level 3 BLAS */
    size_gemm_buff=n;
    gemm_buff=(doubleprecision *)MAlloc((size_t)size_gemm_buff*sizeof(doubleprecision),"DSYMselbinv:gemm_buff");

    /* inverse mapping index -> block number */
    block=(integer *)CAlloc((size_t)n,sizeof(integer),"DSYMselbinv:block");
    for (i=0; i<nblocks; i++) {
        BL_block=mxGetCell(BL,i);
	if (!mxIsStruct(BL_block))
	   mexErrMsgTxt("Field BL{i} must be a structure.");
	BL_blockJ=mxGetField(BL_block,0,"J");
	if (BL_blockJ==NULL)
	   mexErrMsgTxt("Field BL{i}.J does not exist.");
	if (!mxIsNumeric(BL_blockJ))
	   mexErrMsgTxt("Field BL{i}.J must be numerical.");
	n_size=mxGetN(BL_blockJ)*mxGetM(BL_blockJ);
#ifdef PRINT_CHECK
	if (mxGetM(BL_blockJ)!=1 && mxGetN(BL_blockJ)!=1) {
           mexPrintf("BL{%d}.J must be a 1-dim. array!\n",i+1);
	   fflush(stdout);
        }
#endif
	prBLJ=(double *)mxGetPr(BL_blockJ);
	for (k=0; k<n_size; k++) {
	    j=*prBLJ++;
	    /* remember that the structure stores indices from 1,...,n */
#ifdef PRINT_CHECK
	    if (j<1 || j>n) {
	       mexPrintf("index %d=BL{%d}.J(%d) out of range!\n",j,i+1,k+1);
	       fflush(stdout);
	    }
	    if (block[j-1]!=0) {
	       mexPrintf("block[%d]=%d nonzero!\n",j,block[j-1]+1);
	       fflush(stdout);
	    }
#endif
	    block[j-1]=i;
	} /* end for k */
    } /* end for i */
#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: inverse mapping index -> block number computed\n");fflush(stdout);
    for (i=0; i<n; i++)
        mexPrintf("%4d", block[i]+1);
    mexPrintf("\n");fflush(stdout);
#endif



    /* start selective block inversion from the back */
    k=nblocks-1;

    /* extract BL{k} */
    BL_block=mxGetCell(BL,k);

    /* extract source BL{k}.J */
    BL_blockJ=mxGetField(BL_block,0,"J");
    n_size=mxGetN(BL_blockJ)*mxGetM(BL_blockJ);
    prBLJ=(double *)mxGetPr(BL_blockJ);
    /* BL{k}.I and BL{k}.L should be empty */
    /* extract source BL{k}.D */
    BL_blockD=mxGetField(BL_block,0,"D");
    if (BL_blockD==NULL)
       mexErrMsgTxt("Field BL{k}.D does not exist.");
    if (mxGetN(BL_blockD)!=n_size || mxGetM(BL_blockD)!=n_size || !mxIsNumeric(BL_blockD))
       mexErrMsgTxt("Field BL{k}.D must be square dense matrix of same size as BL{k}.J.");
    /* numerical values of BL{k}.D */
    prBLD=(double *)mxGetPr(BL_blockD);

    /* extract BD{k} */
    BD_block=mxGetCell(BD,k);
    if (!mxIsStruct(BD_block))
       mexErrMsgTxt("Field BD{k} must be a structure.");

    /* extract source BD{k}.D */
    BD_blockD=mxGetField(BD_block,0,"D");
    if (BD_blockD==NULL)
       mexErrMsgTxt("Field BD{k}.D does not exist.");
    if (mxGetN(BD_blockD)!=n_size || mxGetM(BD_blockD)!=n_size || !mxIsSparse(BD_blockD))
       mexErrMsgTxt("Field BD{k}.D must be square sparse matrix of same size as BL{k}.J.");
    /* sparse representation of BD{k}.D */
    ia   =(mwIndex *)mxGetJc(BD_blockD);
    ja   =(mwIndex *)mxGetIr(BD_blockD);
    prBDD=(double *) mxGetPr(BD_blockD);



    /* set up new block column for BLinv{k} with four elements J, I, L, D */
    BLinv_block=mxCreateStructMatrix((mwSize)1, (mwSize)1, 4, BLnames);

    /* structure element 0:  J */
    /* create BLinv{k}.J */
    BLinv_blockJ=mxCreateDoubleMatrix((mwSize)1,(mwSize)n_size, mxREAL);
    /* copy data */
    prBLinvJ=(double *)mxGetPr(BLinv_blockJ);
    memcpy(prBLinvJ, prBLJ, (size_t)n_size*sizeof(double));
    /* set each field in BLinv_block structure */
    mxSetFieldByNumber(BLinv_block, (mwIndex)0, 0, BLinv_blockJ);

    /* structure element 1:  I */
    /* create empty BLinv{k}.I */
    m_size=0;
    BLinv_blockI=mxCreateDoubleMatrix((mwSize)1,(mwSize)m_size, mxREAL);
    /* set each field in BLinv_block structure */
    mxSetFieldByNumber(BLinv_block, (mwIndex)0, 1, BLinv_blockI);

    /* structure element 2:  L */
    /* create empty BLinv{k}.L */
    BLinv_blockL=mxCreateDoubleMatrix((mwSize)m_size,(mwSize)n_size, mxREAL);
    /* set each field in BLinv_block structure */
    mxSetFieldByNumber(BLinv_block, (mwIndex)0, 2, BLinv_blockL);

    /* structure element 3:  D */
    /* create dense n_size x n_size matrix BLinv{k}.D */
    BLinv_blockD=mxCreateDoubleMatrix((mwSize)n_size,(mwSize)n_size, mxREAL);
    prBLinvD=(double *)mxGetPr(BLinv_blockD);
    /* copy lower triangular part column by column */
    for (j=0; j<n_size; j++) {
        /* init strict upper triangular part with zeros */
        for (i=0; i<j; i++)
	    *prBLinvD++=0.0;

	/* diagonal (block) */
	/* scalar case */
	if (ia[j+1]-ia[j]==1) {
	   /* copy diagonal entry from sparse BD{k}.D */
	   *prBLinvD++=*prBDD++;
	   /* advance source BL{k}.D to its strict lower triangular part of column j */
	   prBLD+=j+1;
	   /* 1x1 block no pivoting */
	   ipiv[j]=j+1;
	   /* index for strict lower triangular part */
	   i++;
	}
	/* 2x2 case, column j starts with row index j */
	else if (ja[ia[j]]==j) {
	   /* copy diagonal entry (j,j) from sparse BD{k}.D */
	   *prBLinvD++=*prBDD++;
	   /* copy sub-diagonal entry (j+1,j) from sparse BD{k}.D */
	   *prBLinvD++=*prBDD++;
	   /* advance source BL{k}.D to the strict lower triangular part of
	      column j, excluding the sub-diagonal entry */
	   prBLD+=j+2;
	   /* 2x2 block (but no pivoting), remember that indexing for dsytri_
	      must be in FORTRAN style 1,...,n_size */
	   ipiv[j]=-(j+2);
	   /* index for strict lower triangular part excluding sub-diagonal entry */
	   i+=2;
	}
	/* 2x2 case, but second column since it starts with different row index
	   (must be j-1 in this case). This refers more or less to the scalar case */
	else  {
#ifdef PRINT_CHECK
	   if (ja[ia[j]]!=j-1) {
	      mexPrintf("column %d, leading has index %d but should be %d\n",j+1, ja[ia[j]]+1,j);
	      fflush(stdout);
	   }
#endif
	   /* skip super-diagonal entry (j-1,j) of sparse BD{k}.D */
	   prBDD++;
	   /* copy diagonal entry from sparse BD{k}.D */
	   *prBLinvD++=*prBDD++;
	   /* advance source BL{k}.D to the strict lower triangular part of column j */
	   prBLD+=j+1;
	   /* 2x2 block (but no pivoting) */
	   ipiv[j]=-(j+1);
	   /* index for strict lower triangular part */
	   i++;
	}
        /* copy strict lower triangular part from BL{k}.D */
        for (; i<n_size; i++)
	    *prBLinvD++=*prBLD++;
    } /* end for j */
#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: final lower triangular part copied\n");fflush(stdout);
    prBLinvD=(double *)mxGetPr(BLinv_blockD);
    mexPrintf("        ");
    for (j=0; j<n_size; j++)
        mexPrintf("%8d", ipiv[j]);
    mexPrintf("\n");fflush(stdout);
    mexPrintf("        ");
    for (j=0; j<n_size; j++)
        mexPrintf("%8d", (integer)prBLinvJ[j]);
    mexPrintf("\n");fflush(stdout);
    for (i=0; i<n_size; i++) {
        mexPrintf("%8d", (integer)prBLinvJ[i]);
	for (j=0; j<n_size; j++)
	    mexPrintf("%8.1le", prBLinvD[i+j*n_size]);
	mexPrintf("\n");fflush(stdout);
    }
#endif

    /* use LAPACK's dsytri_ for matrix inversion given the LDL^T decompositon */
    prBLinvD=(double *)mxGetPr(BLinv_blockD);
    j=0;
    dsytri_("L", &n_size, prBLinvD, &n_size, ipiv, work, &j,1);
    if (j<0) {
       mexPrintf("the %d-th argument had an illegal value\n",-j);
       mexErrMsgTxt("dsytri_ failed\n");
    }
    if (j>0) {
       mexPrintf("D(%d,%d) = 0; the matrix is singular and its inverse could not be computed\n",j,j);
       mexErrMsgTxt("dsytri_ failed\n");
    }
#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: final inverse lower triangular part computed\n");fflush(stdout);
    prBLinvD=(double *)mxGetPr(BLinv_blockD);
    mexPrintf("        ");
    for (j=0; j<n_size; j++)
        mexPrintf("%8d", (integer)prBLinvJ[j]);
    mexPrintf("\n");fflush(stdout);
    for (i=0; i<n_size; i++) {
        mexPrintf("%8d", (integer)prBLinvJ[i]);
	for (j=0; j<n_size; j++)
	    mexPrintf("%8.1le", prBLinvD[i+j*n_size]);
	mexPrintf("\n");fflush(stdout);
    }
#endif


    /* successively downdate "n" by the size "n_size" of the diagonal block */
    sumn=n-n_size;
    /* for convenience copy data to the upper triangular part */
    for (j=0; j<n_size; j++) {
        /* advance to the diagonal part of column j */
        prBLinvD+=j;
        /* extract diagonal entry and advance to the strict lower triangular part of column j */
	Dbuff[sumn+j]=*prBLinvD++;

	/* pointer to BLinv{k}.D(j,j+1) */
	pr=prBLinvD+n_size-1;
	/* copy data to the strict upper triangular part of row j */
        for (i=j+1; i<n_size; i++,pr+=n_size)
	    *pr=*prBLinvD++;
    } /* end for j */
#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: inverse diagonal entries extracted\n");fflush(stdout);
    for (j=0; j<n_size; j++)
        mexPrintf("%8.1le", Dbuff[sumn+j]);
    mexPrintf("\n");fflush(stdout);
    mexPrintf("DSYMselbinv: final inverse diagonal block computed\n");fflush(stdout);
    prBLinvD=(double *)mxGetPr(BLinv_blockD);
    mexPrintf("        ");
    for (j=0; j<n_size; j++)
      mexPrintf("%8d", (integer)prBLinvJ[j]);
    mexPrintf("\n");fflush(stdout);
    for (i=0; i<n_size; i++) {
        mexPrintf("%8d", (integer)prBLinvJ[i]);
	for (j=0; j<n_size; j++)
	    mexPrintf("%8.1le", prBLinvD[i+j*n_size]);
	mexPrintf("\n");fflush(stdout);
    }
#endif

    /* set each field in BLinv_block structure */
    mxSetFieldByNumber(BLinv_block, (mwIndex)0, 3, BLinv_blockD);

    /* finally set output BLinv{k} */
    mxSetCell(BLinv,(mwIndex)k,BLinv_block);


    /* advance backwards toward the top */
    k--;

    /* main loop */
    while (k>=0) {

          /* extract BL{k} */
          BL_block=mxGetCell(BL,k);

	  /* 1. BL{k}.J */
	  BL_blockJ=mxGetField(BL_block,0,"J");
	  n_size=mxGetN(BL_blockJ)*mxGetM(BL_blockJ);
	  prBLJ=(double *)mxGetPr(BL_blockJ);
	  /* 2. BL{k}.I */
	  BL_blockI=mxGetField(BL_block,0,"I");
	  if (BL_blockI==NULL)
	     mexErrMsgTxt("Field BL{k}.I does not exist.");
	  m_size=mxGetN(BL_blockI)*mxGetM(BL_blockI);
#ifdef PRINT_CHECK
	  if (mxGetM(BL_blockI)!=1 && mxGetN(BL_blockI)!=1) {
	     mexPrintf("BL{%d}.I must be a 1-dim. array!\n",k+1);
	     fflush(stdout);
	  }
#endif
	  prBLI=(double *)mxGetPr(BL_blockI);
	  /* 3. BL{k}.L */
	  BL_blockL=mxGetField(BL_block,0,"L");
	  if (BL_blockL==NULL)
	     mexErrMsgTxt("Field BL{k}.L does not exist.");
	  /* numerical values of BL{k}.L */
	  prBLL=(double *)mxGetPr(BL_blockL);
	  /* 4. BL{k}.D */
	  BL_blockD=mxGetField(BL_block,0,"D");
	  if (BL_blockD==NULL)
	     mexErrMsgTxt("Field BL{k}.D does not exist.");
	  if (mxGetN(BL_blockD)!=n_size || mxGetM(BL_blockD)!=n_size || !mxIsNumeric(BL_blockD))
	     mexErrMsgTxt("Field BL{k}.D must be square dense matrix of same size as BL{k}.J.");
	  /* numerical values of BL{k}.D */
	  prBLD=(double *)mxGetPr(BL_blockD);

          /* extract BD{k} */
          BD_block=mxGetCell(BD,k);
	  if (!mxIsStruct(BD_block))
	     mexErrMsgTxt("Field BD{k} must be a structure.");

	  /* extract source BD{k}.D */
	  BD_blockD=mxGetField(BD_block,0,"D");
	  if (BD_blockD==NULL)
	     mexErrMsgTxt("Field BD{k}.D does not exist.");
	  if (mxGetN(BD_blockD)!=n_size || mxGetM(BD_blockD)!=n_size || !mxIsSparse(BD_blockD))
	     mexErrMsgTxt("Field BD{k}.D must be square sparse matrix of same size as BL{k}.J.");
	  /* sparse representation of BD{k}.D */
	  ia   =(mwIndex *)mxGetJc(BD_blockD);
	  ja   =(mwIndex *)mxGetIr(BD_blockD);
	  prBDD=(double *) mxGetPr(BD_blockD);


          /* set up new block column for BLinv{k} with four elements J, I, L, D */
          BLinv_block=mxCreateStructMatrix((mwSize)1, (mwSize)1, 4, BLnames);

	  /* structure element 0:  J */
	  /* create BLinv{k}.J */
	  BLinv_blockJ=mxCreateDoubleMatrix((mwSize)1,(mwSize)n_size, mxREAL);
	  /* copy data */
	  prBLinvJ=(double *)mxGetPr(BLinv_blockJ);
	  memcpy(prBLinvJ, prBLJ, (size_t)n_size*sizeof(double));
	  /* set each field in BLinv_block structure */
	  mxSetFieldByNumber(BLinv_block, (mwIndex)0, 0, BLinv_blockJ);

	  /* structure element 1:  I */
	  /* create empty BLinv{k}.I */
	  BLinv_blockI=mxCreateDoubleMatrix((mwSize)1,(mwSize)m_size, mxREAL);
	  /* copy data */
	  prBLinvI=(double *)mxGetPr(BLinv_blockI);
	  memcpy(prBLinvI, prBLI, (size_t)m_size*sizeof(double));
	  /* set each field in BLinv_block structure */
	  mxSetFieldByNumber(BLinv_block, (mwIndex)0, 1, BLinv_blockI);


	  /* structure element 2:  L */
	  /* create empty BLinv{k}.L */
	  BLinv_blockL=mxCreateDoubleMatrix((mwSize)m_size,(mwSize)n_size, mxREAL);
	  prBLinvL=(double *)mxGetPr(BLinv_blockL);
	  /* init with zeros */
	  for (j=0; j<m_size*n_size; j++)
	      *prBLinvL++=0.0;
	  prBLinvL=(double *)mxGetPr(BLinv_blockL);

	  /* scan the indices of BL{k}.I to find out which block columns are required */
	  l=0;
	  while (l<m_size) {
		/* associated index I[l] converted to C-style */
	        ii=(integer)prBLI[l]-1;
	        i=block[ii];

		/* find out how many indices of I are associated with block column i */
		j=l+1;
		flag=-1;
		while (flag) {
		      if (j>=m_size) {
			 j=m_size-1;
			 flag=0;
		      }
		      else {
			 /* associated index I[j] converted to C-style */
			 ii=(integer)prBLI[j]-1;
			 if (block[ii]>i) {
			    j--;
			    flag=0;
			 }
			 else
			    j++;
		      } /* end if-else j>=m_size */
		} /* end while flag */
		/* now BL{k}.I(l:j) are associated with block column BLinv{i} */

		/* extract already computed BLinv{i}, i>k */
		BLinv_blocki =mxGetCell(BLinv,(mwIndex)i);
#ifdef PRINT_CHECK
		if (BLinv_blocki==NULL) {
		   mexPrintf("BLinv{%d} does not exist!\n",i+1);
		   fflush(stdout);
		}
		else if (!mxIsStruct(BLinv_blocki)) {
		   mexPrintf("BLinv{%d} must be structure!\n",i+1);
		   fflush(stdout);
		}
#endif

		/* BLinv{i}.J */
		BLinv_blockJi=mxGetField(BLinv_blocki,0,"J");
#ifdef PRINT_CHECK
		if (BLinv_blockJi==NULL) {
		   mexPrintf("BLinv{%d}.J does not exist!\n",i+1);
		   fflush(stdout);
		}
		else if (mxGetM(BLinv_blockJi)!=1 && mxGetN(BLinv_blockJi)!=1) {
		   mexPrintf("BLinv{%d}.J must be a 1-dim. array!\n",i+1);
		   fflush(stdout);
		}
#endif
		ni_size=mxGetN(BLinv_blockJi)*mxGetM(BLinv_blockJi);
		prBLinvJi=(double *)mxGetPr(BLinv_blockJi);

		/* BLinv{i}.I */
		BLinv_blockIi=mxGetField(BLinv_blocki,0,"I");
#ifdef PRINT_CHECK
		if (BLinv_blockIi==NULL) {
		   mexPrintf("BLinv{%d}.I does not exist!\n",i+1);
		   fflush(stdout);
		}
		else if (mxGetM(BLinv_blockIi)!=1 && mxGetN(BLinv_blockIi)!=1) {
		   mexPrintf("BLinv{%d}.I must be a 1-dim. array!\n",i+1);
		   fflush(stdout);
		}
#endif
		mi_size=mxGetN(BLinv_blockIi)*mxGetM(BLinv_blockIi);
		prBLinvIi=(double *)mxGetPr(BLinv_blockIi);

		/* BLinv{i}.L */
		BLinv_blockLi=mxGetField(BLinv_blocki,0,"L");
#ifdef PRINT_CHECK
		if (BLinv_blockLi==NULL) {
		   mexPrintf("BLinv{%d}.L does not exist!\n",i+1);
		   fflush(stdout);
		}
#endif
		prBLinvLi=(double *)mxGetPr(BLinv_blockLi);

		/* BLinv{i}.D */
		BLinv_blockDi=mxGetField(BLinv_blocki,0,"D");
#ifdef PRINT_CHECK
		if (BLinv_blockDi==NULL) {
		   mexPrintf("BLinv{%d}.D does not exist!\n",i+1);
		   fflush(stdout);
		}
#endif
		prBLinvDi=(double *)mxGetPr(BLinv_blockDi);


		/* l:j refers to continuously chosen indices !!! */
		/* Ji, Ik and Ii may exclude some entries !!! */


		/* check if I(l:j)==I(l):I(j) (continuous sequence of indices) */
		/* flag for contiguous index set */
		Ji_cont=-1;
		/* BLinv{i}.D(Ji,Ji) will physically start at position j_first,
		   where Ji refers to the sequence of positions in BLinv{i}.D
		   associated with I(l:j)
		*/
#ifdef PRINT_INFO
		mexPrintf("BL{%d}.I(%d:%d)\n",k+1,l+1,j+1);
		for (jj=l; jj<=j; jj++)
		    mexPrintf("%4d",(integer)prBLI[jj]);
		mexPrintf("\n");
		mexPrintf("BLinv{%d}.J=%d:%d\n",i+1,(integer)prBLinvJi[0],
			  (integer)prBLinvJi[ni_size-1]);
		fflush(stdout);
#endif
		j_first=((integer)prBLI[l])-((integer)prBLinvJi[0]);
		for (jj=l; jj<=j; jj++) {
		    /* index I[jj] in MATLAB-style 1,...,n */
		    ii=(integer)prBLI[jj];
		    /* non-contiguous index found, break! */
		    if (ii>(integer)prBLI[l]+jj-l) {
		       Ji_cont=0;
		       jj=j+1;
		    }
		} /* end for jj */
#ifdef PRINT_INFO
		if (Ji_cont)
		   mexPrintf("BL{%d}.I(%d:%d) is a contiguous subsequence of BLinv{%d}.J\n",
			     k+1,l+1,j+1,i+1);
		else
		   mexPrintf("BL{%d}.I(%d:%d) does not refer to a contiguous subsequence of BLinv{%d}.J\n",
			     k+1,l+1,j+1,i+1);
		fflush(stdout);
#endif

		/* check if the intersection of BLinv{k}.I and BLinv{i}.I
		   consists of contiguous indices */
		Ik_cont=-1; Ii_cont=-1;
		p=0; q=0;
		t=0;
		k_first=0; i_first=0;
		while (p<m_size && q<mi_size) {
		      /* indices in MATLAB-style */
		      ii=(integer)prBLI[p];
		      jj=(integer)prBLinvIi[q];
		      if (ii<jj) {
			 p++;
			 /* If we already have common indices, BLinv{k}.I[p]<BLinv{i}.I[q] refers
			    to a gap in the intersection w.r.t. BLinv{k}.I
			 */
		      }
		      else if (ii>jj) {
			 q++;
		      }
		      else { /* indices match */
			 /* store number of the first common index */
			 if (Ik_cont==-1) {
			    /* BLinv{k}.L(Ik,:) will physically start at position
			       k_first, where Ik refers to the sequence of positions
			       in BLinv{k}.L associated with the intersection of
			       BLinv{k}.I and BLinv{i}.I
			    */
			    k_first=p;
			    /* BLinv{i}.L(Ii,:) will physically start at position
			       i_first, where Ii refers to the sequence of positions
			       in BLinv{i}.L associated with the intersection of
			       BLinv{k}.I and BLinv{i}.I
			    */
			    i_first=q;
			    /* store positions of the next indices to stay contiguous */
			    Ik_cont=p+1;
			    Ii_cont=q+1;
			 }
			 else {
			    /* there exists at least one common index */
			    /* check if the current index position is the
			       successor of the previous position */
			    if (p==Ik_cont)
			       /* store position of the next index to stay contiguous */
			       Ik_cont=p+1;
			    else
			       Ik_cont=0;
			    if (q==Ii_cont)
			       /* store position of the next index to stay contiguous */
			       Ii_cont=q+1;
			    else
			       Ii_cont=0;
			 }
			 p++; q++; t++;
		      } /* end if-elseif-else */
		} /* end while p&q */
#ifdef PRINT_INFO
		mexPrintf("BL{%d}.I\n",k+1);
		for (p=0; p<m_size; p++)
		    mexPrintf("%4d",(integer)prBLI[p]);
		mexPrintf("\n");
		fflush(stdout);
		mexPrintf("BLinv{%d}.I\n",i+1);
		for (q=0; q<mi_size; q++)
		    mexPrintf("%4d",(integer)prBLinvIi[q]);
		mexPrintf("\n");
		fflush(stdout);
		if (Ik_cont)
		   mexPrintf("intersection leads to a contiguous sequence inside BL{%d}.I of length %d\n",
			     k+1,t);
		else
		   mexPrintf("intersection does not yield a contiguous sequence of BL{%d}.I\n",
			     k+1);
		if (Ii_cont)
		   mexPrintf("intersection leads to a contiguous sequence inside BLinv{%d}.I  of length %d\n",
			     i+1,t);
		else
		   mexPrintf("intersection does not yield a contiguous sequence of BLinv{%d}.I\n",
			     i+1);
		fflush(stdout);
#endif


		/* optimal case, all index sets refer to successively stored rows and columns.
		   We can easily use Level 3 BLAS
		*/
		if (Ii_cont && Ik_cont && Ji_cont) {
#ifdef PRINT_INFO
		   mexPrintf("ideal case, use level 3 BLAS directly!\n");
		   fflush(stdout);
#endif
		   /* contribution from the strict lower triangular part */
		   /* BLinv{k}.L(Ik,:)  = - BLinv{i}.L(Ii,Ji) *BL{k}.L(l:j,:)  + BLinv{k}.L(Ik,:) */
		   alpha=-1.0; beta=1.0;
		   ii=j-l+1;
		   if (t)
		      dgemm_("N","N",&t,&n_size,&ii,
			     &alpha,
			     prBLinvLi+i_first+mi_size*j_first,&mi_size,
			     prBLL+l,&m_size,
			     &beta,
			     prBLinvL+k_first,&m_size,1,1);
#ifdef PRINT_INFO
		   mexPrintf("Ik=[");
		   r=0; s=0;
		   while (r<m_size && s<mi_size) {
		         if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			    r++;
			 else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			    s++;
			 else {
			    mexPrintf("%8d", r+1);
			    r++; s++;
			 }
		   }
		   mexPrintf("];\n");
		   mexPrintf("Ii=[");
		   r=0; s=0;
		   while (r<m_size && s<mi_size) {
		         if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			    r++;
			 else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			    s++;
			 else {
			    mexPrintf("%8d", s+1);
			    r++; s++;
			 }
		   }
		   mexPrintf("];\n");
		   mexPrintf("Ji=[");
		   r=l; s=0;
		   while (s<ni_size) {
		         if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
			    mexPrintf("%8d", s+1);
			    r++;
			 }
			 s++;
		   }
		   mexPrintf("];\n");
		   mexPrintf("DSYMselbinv: BLinv{%d}.L(Ik,:) = - BLinv{%d}.L(Ii,Ji)  *BL{%d}.L(%d:%d,:)  + BLinv{%d}.L(Ik,:)\n",k+1,i+1,k+1,l+1,j+1,k+1);
		   r=0; s=0;
		   while (r<m_size && s<mi_size) {
		         if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			    r++;
			 else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			    s++;
			 else {
			    for (jj=0; jj<n_size; jj++)
			        mexPrintf("%8.1le",prBLinvL[r+m_size*jj]);
			    mexPrintf("\n");
			    fflush(stdout);
			    r++; s++;
			 } /* end if-elseif-else */
		   } /* end while r&s */
#endif

		   /* contribution from the strict upper triangular part */
		   /* BLinv{k}.L(l:j,:) = - BLinv{i}.L(Ii,Ji)^T*BL{k}.L(Ik,:)  + BLinv{k}.L(l:j,:)  */
		   if (t)
		      dgemm_("T","N",&ii,&n_size,&t,
			     &alpha,
			     prBLinvLi+i_first+mi_size*j_first,&mi_size,
			     prBLL+k_first,&m_size,
			     &beta,
			     prBLinvL+l,&m_size,1,1);
#ifdef PRINT_INFO
		   mexPrintf("Ik=[");
		   r=0; s=0;
		   while (r<m_size && s<mi_size) {
		         if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			    r++;
			 else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			    s++;
			 else {
			    mexPrintf("%8d", r+1);
			    r++; s++;
			 }
		   }
		   mexPrintf("];\n");
		   mexPrintf("Ii=[");
		   r=0; s=0;
		   while (r<m_size && s<mi_size) {
		         if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			    r++;
			 else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			    s++;
			 else {
			    mexPrintf("%8d", s+1);
			    r++; s++;
			 }
		   }
		   mexPrintf("];\n");
		   mexPrintf("Ji=[");
		   r=l; s=0;
		   while (s<ni_size) {
		         if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
			    mexPrintf("%8d", s+1);
			    r++;
			 }
			 s++;
		   }
		   mexPrintf("];\n");
		   mexPrintf("DSYMselbinv: BLinv{%d}.L(%d:%d,:) = - BLinv{%d}.L(Ii,Ji)' *BL{%d}.L(Ik,:)  + BLinv{%d}.L(%d:%d,:)\n",k+1,l+1,j+1,i+1,k+1,k+1,l+1,j+1);
		   for (jj=l; jj<=j; jj++) {
		       for (q=0; q<n_size; q++)
			   mexPrintf("%8.1le",prBLinvL[jj+m_size*q]);
		       mexPrintf("\n");
		       fflush(stdout);
		   }
#endif

		   /*  contribution from the diagonal block */
		   /* BLinv{k}.L(l:j,:) = - BLinv{i}.D(Ji,Ji)  *BL{k}.L(l:j,:) + BLinv{k}.L(l:j,:)  */
		   dgemm_("N","N",&ii,&n_size,&ii,
			  &alpha,
			  prBLinvDi+j_first+j_first*ni_size,&ni_size,
			  prBLL+l,&m_size,
			  &beta,
			  prBLinvL+l,&m_size,1,1);
#ifdef PRINT_INFO
		   r=l; s=0;
		   mexPrintf("Ji=[");
		   while (s<ni_size) {
		         if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
			    mexPrintf("%8d", s+1);
			    r++;
			 }
			 s++;
		   }
		   mexPrintf("];\n");
		   mexPrintf("DSYMselbinv: BLinv{%d}.L(%d:%d,:) = - BLinv{%d}.D(Ji,Ji)  *BL{%d}.L(%d:%d,:)  + BLinv{%d}.L(%d:%d,:)\n",
			     k+1,l+1,j+1,i+1,k+1,l+1,j+1,k+1,l+1,j+1);
		   for (r=l; r<=j; r++) {
		       for (s=0; s<n_size; s++)
			   mexPrintf("%8.1le",prBLinvL[r+m_size*s]);
		       mexPrintf("\n");
		       fflush(stdout);
		   }
#endif
		} /* end if Ii_cont & Ik_cont & Ji_cont */
		else { /* now at least one block is not contiguous. The decision
			  whether to stik with level 3 BLAS or not will be made on
			  the cost for copying part of the data versus the
			  computational cost. This is definitely not optimal
		       */



  		   /**********************************************************/
  		   /**********************************************************/
		   /*** contribution from the strict lower triangular part ***/
		   /* BLinv{k}.L(Ik,:)  = - BLinv{i}.L(Ii,Ji)  *BL{k}.L(l:j,:) + BLinv{k}.L(Ik,:) */
		   /* determine amount of auxiliary memory */
#ifdef PRINT_INFO
		   if (!Ji_cont)
		      mexPrintf("Ji not contiguous\n");
		   if (!Ii_cont)
		      mexPrintf("Ii not contiguous\n");
		   if (!Ik_cont)
		      mexPrintf("Ik not contiguous\n");
		   fflush(stdout);
#endif
		   copy_cnt=0;
		   /* level 3 BLAS have to use |Ii| x |Ji| buffer rather than BLinv{i}.L(Ii,Ji) */
		   if (!Ii_cont || !Ji_cont)
		      copy_cnt+=t*(j-l+1);
		   /* level 3 BLAS have to use |Ik| x n_size buffer rather than BLinv{k}.L(Ik,:) */
		   if (!Ik_cont)
		      copy_cnt+=t*n_size;

		   if (copy_cnt<t*(j-l+1)*n_size)
		      level3_BLAS=-1;
		   else
		      level3_BLAS=0;

		   /* it could pay off to copy the data into one or two auxiliary buffers */
		   if (level3_BLAS && t) {
#ifdef PRINT_INFO
		      mexPrintf("contribution from the strict lower triangular part, still use level 3 BLAS\n");
		      fflush(stdout);
#endif
		      size_gemm_buff=MAX(size_gemm_buff,copy_cnt);
		      gemm_buff=(doubleprecision *)ReAlloc(gemm_buff,
							   (size_t)size_gemm_buff*sizeof(doubleprecision),
							   "DSYMselbinv:gemm_buff");
		      if (!Ii_cont || !Ji_cont) {
		         /* copy BLinv{i}.L(Ii,Ji) to buffer */
		         pr=gemm_buff;
			 p=0; q=0;
			 while (p<m_size && q<mi_size) {
			       ii=(integer)prBLI[p];
			       jj=(integer)prBLinvIi[q];
			       if (ii<jj)
			 	  p++;
			       else if (ii>jj)
				  q++;
			       else { /* indices match */

				  /* copy parts of the current row BLinv{i}.L(p,:) of
				     BLinv{i}.L(Ii,Ji) associated with Ji to gemm_buff */
				  pr3=pr;
				  pr2=prBLinvLi+q;
				  r=l; s=0;
				  while (s<ni_size) {
				        /* does column BL{k}.I(r) match some BLinv{i}.J(s)?
					   Recall that I(l:j) is a subset of Ji
					*/
				        if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
					   *pr3=*pr2;
					   pr3+=t;
					   r++;
					}
					s++;
					pr2+=mi_size;
				  } /* end while s */
				  pr++;

				  p++; q++;
			       } /* end if-elseif-else */
			 } /* end while p&q */
#ifdef PRINT_INFO
			 mexPrintf("Ik=[");
			 r=0; s=0;
			 while (r<m_size && s<mi_size) {
		               if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			          r++;
			       else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
				  s++;
			       else {
		                  mexPrintf("%8d", r+1);
				  r++; s++;
			       }
			 }
			 mexPrintf("];\n");
			 mexPrintf("Ii=[");
			 r=0; s=0;
			 while (r<m_size && s<mi_size) {
			       if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			          r++;
			       else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
				  s++;
			       else {
		                  mexPrintf("%8d", s+1);
				  r++; s++;
			       }
			 }
			 mexPrintf("];\n");
			 mexPrintf("Ji=[");
			 r=l; s=0;
			 while (s<ni_size) {
		               if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
		                  mexPrintf("%8d", s+1);
				  r++;
			       }
			       s++;
			 }
			 mexPrintf("];\n");
			 mexPrintf("DSYMselbinv: BLinv{%d}.L(Ii,Ji) cached\n",i+1);fflush(stdout);
			 mexPrintf("        ");
			 r=l; s=0;
			 while (s<ni_size) {
			       if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
				  mexPrintf("%8d", (integer)prBLinvJi[s]);
				  r++;
			       }
			       s++;
			 } /* end while s */
			 mexPrintf("\n");fflush(stdout);
			 p=0; q=0;
			 pr=gemm_buff;
			 while (p<m_size && q<mi_size) {
			       ii=(integer)prBLI[p];
			       jj=(integer)prBLinvIi[q];
			       if (ii<jj)
				  p++;
			       else if (ii>jj)
				  q++;
			       else { /* indices match */
				  mexPrintf("%8d", ii);

				  r=l; s=0;
				  pr2=pr;
				  while (s<ni_size) {
				        if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
					   mexPrintf("%8.1le", *pr2);
					   pr2+=t;
					   r++;
					}
					s++;
				  }
				  pr++;
				  mexPrintf("\n");fflush(stdout);

				  p++; q++;
			       }
			 }
#endif

			 pr=gemm_buff; p=t;
		      }
		      else {
			 /* pointer to BLinv{i}.L(Ii,Ji) and LDA */
		         pr=prBLinvLi+i_first+mi_size*j_first; p=mi_size;
		      } /* end if-else */

		      if (!Ik_cont) {
		         /* init buffer with zeros */
		         if (!Ii_cont || !Ji_cont)
			    pr2=gemm_buff+t*(j-l+1);
			 else
			    pr2=gemm_buff;
			 for (q=0; q<t*n_size; q++)
			     *pr2++=0.0;
			 /* pointer and LDC */
			 if (!Ii_cont || !Ji_cont)
			    pr2=gemm_buff+t*(j-l+1);
			 else
			    pr2=gemm_buff;
			 q=t;
			 /* since we initialized everything with zero, beta is
			    almost arbitrary, we indicate this changing beta to 0
			 */
			 alpha=1.0; beta=0.0;
#ifdef PRINT_INFO
			 mexPrintf("DSYMselbinv: cached zeros instead of  BLinv{%d}.L(Ik,:)\n",k+1);
			 fflush(stdout);
#endif
		      }
		      else {
			 /* pointer to BLinv{k}.L(Ik,:) and LDC */
		         pr2=prBLinvL+k_first; q=m_size;
			 alpha=-1.0; beta=1.0;
		      } /* end if-else */

		      /* call level 3 BLAS */
		      ii=j-l+1;
		      dgemm_("N","N",&t,&n_size,&ii,
			     &alpha,
			     pr,&p,
			     prBLL+l,&m_size,
			     &beta,
			     pr2,&q,1,1);
#ifdef PRINT_INFO
		      mexPrintf("Ik=[");
		      r=0; s=0;
		      while (r<m_size && s<mi_size) {
		            if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			       r++;
			    else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			       s++;
			    else {
		               mexPrintf("%8d", r+1);
			       r++; s++;
		            }
		      }
		      mexPrintf("];\n");
		      mexPrintf("Ii=[");
		      r=0; s=0;
		      while (r<m_size && s<mi_size) {
			    if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			       r++;
			    else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			       s++;
			    else {
		               mexPrintf("%8d", s+1);
			       r++; s++;
		            }
		      }
		      mexPrintf("];\n");
		      r=l; s=0;
		      mexPrintf("Ji=[");
		      while (s<ni_size) {
		            if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
		               mexPrintf("%8d", s+1);
			       r++;
		            }
			    s++;
		      }
		      mexPrintf("];\n");
		      if (Ik_cont)
			 mexPrintf("DSYMselbinv: BLinv{%d}.L(Ik,:) = - BLinv{%d}.L(Ii,Ji)  *BL{%d}.L(%d:%d,:)  + BLinv{%d}.L(Ik,:)\n",
		                   k+1,i+1,k+1,l+1,j+1,k+1);
		      else
			 mexPrintf("DSYMselbinv: cached                BLinv{%d}.L(Ii,Ji)  *BL{%d}.L(%d:%d,:)\n",
		                   i+1,k+1,l+1,j+1);

		      for (r=0; r<t; r++) {
		          for (s=0; s<n_size; s++)
			      mexPrintf("%8.1le",pr2[r+q*s]);
			  mexPrintf("\n");
			  fflush(stdout);
		      }
#endif


		      if (!Ik_cont) {
		         /* init buffer with zeros */
		         if (!Ii_cont || !Ji_cont)
			    pr2=gemm_buff+t*(j-l+1);
			 else
			    pr2=gemm_buff;
			 p=0; q=0;
			 while (p<m_size && q<mi_size) {
			       ii=(integer)prBLI[p];
			       jj=(integer)prBLinvIi[q];
			       if (ii<jj)
				  p++;
			       else if (ii>jj)
				  q++;
			       else { /* indices match */

				  /* copy current row of pr2 to BLinv{k}.L(Ik,:) */
				  pr=(double *)mxGetPr(BLinv_blockL)+p;
				  pr3=pr2;
				  for (r=0; r<n_size; r++, pr+=m_size, pr3+=t)
				      *pr-=*pr3;
				  pr2++;

				  p++; q++;
			       } /* end if-elseif-else */
			 } /* end while p&q */
#ifdef PRINT_INFO
			 mexPrintf("Ik=[");
			 r=0; s=0;
			 while (r<m_size && s<mi_size) {
			       if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			          r++;
			       else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			          s++;
			       else {
		                  mexPrintf("%8d", r+1);
			          r++; s++;
		               }
		         }
			 mexPrintf("];\n");
			 mexPrintf("Ii=[");
			 r=0; s=0;
			 while (r<m_size && s<mi_size) {
			       if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			          r++;
			       else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			          s++;
			       else {
		                  mexPrintf("%8d", s+1);
				  r++; s++;
		               }
		         }
			 mexPrintf("];\n");
			 mexPrintf("Ji=[");
			 r=l; s=0;
			 while (s<ni_size) {
		               if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
		                  mexPrintf("%8d", s+1);
				  r++;
		               }
			       s++;
		         }
			 mexPrintf("];\n");
			 mexPrintf("DSYMselbinv: BLinv{%d}.L(Ik,:) = - BLinv{%d}.L(Ii,Ji)  *BL{%d}.L(%d:%d,:)  + BLinv{%d}.L(Ik,:)\n",k+1,i+1,k+1,l+1,j+1,k+1);

			 p=0; q=0;
			 while (p<m_size && q<mi_size) {
			       ii=(integer)prBLI[p];
			       jj=(integer)prBLinvIi[q];
			       if (ii<jj)
			 	  p++;
			       else if (ii>jj)
				  q++;
			       else {
		                  for (s=0; s<n_size; s++)
			              mexPrintf("%8.1le",prBLinvL[p+m_size*s]);
				  mexPrintf("\n");
				  fflush(stdout);
				  p++; q++;
		               } /* end if-elseif-else */
		         } /* end while p&q */
#endif


		      } /* if !Ik_cont */
		   } /* end if level3_BLAS */
		   else if (t) { /* it might not pay off, therefore we use a simple hand-coded loop */
		      /* BLinv{k}.L(Ik,:)  -=  BLinv{i}.L(Ii,Ji) * BL{k}.L(l:j,:) */
#ifdef PRINT_INFO
		      mexPrintf("contribution from the strict lower triangular part, use hand-coded loops\n");
		      fflush(stdout);
#endif
		      p=0; q=0;
		      while (p<m_size && q<mi_size) {
			    ii=(integer)prBLI[p];
			    jj=(integer)prBLinvIi[q];
			    if (ii<jj)
			       p++;
			    else if (ii>jj)
			       q++;
			    else { /* row indices BLinv{k}.I[p]=BLinv{i}.I[q] match */
			       pr =prBLinvL+p;
			       pr3=prBLL+l;
			       /* BLinv{k}.L(p,:)  -=  BLinv{i}.L(q,Ji) * BL{k}.L(l:j,:) */
			       for (ii=0; ii<n_size; ii++,pr+=m_size,pr3+=m_size-(j-l+1)) {
				   /* BLinv{k}.L(p,ii)  -=  BLinv{i}.L(q,Ji) * BL{k}.L(l:j,ii) */
				   pr2=prBLinvLi+q;
				   r=l; s=0;
				   while (s<ni_size) {
				         /* column Ji[s] occurs within I(l:j).
					    Recall that I(l:j) is a subset of Ji
					 */
				         if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
					    /* BLinv{k}.L(p,ii)  -=  BLinv{i}.L(q,s) * BL{k}.L(r,ii) */
					    *pr-=(*pr2)*(*pr3++);
					    r++;
					 }
					 s++;
					 pr2+=mi_size;
				   } /* end while s */
			       } /* end for ii */
			       p++; q++;
			    } /* end if-elseif-else */
		      } /* end while p&q */
#ifdef PRINT_INFO
		      mexPrintf("Ik=[");
		      r=0; s=0;
		      while (r<m_size && s<mi_size) {
			    if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			       r++;
			    else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			       s++;
			    else {
			       mexPrintf("%8d", r+1);
			       r++; s++;
			    }
		      }
		      mexPrintf("];\n");
		      mexPrintf("Ii=[");
		      r=0; s=0;
		      while (r<m_size && s<mi_size) {
			    if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			       r++;
			    else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			       s++;
			    else {
			       mexPrintf("%8d", s+1);
			       r++; s++;
			    }
		      }
		      mexPrintf("];\n");
		      mexPrintf("Ji=[");
		      r=l; s=0;
		      while (s<ni_size) {
			    if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
			       mexPrintf("%8d", s+1);
			       r++;
			    }
			    s++;
		      }
		      mexPrintf("];\n");
		      mexPrintf("DSYMselbinv: BLinv{%d}.L(Ik,:) = - BLinv{%d}.L(Ii,Ji)  *BL{%d}.L(%d:%d,:)  + BLinv{%d}.L(Ik,:)\n",k+1,i+1,k+1,l+1,j+1,k+1);
		      p=0; q=0;
		      while (p<m_size && q<mi_size) {
			    ii=(integer)prBLI[p];
			    jj=(integer)prBLinvIi[q];
			    if (ii<jj)
			       p++;
			    else if (ii>jj)
			       q++;
			    else {
			       for (s=0; s<n_size; s++)
				   mexPrintf("%8.1le",prBLinvL[p+m_size*s]);
			       mexPrintf("\n");
			       fflush(stdout);
			       p++; q++;
			    } /* end if-elseif-else */
		      } /* end while p&q */
#endif
		   } /* end if-else level3_BLAS */
		   else {
#ifdef PRINT_INFO
		      mexPrintf("contribution from the strict lower triangular part empty\n");
		      fflush(stdout);
#endif
		   }
		   /* end contribution from the strict lower triangular part */
		   /**********************************************************/
		   /**********************************************************/




  		   /**********************************************************/
  		   /**********************************************************/
		   /*** contribution from the strict upper triangular part ***/
		   /* BLinv{k}.L(l:j,:) = - BLinv{i}.L(Ii,Ji)^T*BL{k}.L(Ik,:)  + BLinv{k}.L(l:j,:)  */
		   /* determine amount of auxiliary memory */
		   copy_cnt=0;
		   /* level 3 BLAS have to use |Ii| x |Ji| buffer rather than BLinv{i}.L(Ii,Ji) */
		   if (!Ii_cont || !Ji_cont)
		      copy_cnt+=t*(j-l+1);
		   /* level 3 BLAS have to use |Ik| x n_size buffer rather than BLinv{k}.L(Ik,:) */
		   if (!Ik_cont)
		      copy_cnt+=t*n_size;

		   if (copy_cnt<t*(j-l+1)*n_size)
		      level3_BLAS=-1;
		   else
		      level3_BLAS=0;

		   /* it could pay off to copy the data into one or two auxiliary buffers */
		   if (level3_BLAS && t) {
#ifdef PRINT_INFO
		      mexPrintf("contribution from the strict upper triangular part, still use level 3 BLAS\n");
		      fflush(stdout);
#endif
		      size_gemm_buff=MAX(size_gemm_buff,copy_cnt);
		      gemm_buff=(doubleprecision *)ReAlloc(gemm_buff,(size_t)size_gemm_buff*sizeof(doubleprecision),
							   "DSYMselbinv:gemm_buff");
		      if (!Ii_cont || !Ji_cont) {
		         /* copy BLinv{i}.L(Ii,Ji) to buffer */
		         pr=gemm_buff;
			 p=0; q=0;
			 while (p<m_size && q<mi_size) {
			       ii=(integer)prBLI[p];
			       jj=(integer)prBLinvIi[q];
			       if (ii<jj)
			 	  p++;
			       else if (ii>jj)
				  q++;
			       else { /* indices match */

				  /* copy parts of the current row of BLinv{i}.L(Ii,:)
				     associated with Ji to gemm_buff */
				  pr3=pr;
				  pr2=prBLinvLi+q;
				  r=l; s=0;
				  while (s<ni_size) {
				        /* column Ji[s] occurs within I(l:j).
					   Recall that I(l:j) is a subset of Ji
					*/
				        if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
					   *pr3=*pr2;
					   pr3+=t;
					   r++;
					}
					s++;
					pr2+=mi_size;
				  } /* end while s */
				  pr++;

				  p++; q++;
			       } /* end if-elseif-else */
			 } /* end while p&q */
#ifdef PRINT_INFO
			 mexPrintf("DSYMselbinv: cached copy of BLinv{%d}.L(Ii,Ji)\nIndex set Ji:\n",i+1);
			 fflush(stdout);
			 r=l; s=0;
			 while (s<ni_size) {
			       if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
				  mexPrintf("%8d",(integer)prBLinvJi[s]);
				  r++;
			       }
			       s++;
			 } /* end while s */
			 mexPrintf("\nIndex set Ii:\n");
			 fflush(stdout);
			 p=0; q=0;
			 while (p<m_size && q<mi_size) {
			       ii=(integer)prBLI[p];
			       jj=(integer)prBLinvIi[q];
			       if (ii<jj)
			 	  p++;
			       else if (ii>jj)
				  q++;
			       else {
				  mexPrintf("%8d",ii);
				  p++; q++;
			       } /* end if-elseif-else */
			 } /* end while p&q */
			 mexPrintf("\n");
			 fflush(stdout);
			 for (p=0; p<t; p++) {
			     for (q=0; q<j-l+1; q++)
			         mexPrintf("%8.1le",gemm_buff[p+q*t]);
			     mexPrintf("\n");
			     fflush(stdout);
			 }
#endif

			 pr=gemm_buff; p=t;
		      }
		      else {
			 /* pointer to BLinv{i}.L(Ii,Ji) and LDA */
		         pr=prBLinvLi+i_first+mi_size*j_first; p=mi_size;
		      } /* end if-else */

		      if (!Ik_cont) {
		         /* copy BL{k}.L(Ik,:) to buffer */
		         if (!Ii_cont || !Ji_cont)
			    pr2=gemm_buff+t*(j-l+1);
			 else
			    pr2=gemm_buff;

			 r=0; s=0;
			 while (r<m_size && s<mi_size) {
			       ii=(integer)prBLI[r];
			       jj=(integer)prBLinvIi[s];
			       if (ii<jj)
				  r++;
			       else if (ii>jj)
				  s++;
			       else { /* indices match */

				  /* copy BL{k}.L(r,:) to buffer */
				  pr3=pr2;
				  pr4=prBLL+r;
				  for (ii=0; ii<n_size; ii++,pr3+=t,pr4+=m_size)
				      *pr3=*pr4;
				  pr2++;

				  r++; s++;
			       } /* end if-elseif-else */
			 } /* end while p&q */
#ifdef PRINT_INFO
			 mexPrintf("DSYMselbinv: cached copy of BL{%d}.L(Ik,:)\nIndex set J:\n",i+1);
			 fflush(stdout);
			 for (q=0; q<n_size; q++)
			     mexPrintf("%8d",prBLJ[q]);
			 mexPrintf("\nIndex set Ik:\n");
			 fflush(stdout);
			 r=0; s=0;
			 while (r<m_size && s<mi_size) {
			       ii=(integer)prBLI[r];
			       jj=(integer)prBLinvIi[s];
			       if (ii<jj)
			 	  r++;
			       else if (ii>jj)
				  s++;
			       else {
				  mexPrintf("%8d",ii);
				  r++; s++;
			       } /* end if-elseif-else */
			 } /* end while p&q */
			 mexPrintf("\n");
			 fflush(stdout);
			 if (!Ii_cont || !Ji_cont)
			    pr2=gemm_buff+t*(j-l+1);
			 else
			    pr2=gemm_buff;
			 for (r=0; r<t; r++) {
			     for (s=0; s<n_size; s++)
			         mexPrintf("%8.1le",pr2[r+s*t]);
			     mexPrintf("\n");
			     fflush(stdout);
			 }
#endif


			 /* pointer and LDC */
			 if (!Ii_cont || !Ji_cont)
			    pr2=gemm_buff+t*(j-l+1);
			 else
			    pr2=gemm_buff;
			 q=t;
		      }
		      else {
			 /* pointer to BL{k}.L(Ik,:) and LDC */
		         pr2=prBLL+k_first; q=m_size;
		      } /* end if-else */

		      /* call level 3 BLAS */
		      alpha=-1; beta=1.0;
		      ii=j-l+1;
		      dgemm_("T","N",&ii,&n_size,&t,
			     &alpha,
			     pr,&p,
			     pr2,&q,
			     &beta,
			     prBLinvL+l,&m_size,1,1);
#ifdef PRINT_INFO
		      mexPrintf("Ik=[");
		      r=0; s=0;
		      while (r<m_size && s<mi_size) {
		 	    if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			       r++;
			    else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			       s++;
			    else {
		               mexPrintf("%8d", r+1);
			       r++; s++;
		            }
		      }
		      mexPrintf("];\n");
		      mexPrintf("Ii=[");
		      r=0; s=0;
		      while (r<m_size && s<mi_size) {
		 	    if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			       r++;
			    else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			       s++;
			    else {
		               mexPrintf("%8d", s+1);
			       r++; s++;
		            }
		      }
		      mexPrintf("];\n");
		      r=l; s=0;
		      mexPrintf("Ji=[");
		      while (s<ni_size) {
		            if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
		               mexPrintf("%8d", s+1);
			       r++;
		            }
			    s++;
		      }
		      mexPrintf("];\n");
		      mexPrintf("DSYMselbinv: BLinv{%d}.L(%d:%d,:) = - BLinv{%d}.L(Ii,Ji)^T*BL{%d}.L(Ik,:)  + BLinv{%d}.L(%d:%d,:)\n",k+1,l+1,j+1,i+1,k+1,k+1,l+1,j+1);
		      for (jj=l; jj<=j; jj++) {
			  for (q=0; q<n_size; q++)
			      mexPrintf("%8.1le",prBLinvL[jj+m_size*q]);
			  mexPrintf("\n");
			  fflush(stdout);
		      }
#endif

		   } /* end if level3_BLAS */
		   else if (t) { /* it might not pay off, therefore we use a simple hand-coded loop */
		      /* BLinv{k}.L(l:j,:) -=  BLinv{i}.L(Ii,Ji)^T * BL{k}.L(Ik,:) */
#ifdef PRINT_INFO
		      mexPrintf("contribution from the strict upper triangular part, use hand-coded loops\n");
		      fflush(stdout);
#endif
		      p=0; q=0;
		      while (p<m_size && q<mi_size) {
			    ii=(integer)prBLI[p];
			    jj=(integer)prBLinvIi[q];
			    if (ii<jj)
			       p++;
			    else if (ii>jj)
			       q++;
			    else { /* row indices BL{k}.I[p]=BLinv{i}.I[q] match */
			       pr =prBLL+p;
			       pr3=prBLinvL+l;
			       /* BLinv{k}.L(l:j,:) -=  BLinv{i}.L(q,Ji)^T * BL{k}.L(p,:) */
			       for (ii=0; ii<n_size; ii++,pr+=m_size,pr3+=m_size-(j-l+1)) {
				   /* BLinv{k}.L(l:j,ii) -=  BLinv{i}.L(q,Ji)^T * BL{k}.L(p,ii) */
				   pr2=prBLinvLi+q;
				   r=l; s=0;
				   while (s<ni_size) {
				         /* column Ji[s] occurs within I(l:j).
					    Recall that I(l:j) is a subset of Ji
					 */
				         if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
					    /* BLinv{k}.L(r,ii)  -=  BLinv{i}.L(q,s)^T * BL{k}.L(p,ii) */
					    *pr3++ -= (*pr2)*(*pr);
					    r++;
					 }
					 s++;
					 pr2+=mi_size;
				   } /* end while s */
			       } /* end for ii */
			       p++; q++;
			    } /* end if-elseif-else */
		      } /* end while p&q */
#ifdef PRINT_INFO
		      mexPrintf("Ik=[");
		      r=0; s=0;
		      while (r<m_size && s<mi_size) {
			    if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			       r++;
			    else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			       s++;
			    else {
		               mexPrintf("%8d", r+1);
			       r++; s++;
			    }
		      }
		      mexPrintf("];\n");
		      mexPrintf("Ii=[");
		      r=0; s=0;
		      while (r<m_size && s<mi_size) {
			    if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			       r++;
			    else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			       s++;
			    else {
		               mexPrintf("%8d", s+1);
			       r++; s++;
			    }
		      }
		      mexPrintf("];\n");
		      r=l; s=0;
		      mexPrintf("Ji=[");
		      while (s<ni_size) {
		            if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
		               mexPrintf("%8d", s+1);
			       r++;
			    }
			    s++;
		      }
		      mexPrintf("];\n");
		      mexPrintf("DSYMselbinv: BLinv{%d}.L(%d:%d,:) = - BLinv{%d}.L(Ii,Ji)^T*BL{%d}.L(Ik,:)  + BLinv{%d}.L(%d:%d,:)\n",k+1,l+1,j+1,i+1,k+1,k+1,l+1,j+1);
		      for (p=l; p<=j; p++) {
			  for (q=0; q<n_size; q++)
			      mexPrintf("%8.1le",prBLinvL[p+m_size*q]);
			  mexPrintf("\n");
			  fflush(stdout);
		      }
#endif
		   } /* end if-else level3_BLAS */
		   else {
#ifdef PRINT_INFO
		      mexPrintf("contribution from the strict upper triangular part empty\n");
		      fflush(stdout);
#endif
		   }
		   /* end contribution from the strict upper triangular part */
		   /**********************************************************/
		   /**********************************************************/



  		   /**********************************************************/
  		   /**********************************************************/
		   /**********  contribution from the diagonal block *********/
		   /* BLinv{k}.L(l:j,:) = - BLinv{i}.D(Ji,Ji)  *BL{k}.L(l:j,:) + BLinv{k}.L(l:j,:)  */
		   /* determine amount of auxiliary memory */
		   copy_cnt=0;
		   /* level 3 BLAS have to use |Ji| x |Ji| buffer rather than BLinv{i}.D(Ji,Ji) */
		   if (!Ji_cont)
		      copy_cnt=(j-l+1)*(j-l+1);

		   /* it pays off to copy the data into one or two auxiliary buffers */
		   size_gemm_buff=MAX(size_gemm_buff,copy_cnt);
		   gemm_buff=(doubleprecision *)ReAlloc(gemm_buff,(size_t)size_gemm_buff*sizeof(doubleprecision),
							"DSYMselbinv:gemm_buff");
		   if (!Ji_cont) {
		      /* copy BLinv{i}.D(Ji,Ji) to buffer */
		      pr=gemm_buff;
		      pr2=prBLinvDi;
		      p=l; q=0;
		      while (q<ni_size) {
			    /* column Ji[q] occurs within I(l:j).
			       Recall that I(l:j) is a subset of Ji
			    */
			    if ((integer)prBLinvJi[q]==(integer)prBLI[p]) {
			       /* copy BLinv{i}.D(Ji,q) to buffer */
			       r=l; s=0;
			       while (s<ni_size) {
				     /* column Ji[s] occurs within I(l:j).
					Recall that I(l:j) is a subset of Ji
				     */
				     if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
				        *pr++=*pr2;
					r++;
				     }
				     pr2++;

				     s++;
			       } /* end while r&s */

			       p++;
			    } /* end if */
			    else
			       pr2+=ni_size;
			    q++;
		      } /* end while p&q */
#ifdef PRINT_INFO
		      mexPrintf("DSYMselbinv: cached copy of BLinv{%d}.D(Ji,Ji)\nIndices:\n",i+1);
		      fflush(stdout);
		      p=l; q=0;
		      while (q<ni_size) {
			    if ((integer)prBLinvJi[q]==(integer)prBLI[p]) {
			       mexPrintf("%8d",(integer)prBLinvJi[q]);
			       p++;
			    } /* end if */
			    q++;
		      } /* end while p&q */
		      mexPrintf("\n");
		      fflush(stdout);
		      for (p=0; p<j-l+1; p++) {
			  for (q=0; q<j-l+1; q++)
			      mexPrintf("%8.1le",gemm_buff[p+q*(j-l+1)]);
			  mexPrintf("\n");
			  fflush(stdout);
		      }

#endif
		      pr=gemm_buff; p=j-l+1;
		   }
		   else {
		      /* pointer to BLinv{i}.D(Ji,Ji) and LDA */
		      pr=prBLinvDi+j_first+j_first*ni_size; p=ni_size;
		   } /* end if-else */


		   /* call level 3 BLAS */
		   alpha=-1; beta=1.0;
		   ii=j-l+1;
		   dgemm_("N","N",&ii,&n_size,&ii,
			  &alpha,
			  pr,&p,
			  prBLL+l,&m_size,
			  &beta,
			  prBLinvL+l,&m_size,1,1);
#ifdef PRINT_INFO
		   mexPrintf("Ik=[");
		   r=0; s=0;
		   while (r<m_size && s<mi_size) {
		         if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			    r++;
			 else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			    s++;
			 else {
			    mexPrintf("%8d", r+1);
			    r++; s++;
			 }
		   }
		   mexPrintf("];\n");
		   mexPrintf("Ii=[");
		   r=0; s=0;
		   while (r<m_size && s<mi_size) {
		         if ((integer)prBLI[r]<(integer)prBLinvIi[s])
			    r++;
			 else if ((integer)prBLI[r]>(integer)prBLinvIi[s])
			    s++;
			 else {
			    mexPrintf("%8d", s+1);
			    r++; s++;
			 }
		   }
		   mexPrintf("];\n");
		   r=l; s=0;
		   mexPrintf("Ji=[");
		   while (s<ni_size) {
		         if ((integer)prBLinvJi[s]==(integer)prBLI[r]) {
			    mexPrintf("%8d", s+1);
			    r++;
			 }
			 s++;
		   }
		   mexPrintf("];\n");
		   mexPrintf("DSYMselbinv: BLinv{%d}.L(%d:%d,:) = - BLinv{%d}.D(Ji,Ji)  *BL{%d}.L(%d:%d,:)  + BLinv{%d}.L(Ik,:)\n",
			     k+1,l+1,j+1,i+1,k+1,l+1,j+1,k+1);
		   for (r=l; r<=j; r++) {
		       for (s=0; s<n_size; s++)
			   mexPrintf("%8.1le",prBLinvL[r+m_size*s]);
		       mexPrintf("\n");
		       fflush(stdout);
		   }
#endif

		   /* end contribution from the strict lower triangular part */
		   /**********************************************************/
		   /**********************************************************/
		} /* end if-else Ii_cont & Ik_cont & Ji_cont */

		/* advance to the next block column */
		l=j+1;
	  } /* end while l<p */

#ifdef PRINT_INFO
	  mexPrintf("DSYMselbinv: %d-th inverse sub-diagonal block computed\n", k+1);fflush(stdout);
	  mexPrintf("        ");
	  for (j=0; j<n_size; j++)
	      mexPrintf("%8d", (integer)prBLinvJ[j]);
	  mexPrintf("\n");fflush(stdout);
	  for (i=0; i<m_size; i++) {
	      mexPrintf("%8d", (integer)prBLinvI[i]);
	      for (j=0; j<n_size; j++)
		  mexPrintf("%8.1le", prBLinvL[i+j*m_size]);
	      mexPrintf("\n");fflush(stdout);
	  }
#endif


	  /* set each field in BLinv_block structure */
	  mxSetFieldByNumber(BLinv_block, (mwIndex)0, 2, BLinv_blockL);





	  /* structure element 3:  BLinv{k}.D */
	  /* create dense n_size x n_size BLinv{k}.D */
	  BLinv_blockD=mxCreateDoubleMatrix((mwSize)n_size,(mwSize)n_size, mxREAL);
	  prBLinvD=(double *)mxGetPr(BLinv_blockD);
	  /* copy strict lower triangular par */
	  for (j=0; j<n_size; j++) {
	      /* init strict upper triangular part with zeros */
	      for (i=0; i<j; i++)
		  *prBLinvD++=0.0;

	      /* diagonal (block) */
	      /* scalar case */
	      if (ia[j+1]-ia[j]==1) {
		 /* copy diagonal entry from sparse BD{k}.D */
		 *prBLinvD++=*prBDD++;
		 /* advance source BL{k}.D to the strict lower triangular part of column j */
		 prBLD+=j+1;
		 /* 1x1 block no pivoting */
		 ipiv[j]=j+1;
		 /* index for strict lower triangular part */
		 i++;
	      }
	      /* 2x2 case, column starts with row index j */
	      else if (ja[ia[j]]==j) {
		 /* copy diagonal entry from sparse BD{k}.D */
		 *prBLinvD++=*prBDD++;
		 /* copy sub-diagonal entry from sparse BD{k}.D */
		 *prBLinvD++=*prBDD++;
		 /* advance source BL{k}.D to the strict lower triangular part of
		    column j, excluding the sub-diagonal entry */
		 prBLD+=j+2;
		 /* 2x2 block (but no pivoting), remember that indexing for dsytri_
		    must be in FORTRAN style 1,...,n_size */
		 ipiv[j]=-(j+2);
		 /* index for strict lower triangular part excluding sub-diagonal entry */
		 i+=2;
	      }
	      /* 2x2 case, but second column since it starts with different row index
		 (must be j-1 in this case). This refers more or less to the scalar case */
	      else  {
		 /* skip super-diagonal entry of sparse BD{k}.D */
		 prBDD++;
		 /* copy diagonal entry from sparse BD{k}.D */
		 *prBLinvD++=*prBDD++;
		 /* advance source BL{k}.D to the strict lower triangular part of column j */
		 prBLD+=j+1;
		 /* 2x2 block (but no pivoting) */
		 ipiv[j]=-(j+1);
		 /* index for strict lower triangular part */
		 i++;
	      }
	      /* copy strict lower triangular part from BL{k}.D */
	      for (; i<n_size; i++)
		  *prBLinvD++=*prBLD++;
	  } /* end for j */
	  /* use LAPACK's dsytri_ for matrix inversion given the LDL^T decompositon */
	  prBLinvD=(double *)mxGetPr(BLinv_blockD);
	  j=0;
	  dsytri_("L", &n_size, prBLinvD, &n_size, ipiv, work, &j,1);
	  if (j<0) {
	     mexPrintf("the %d-th argument had an illegal value\n",-j);
	     mexErrMsgTxt("dsytri_ failed\n");
	  }
	  if (j>0) {
	     mexPrintf("D(%d,%d) = 0; the matrix is singular and its inverse could not be computed\n",j,j);
	     mexErrMsgTxt("dsytri_ failed\n");
	  }
	  /* for convenience copy data to the upper triangular part */
	  for (j=0; j<n_size; j++) {
	      /* advance to the diagonal part of column j */
	      prBLinvD+=j+1;

	      /* pointer to BLinv{k}.D(j,j+1) */
	      pr=prBLinvD+n_size-1;
	      /* copy data to the strict upper triangular part of row j */
	      for (i=j+1; i<n_size; i++, pr+=n_size)
		  *pr=*prBLinvD++;
	  } /* end for j */
	  prBLinvD=(double *)mxGetPr(BLinv_blockD);



	  /* BLinv{k}.D = - BL{k}.L^T *BLinv{k}.L + BLinv{k}.D */
	  /* call level 3 BLAS */
	  alpha=-1; beta=1.0;
	  ii=j-l+1;
	  if (m_size)
	     dgemm_("T","N",&n_size,&n_size,&m_size,
		    &alpha,
		    prBLL,&m_size,
		    prBLinvL,&m_size,
		    &beta,
		    prBLinvD,&n_size,1,1);

	  /* successively downdate "n" by the size "n_size" of the diagonal block */
	  sumn-=n_size;
	  /* extract diagonal part of BLinv{k}.D and symmetrize */
	  for (j=0; j<n_size; j++) {
	      /* advance to the diagonal part of column j */
	      prBLinvD+=j;
	      /* extract diagonal entry and advance to the strict lower triangular part of column j */
	      Dbuff[sumn+j]=*prBLinvD++;

	      /* pointer to BLinv{k}.D(j,j+1) */
	      pr=prBLinvD+n_size-1;
	      /* symmetrize data from the strict lower/upper triangular part of row j */
	      for (i=j+1; i<n_size; i++, pr+=n_size) {
		  val=(*pr+*prBLinvD)/2.0;
		  *pr=*prBLinvD++=val;
	      } /* end for i */
	  } /* end for j */
#ifdef PRINT_INFO
	  mexPrintf("DSYMselbinv: %d-th inverse diagonal block computed\n", k+1);fflush(stdout);
	  prBLinvD=(double *)mxGetPr(BLinv_blockD);
	  mexPrintf("        ");
	  for (j=0; j<n_size; j++)
	      mexPrintf("%8d", (integer)prBLinvJ[j]);
	  mexPrintf("\n");fflush(stdout);
	  for (i=0; i<n_size; i++) {
	      mexPrintf("%8d", (integer)prBLinvJ[i]);
	      for (j=0; j<n_size; j++)
		  mexPrintf("%8.1le", prBLinvD[i+j*n_size]);
	      mexPrintf("\n");fflush(stdout);
	  }
#endif


	  /* set each field in BLinv_block structure */
	  mxSetFieldByNumber(BLinv_block, (mwIndex)0, 3, BLinv_blockD);





	  /* finally set output BLinv{k} */
	  mxSetCell(BLinv,(mwIndex)k,BLinv_block);

          k--;
    } /* end while k>=0 */



    /* Compute D=Delta*(Delta*D(invperm)) */
    /* 1. compute inverse permutation */
    pr=(double *) mxGetPr(perm);
    for (i=0; i<n; i++) {
        j=*pr++;
	ipiv[j-1]=i;
    } /* end for i */
    /* 2. create memory for output vector D */
    plhs[0]=mxCreateDoubleMatrix((mwSize)n,(mwSize)1, mxREAL);
    pr=mxGetPr(plhs[0]);
    /* 3. reorder and rescale */
    pr2=(double *) mxGetPr(Delta);
    for (i=0; i<n; i++,pr2++) {
        *pr++=(*pr2)*(*pr2)*Dbuff[ipiv[i]];
    } /* end for i */



    /* finally release auxiliary memory */
    free(ipiv);
    free(work);
    free(Dbuff);
    free(gemm_buff);
    free(block);

#ifdef PRINT_INFO
    mexPrintf("DSYMselbinv: memory released\n");fflush(stdout);
#endif

    return;
}
