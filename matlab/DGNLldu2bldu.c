/* $Id: DGNLldu2bldu.c 807 2015-11-16 16:21:04Z bolle $ */
/* ========================================================================== */
/* === DGNLldu2bldu mexFunction ============================================= */
/* ========================================================================== */

/*
    Usage:

    Return cell arrays BL, BD and BUT that refer to a block triangular factorization

    Example:

    % for initializing parameters
    [BL,BD,BUT]=DGNLldu2bldu(L,D,UT,threshold,maxsize,tol)


    Authors:

	Matthias Bollhoefer, TU Braunschweig

    Date:

	November 14, 2015. ILUPACK V2.5.

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
#include <lapack.h>
#include <blas.h>

#define MAX_FIELDS 100
#define MAX(A,B) (((A)>=(B))?(A):(B))
#define MIN(A,B) (((A)>=(B))?(B):(A))
#define ELBOW    MAX(4.0,2.0)
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
    mwSize          dims[1];
    const char      *BLnames[]={"J","I", "L","D"};
    const char      *BDnames[]={"J","D"};
    mxArray         *L_input, *D_input, *UT_input, *threshold_input, *maxsize_input, *tol_input, *block_column,
                    *D_matrix,*L_matrix, *block_index, *BL, *BD, *BUT;
    integer         i,j,k,l,m,kk,ll,n,flag,nnz,p, cnt, cnti, cntj, cntij,
                    cntu, cntui, cntuj, cntuij,
                    *ia, *ja, *idxpos, *idxlst, *idxposu, *idxlstu, maxsize;
    doubleprecision tol, threshold, *prL, *prD, *prUT, mx, val, *pr;
    size_t          mrows, ncols;
    double          *L_valuesR, *D_valuesR, *UT_valuesR;
    mwIndex         *L_ja,                 /* row indices of input matrix L         */
                    *L_ia,                 /* column pointers of input matrix L     */
                    *D_ja,                 /* row indices of input matrix D         */
                    *D_ia,                 /* column pointers of input matrix D     */
                    *UT_ja,                /* row indices of input matrix UT        */
                    *UT_ia;                /* column pointers of input matrix UT    */


    if (nrhs!=6)
       mexErrMsgTxt("Six input arguments required.");
    else if (nlhs!=3)
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
    else if (!mxIsNumeric(prhs[5]))
       mexErrMsgTxt("Fifth input must be a number.");

    /* The first input must be a square matrix.*/
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
#ifdef PRINT_INFO
    mexPrintf("DGNLldu2bldu: input parameter L imported\n");fflush(stdout);
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
    D_valuesR=(double *) mxGetPr(D_input);
#ifdef PRINT_INFO
    mexPrintf("DGNLldu2bldu: input parameter D imported\n");fflush(stdout);
#endif

    /* The third input must be a square matrix.*/
    UT_input=(mxArray *)prhs[2];
    /* get size of input matrix UT */
    mrows=mxGetM(UT_input);
    ncols=mxGetN(UT_input);
    if (mrows!=ncols) {
       mexErrMsgTxt("Third input must be a square matrix.");
    }
    if (!mxIsSparse (UT_input)) {
        mexErrMsgTxt ("Third input matrix must be in sparse format.") ;
    }
    n=mrows;
    UT_ja     =(mwIndex *)mxGetIr(UT_input);
    UT_ia     =(mwIndex *)mxGetJc(UT_input);
    UT_valuesR=(double *) mxGetPr(UT_input);
#ifdef PRINT_INFO
    mexPrintf("DGNLldu2bldu: input parameter UT imported\n");fflush(stdout);
#endif


    /* The fourth input must a double number */
    threshold_input=(mxArray *)prhs[3];
    /* get size of input matrix D */
    mrows=mxGetM(threshold_input);
    ncols=mxGetN(threshold_input);
    if (1!=ncols || mrows!=1 || !mxIsNumeric(prhs[3])) {
       mexErrMsgTxt("Fourth argument must be scalar number.");
    }
    threshold=*mxGetPr(threshold_input);

#ifdef PRINT_INFO
    mexPrintf("DGNLldu2bldu: input parameter threshold imported\n");fflush(stdout);
#endif


    /* The fifth input must be a  number */
    maxsize_input=(mxArray *)prhs[4];
    /* get size of input matrix Delta */
    mrows=mxGetM(maxsize_input);
    ncols=mxGetN(maxsize_input);
    if (1!=ncols || mrows!=1 || !mxIsNumeric(prhs[4])) {
       mexErrMsgTxt("Fourth argument must be number.");
    }
    maxsize=*mxGetPr(maxsize_input);
#ifdef PRINT_INFO
    mexPrintf("DGNLldu2bldu: input parameter maxsize imported\n");fflush(stdout);
#endif

    /* The sixth input must be a scalar */
    tol_input=(mxArray *)prhs[5];
    /* get size of input matrix Delta */
    mrows=mxGetM(tol_input);
    ncols=mxGetN(tol_input);
    if (1!=ncols || mrows!=1 || !mxIsNumeric(prhs[5])) {
       mexErrMsgTxt("Sixth argument must be number.");
    }
    tol=*mxGetPr(tol_input);
#ifdef PRINT_INFO
    mexPrintf("DGNLldu2bldu: input parameter tol imported\n");fflush(stdout);
#endif

#ifdef PRINT_INFO
    mexPrintf("DGNLldu2bldu: input parameters imported\n");fflush(stdout);
#endif


    idxlst =(integer *)MAlloc((size_t)n*sizeof(integer),"DGNLldu2bldu:idxlst");
    idxpos =(integer *)CAlloc((size_t)n,sizeof(integer),"DGNLldu2bldu:idxpos");
    idxlstu=(integer *)MAlloc((size_t)n*sizeof(integer),"DGNLldu2bldu:idxlstu");
    idxposu=(integer *)CAlloc((size_t)n,sizeof(integer),"DGNLldu2bldu:idxposu");

    /* temporary output cell arrays of size n */
    dims[0]=n;
    BL =mxCreateCellArray((mwSize)1, dims);
    BD =mxCreateCellArray((mwSize)1, dims);
    BUT=mxCreateCellArray((mwSize)1, dims);


    i=0;
    k=0;
    while (i<n) {
#ifdef PRINT_CHECK
          mexPrintf("DGNLldu2bldu: analyze columns starting with i=%d\n",i+1);fflush(stdout);
#endif
          /* lower triangular part in column i */
          /* compute max_p>=i |l_{pi}| */
          mx=0.0;
          for (m=L_ia[i]; m<L_ia[i+1]; m++) {
	      val=FABS(L_valuesR[m]);
	      if (val>mx)
		 mx=val;
	  } /* end for i */
	  mx*=tol;
	  /* extract essential nonzero subdiagonal pattern */
	  cnt=0;
          for (m=L_ia[i]; m<L_ia[i+1]; m++) {
	      p=L_ja[m];
	      val=L_valuesR[m];
	      if (p>i && FABS(val)>=tol) {
	         idxlst[cnt]=p;
		 idxpos[p]=++cnt;
#ifdef PRINT_CHECK
		 mexPrintf("DGNLldu2bldu: store %d at position %d\n",p+1,cnt);fflush(stdout);
#endif
	      } /* end if */
	  } /* end for m */
	  cnti=cnt;

          /* upper triangular part in row i */
          /* compute max_p>=i |ut_{pi}| */
          mx=0.0;
          for (m=UT_ia[i]; m<UT_ia[i+1]; m++) {
	      val=FABS(UT_valuesR[m]);
	      if (val>mx)
		 mx=val;
	  } /* end for i */
	  mx*=tol;
	  /* extract essential nonzero subdiagonal pattern */
	  cntu=0;
          for (m=UT_ia[i]; m<UT_ia[i+1]; m++) {
	      p=UT_ja[m];
	      val=UT_valuesR[m];
	      if (p>i && FABS(val)>=tol) {
	         idxlstu[cntu]=p;
		 idxposu[p]=++cntu;
#ifdef PRINT_CHECK
		 mexPrintf("DGNLldu2bldu: store %d at position %d\n",p+1,cnt);fflush(stdout);
#endif
	      } /* end if */
	  } /* end for m */
	  cntui=cntu;

	  /* scan column j */
	  j=i+1;
	  flag=-1;
	  while (j<n && flag) {
#ifdef PRINT_CHECK
	        mexPrintf("DGNLldu2bldu: analyze column j=%d\n",j+1);fflush(stdout);
#endif
	        /* maximum number of columns exceeded? */
	        if (j-i+1>maxsize) {
		   flag=0;
		   j=j-1;
		}
		else {
		   /* strict lower triangular part in column j */
		   /* compute max_p |l_{pj}| */
		   mx=0.0;
		   for (m=L_ia[j]; m<L_ia[j+1]; m++) {
		       val=FABS(L_valuesR[m]);
		       if (val>mx)
			  mx=val;
		   } /* end for m */
		   mx*=tol;

		   /* extract essential nonzero subdiagonal pattern */
		   cntj=0;
		   cntij=0;
		   for (m=L_ia[j]; m<L_ia[j+1]; m++) {
		       p=L_ja[m];
		       val=L_valuesR[m];
		       if (p>j && FABS(val)>=tol) {
			  /* do we meet an already existing nonzero entry? */
			  if (idxpos[p])
			     cntij++;
			  else {
			     cntj++;
			     idxlst[cnt]=p;
			     idxpos[p]=++cnt;
#ifdef PRINT_CHECK
			     mexPrintf("DGNLldu2bldu: store %d at position %d\n",p+1,cnt);fflush(stdout);
#endif
			  } /* end if-else */
		       } /* end if */
		   } /* end for m */

		   /* strict upper triangular part in row j */
		   /* compute max_p |ut_{pj}| */
		   mx=0.0;
		   for (m=UT_ia[j]; m<UT_ia[j+1]; m++) {
		       val=FABS(UT_valuesR[m]);
		       if (val>mx)
			  mx=val;
		   } /* end for m */
		   mx*=tol;

		   /* extract essential nonzero subdiagonal pattern */
		   cntuj=0;
		   cntuij=0;
		   for (m=UT_ia[j]; m<UT_ia[j+1]; m++) {
		       p=UT_ja[m];
		       val=UT_valuesR[m];
		       if (p>j && FABS(val)>=tol) {
			  /* do we meet an already existing nonzero entry? */
			  if (idxposu[p])
			     cntuij++;
			  else {
			     cntuj++;
			     idxlstu[cntu]=p;
			     idxposu[p]=++cntu;
#ifdef PRINT_CHECK
			     mexPrintf("DGNLldu2bldu: store %d at position %d\n",p+1,cntu);fflush(stdout);
#endif
			  } /* end if-else */
		       } /* end if */
		   } /* end for m */

		   /* now cntij/cntuij refer to the intersection of indices,
		      cnti-cntij/cntui-cntuij refer to the entries not shared by column j,
		      cntj/cntuj refer to the entries only existing in column j
		      => we have cntij/cntuij common entries and in total
		         cnti+cntj/cntui+cntuj entries
		      there is a little exception concerning position j,  which
		      possibly has to be excluded when the columns merge
		   */
		   l=(idxpos[j])?-1:0;
		   ll=(idxposu[j])?-1:0;
		   if (cntij<threshold*(cnti+cntj+l) || cntuij<threshold*(cntui+cntuj+ll)) {
#ifdef PRINT_CHECK
		      mexPrintf("DGNLldu2bldu: intersection: %d/%d, union: %d/%d, do not merge\n",cntij,cntuij,cnti+cntj+l,cntui+cntuj+ll);fflush(stdout);
#endif
		      /* remove additional entries from column j */
		      for (m=0; m<cntj; m++) {
			  /* additional index from column j */
			  l=idxlst[cnti+m];
			  idxpos[l]=0;
		      } /* end for m */
		      cnt-=cntj;
		      /* remove additional entries from row j */
		      for (m=0; m<cntuj; m++) {
			  /* additional index from row j */
			  l=idxlstu[cntui+m];
			  idxposu[l]=0;
		      } /* end for m */
		      cntu-=cntuj;
#ifdef PRINT_INFO
		      mexPrintf("DGNLldu2bldu: additional entries removed\n");fflush(stdout);
#endif

		      flag=0;
		      j=j-1;
		   }
		   else {
		      /* remove index j from the list */
#ifdef PRINT_CHECK
		      mexPrintf("DGNLldu2bldu: intersection: %d/%d, union: %d/%d, merge\n",cntij,cntuij,cnti+cntj+l,cntui+cntuj+ll);fflush(stdout);
#endif
		      if (l) {
			 /* shuffle last entry to the former position of j */
			 /* reduce number of indices */
			 cnt--;
			 /* position of j inside idxlst */
			 l=idxpos[j]-1;
			 /* last nonzero index */
			 m=idxlst[cnt];
			 /* overwrite j */
			 idxlst[l]=m;
			 /* new position of m, shifted by one */
			 idxpos[m]=l+1;
			 /* index j is now removed */
			 idxpos[j]=0;
		      } /* end if l */
		      /* update current number of nonzeros */
		      cnti=cnt;

		      if (ll) {
			 /* shuffle last entry to the former position of j */
			 /* reduce number of indices */
			 cntu--;
			 /* position of j inside idxlst */
			 l=idxposu[j]-1;
			 /* last nonzero index */
			 m=idxlstu[cntu];
			 /* overwrite j */
			 idxlstu[l]=m;
			 /* new position of m, shifted by one */
			 idxposu[m]=l+1;
			 /* index j is now removed */
			 idxposu[j]=0;
		      } /* end if ll */
		      /* update current number of nonzeros */
		      cntui=cntu;

		      j=j+1;
		   } /* end if-else */
		} /* end if j-i+1>maxsize */
	  } /* end while j<n && flag */


	  /* unite columns i:j */
	  /* determine final column j */
	  if (flag)
	     j=n-1;
	  /* 2x2 diagonal block while truncating? */
	  if (j<n-1) {
	     m=0;
	     /* m is the index of the second nonzero entry in column j */
	     if (D_ia[j+1]-D_ia[j]>1)
	        m=D_ja[D_ia[j]+1];
	     if (m>j) {
#ifdef PRINT_CHECK
	        mexPrintf("DGNLldu2bldu: also add column j=%d\n",j+2);fflush(stdout);
#endif
	        /* For simplicity also add column j+1 */
	        j=j+1;
		/* strict lower triangular part in column j */
		/* compute max_p |l_{pj}| */
		mx=0.0;
		for (m=L_ia[j]; m<L_ia[j+1]; m++) {
		    val=FABS(L_valuesR[m]);
		    if (val>mx)
		       mx=val;
		} /* end for m */
		mx*=tol;

		/* extract essential nonzero subdiagonal pattern */
		for (m=L_ia[j]; m<L_ia[j+1]; m++) {
		    p=L_ja[m];
		    val=L_valuesR[m];
		    if (p>j && FABS(val)>=tol) {
		       /* only consider the case of additional fill */
		       if (!idxpos[p]) {
			  idxlst[cnt]=p;
			  idxpos[p]=++cnt;
		       } /* end if */
		    } /* end if */
		} /* end for m */

		/* strict upper triangular part in row j */
		/* compute max_p |ut_{pj}| */
		mx=0.0;
		for (m=UT_ia[j]; m<UT_ia[j+1]; m++) {
		    val=FABS(UT_valuesR[m]);
		    if (val>mx)
		       mx=val;
		} /* end for m */
		mx*=tol;

		/* extract essential nonzero subdiagonal pattern */
		for (m=UT_ia[j]; m<UT_ia[j+1]; m++) {
		    p=UT_ja[m];
		    val=UT_valuesR[m];
		    if (p>j && FABS(val)>=tol) {
		       /* only consider the case of additional fill */
		       if (!idxposu[p]) {
			  idxlstu[cntu]=p;
			  idxposu[p]=++cntu;
		       } /* end if */
		    } /* end if */
		} /* end for m */

		/* remove index j from the list */
		l=(idxpos[j])?-1:0;
		if (l) {
		   /* shuffle last entry to the former position of j */
		   /* reduce number of indices */
		   cnt--;
		   /* position of j inside idxlst */
		   l=idxpos[j]-1;
		   /* last nonzero index */
		   m=idxlst[cnt];
		   /* overwrite j */
		   idxlst[l]=m;
		   /* new position of m, shifted by one */
		   idxpos[m]=l+1;
		   /* index j is now removed */
		   idxpos[j]=0;
		} /* end if l */

		/* remove index j from the list */
		ll=(idxposu[j])?-1:0;
		if (ll) {
		   /* shuffle last entry to the former position of j */
		   /* reduce number of indices */
		   cntu--;
		   /* position of j inside idxlst */
		   l=idxposu[j]-1;
		   /* last nonzero index */
		   m=idxlstu[cntu];
		   /* overwrite j */
		   idxlstu[l]=m;
		   /* new position of m, shifted by one */
		   idxposu[m]=l+1;
		   /* index j is now removed */
		   idxposu[j]=0;
		} /* end if l */
	     } /* end if 2x2 case */
	  } /* end if j<n-1 */


#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: create output structures for BL\n");fflush(stdout);
#endif

	  /* set up new block column with four elements for BL: J, I, L, D */
	  block_column=mxCreateStructMatrix((mwSize)1, (mwSize)1, 4, BLnames);

	  /* structure element 0:  BL.J */
	  block_index=mxCreateDoubleMatrix((mwSize)1,(mwSize)(j-i+1), mxREAL);
	  pr=mxGetPr(block_index);
#ifdef PRINT_CHECK
	  mexPrintf("DGNLldu2bldu: store column indices\n");fflush(stdout);
#endif
	  for (m=0; m<=j-i; m++) {
	      pr[m]=i+m+1;
#ifdef PRINT_CHECK
	      mexPrintf("%6d",i+m+1);
#endif
	  } /* end for m */
#ifdef PRINT_CHECK
	  mexPrintf("\n");fflush(stdout);
#endif
	  /* set each field in output structure */
	  mxSetFieldByNumber(block_column, (mwIndex)0, 0, block_index);

	  /* structure element 1:  BL.I */
	  block_index=mxCreateDoubleMatrix((mwSize)1,(mwSize)cnt, mxREAL);
	  pr=mxGetPr(block_index);
	  /* remove check marks */
	  for (m=0; m<cnt; m++) {
	      l=idxlst[m];
	      idxpos[l]=0;
	  } /* end for m */
#ifdef PRINT_CHECK
	  for (m=0; m<n; m++) {
	      if (idxpos[m]) {
		 mexPrintf("DGNLldu2bldu: idxpos[%d]=%d !=0 !!!\n",m+1,idxpos[m]);fflush(stdout);
	    }
	  }
#endif
	  /* sort indices of "idxlst" in increasing order */
	  qqsorti(idxlst,idxpos,&cnt);
	  /* clear buffer "idxpos" */
	  for (m=0; m<cnt; m++)
	      idxpos[m]=0;
#ifdef PRINT_CHECK
	  for (m=0; m<n; m++) {
	      if (idxpos[m]) {
		 mexPrintf("DGNLldu2bldu: idxpos[%d]=%d !=0 !!!!\n",m+1,idxpos[m]);fflush(stdout);
	    }
	  }
#endif
	  /* transfer sorted indices and store location */
#ifdef PRINT_CHECK
	  mexPrintf("DGNLldu2bldu: store row indices\n");fflush(stdout);
#endif
	  for (m=0; m<cnt; m++) {
	      l=idxlst[m];
	      pr[m]=l+1;
	      idxpos[l]=m+1;
#ifdef PRINT_CHECK
	      mexPrintf("%6d",l+1);
#endif
	  } /* end for m */
#ifdef PRINT_CHECK
	  mexPrintf("\n");fflush(stdout);
#endif
	  /* set each field in output structure */
	  mxSetFieldByNumber(block_column, (mwIndex)0, 1, block_index);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: block_column.I set\n");fflush(stdout);
#endif


	  /* structure element 2:  BL.L */
	  L_matrix=mxCreateDoubleMatrix((mwSize)cnt,(mwSize)(j-i+1), mxREAL);
	  prL=mxGetPr(L_matrix);
	  /* structure element 3:  BL.D */
	  D_matrix=mxCreateDoubleMatrix((mwSize)(j-i+1),(mwSize)(j-i+1), mxREAL);
	  prD=mxGetPr(D_matrix);
	  /* init with zeros */
	  for (m=0; m<cnt*(j-i+1); m++)
	      prL[m]=0.0;
	  for (m=0; m<(j-i+1)*(j-i+1); m++)
	      prD[m]=0.0;
	  /* extract nonzeros from columns i:j */
	  for (m=i; m<=j; m++, prL+=cnt, prD+=j-i+1) {
	      for (l=L_ia[m]; l<L_ia[m+1]; l++) {
		  /* index p of L_{p,m} */
		  p=L_ja[l];
		  /* diagonal index */
		  if (p==m)
		     prD[p-i]=1.0;
		  /* index p is located inside the strict lower triangular part
		     of the diagonal block L_{i:j,i:j}
		  */
		  else if (m<p && p<=j)
		     prD[p-i]=L_valuesR[l];
		  /* index p must be part of L_{j+1:n,i:j} */
		  else if (p>j) {
		     /* is the index in the output structure present? */
		     kk=idxpos[p];
		     /* mexPrintf("index %d located position %d, value=%8.1le\n",p+1,kk,L_valuesR[l]);fflush(stdout); */
		     if (kk) {
		        /* kk-1 is the position of the row index */
		        prL[kk-1]=L_valuesR[l];
		     } /* end if */
		  } /* end if-elseif-elseif */
	      } /* end for l */
	  } /* end for m */
	  /* clear positions from "idxpos" */
	  for (m=0; m<cnt; m++) {
	      l=idxlst[m];
	      idxpos[l]=0;
	  } /* end for m */
#ifdef PRINT_CHECK
	  mexPrintf("DGNLldu2bldu: lower triangular diagonal block\n");fflush(stdout);
	  prD=mxGetPr(D_matrix);
	  for (m=0; m<=j-i; m++) {
	      for (l=0; l<=j-i; l++) {
		  mexPrintf("%8.1le",prD[m+l*(j-i+1)]);
	      }
	      mexPrintf("\n");fflush(stdout);
	  }
#endif
#ifdef PRINT_CHECK
	  mexPrintf("DGNLldu2bldu: sub-diagonal block\n");fflush(stdout);
	  prL=mxGetPr(L_matrix);
	  for (m=0; m<cnt; m++) {
	      for (l=0; l<=j-i; l++) {
		  mexPrintf("%8.1le",prL[m+l*cnt]);
	      }
	      mexPrintf("\n");fflush(stdout);
	  }
#endif
#ifdef PRINT_CHECK
	  for (m=0; m<n; m++) {
	      if (idxpos[m]) {
		 mexPrintf("DGNLldu2bldu: idxpos[%d]=%d !=0 !!!\n",m+1,idxpos[m]);fflush(stdout);
	    }
	  }
#endif
	  /* build L_{i:j,i:j}^{-T}L_{j+1:n,i:j}^T using BLAS function DTRSV */
	  if (cnt && j>i) {
	     prL=mxGetPr(L_matrix);
	     prD=mxGetPr(D_matrix);
	     m=j-i+1;
	     /* prD is lower triangular, we have to solve with the transpose and it
		has unit diagonal part:
		-> "L", "T", "U"
		Furthermore, its size and its leading dimension is j-i+1
		prL is a cnt x (j-i+1) matrix, but we have to use its transpose which
		requires an index jump of cnt rather than 1
	     */
	     for (l=0; l<cnt; l++)
	         dtrsv_("L", "T", "U", &m, prD, &m, prL+l, &cnt, 1,1,1);
#ifdef PRINT_CHECK
	     mexPrintf("DGNLldu2bldu: sub-diagonal block after DTRSV\n");fflush(stdout);
	     prL=mxGetPr(L_matrix);
	     for (m=0; m<cnt; m++) {
	         for (l=0; l<=j-i; l++) {
		     mexPrintf("%8.1le",prL[m+l*cnt]);
		 }
		 mexPrintf("\n");fflush(stdout);
	     }
#endif
	  } /* end if cnt & j>i */

	  /* set each field in output structure */
	  mxSetFieldByNumber(block_column, (mwIndex)0, 2, L_matrix);
	  mxSetFieldByNumber(block_column, (mwIndex)0, 3, D_matrix);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: block_column.L/D set\n");fflush(stdout);
#endif

	  /* assign block column to cell array BL */
	  mxSetCell(BL,(mwIndex)k,block_column);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: BL{%d} set\n",k+1);fflush(stdout);
#endif



#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: create output structures for BD\n");fflush(stdout);
#endif

	  /* set up new block column with two elements J, D */
	  block_column=mxCreateStructMatrix((mwSize)1, (mwSize)1, 2, BDnames);

	  /* structure element 0:  J */
	  block_index=mxCreateDoubleMatrix((mwSize)1,(mwSize)(j-i+1), mxREAL);
	  pr=mxGetPr(block_index);
#ifdef PRINT_CHECK
	  mexPrintf("DGNLldu2bldu: store column indices\n");fflush(stdout);
#endif
	  for (m=0; m<=j-i; m++) {
	      pr[m]=i+m+1;
#ifdef PRINT_CHECK
	      mexPrintf("%6d",i+m+1);
#endif
	  } /* end for m */
#ifdef PRINT_CHECK
	  mexPrintf("\n");fflush(stdout);
#endif
	  /* set each field in output structure */
	  mxSetFieldByNumber(block_column, (mwIndex)0, 0, block_index);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: block_column.J set\n");fflush(stdout);
#endif

	  /* structure element 1:  D */
	  nnz=D_ia[j+1]-D_ia[i];
	  D_matrix=mxCreateSparse((mwSize)(j-i+1),(mwSize)(j-i+1), (mwSize)nnz, mxREAL);
	  ia=(mwIndex *)mxGetJc(D_matrix);
	  ja=(mwIndex *)mxGetIr(D_matrix);
	  prD=(double *)mxGetPr(D_matrix);
	  kk=0;
	  for (m=0; m<=j-i; m++) {
	      ia[m]=kk;
	      for (l=D_ia[m+i]; l<D_ia[m+i+1]; l++)  {
		  ja[kk]=D_ja[l]-i;
		  prD[kk++]=D_valuesR[l];
	      } /* end for l */
	  } /* end for m */
	  ia[m]=kk;
	  /* set each field in output structure */
	  mxSetFieldByNumber(block_column, (mwIndex)0, 1, D_matrix);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: block_column.L/D set\n");fflush(stdout);
#endif

	  /* assign block column to cell array BD */
	  mxSetCell(BD,(mwIndex)k,block_column);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: BD{%d}.D set\n",k+1);fflush(stdout);
#endif



#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: create output structures for BUT\n");fflush(stdout);
#endif

	  /* set up new block column with four elements for BUT: J, I, L, D */
	  block_column=mxCreateStructMatrix((mwSize)1, (mwSize)1, 4, BLnames);

	  /* structure element 0:  BUT.J */
	  block_index=mxCreateDoubleMatrix((mwSize)1,(mwSize)(j-i+1), mxREAL);
	  pr=mxGetPr(block_index);
	  for (m=0; m<=j-i; m++) {
	      pr[m]=i+m+1;
	  } /* end for m */
	  /* set each field in output structure */
	  mxSetFieldByNumber(block_column, (mwIndex)0, 0, block_index);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: block_column.J set\n");fflush(stdout);
#endif

	  /* structure element 1:  BUT.I */
	  block_index=mxCreateDoubleMatrix((mwSize)1,(mwSize)cntu, mxREAL);
	  pr=mxGetPr(block_index);
	  /* remove check marks */
	  for (m=0; m<cntu; m++) {
	      l=idxlstu[m];
	      idxposu[l]=0;
	  } /* end for m */
#ifdef PRINT_CHECK
	  for (m=0; m<n; m++) {
	      if (idxposu[m]) {
		 mexPrintf("DGNLldu2bldu: idxpos[%d]=%d !=0 !!!\n",m+1,idxposu[m]);fflush(stdout);
	    }
	  }
#endif
	  /* sort indices of "idxlst" in increasing order */
	  qqsorti(idxlstu,idxposu,&cntu);
	  /* clear buffer "idxpos" */
	  for (m=0; m<cntu; m++)
	      idxposu[m]=0;
#ifdef PRINT_CHECK
	  for (m=0; m<n; m++) {
	      if (idxposu[m]) {
		 mexPrintf("DGNLldu2bldu: idxpos[%d]=%d !=0 !!!!\n",m+1,idxposu[m]);fflush(stdout);
	    }
	  }
#endif
	  /* transfer sorted indices and store location */
#ifdef PRINT_CHECK
	  mexPrintf("DGNLldu2bldu: store row indices\n");fflush(stdout);
#endif
	  for (m=0; m<cntu; m++) {
	      l=idxlstu[m];
	      pr[m]=l+1;
	      idxposu[l]=m+1;
#ifdef PRINT_CHECK
	      mexPrintf("%6d",l+1);
#endif
	  } /* end for m */
#ifdef PRINT_CHECK
	  mexPrintf("\n");fflush(stdout);
#endif
	  /* set each field in output structure */
	  mxSetFieldByNumber(block_column, (mwIndex)0, 1, block_index);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: block_column.I set\n");fflush(stdout);
#endif


	  /* structure element 2:  BUT.L */
	  L_matrix=mxCreateDoubleMatrix((mwSize)cntu,(mwSize)(j-i+1), mxREAL);
	  prL=mxGetPr(L_matrix);
	  /* structure element 3:  BUT.D */
	  D_matrix=mxCreateDoubleMatrix((mwSize)(j-i+1),(mwSize)(j-i+1), mxREAL);
	  prD=mxGetPr(D_matrix);
	  /* init with zeros */
	  for (m=0; m<cntu*(j-i+1); m++)
	      prL[m]=0.0;
	  for (m=0; m<(j-i+1)*(j-i+1); m++)
	      prD[m]=0.0;
	  /* extract nonzeros from columns i:j */
	  for (m=i; m<=j; m++, prL+=cntu, prD+=j-i+1) {
	      for (l=UT_ia[m]; l<UT_ia[m+1]; l++) {
		  /* index p of UT_{p,m} */
		  p=UT_ja[l];
		  /* diagonal index */
		  if (p==m)
		     prD[p-i]=1.0;
		  /* index p is located inside the strict lower triangular part
		     of the diagonal block UT_{i:j,i:j}
		  */
		  else if (m<p && p<=j)
		     prD[p-i]=UT_valuesR[l];
		  /* index p must be part of UT_{j+1:n,i:j} */
		  else if (p>j) {
		     /* is the index in the output structure present? */
		     kk=idxposu[p];
		     /* mexPrintf("index %d located position %d, value=%8.1le\n",p+1,kk,UT_valuesR[l]);fflush(stdout); */
		     if (kk) {
		        /* kk-1 is the position of the row index */
		        prL[kk-1]=UT_valuesR[l];
		     } /* end if */
		  } /* end if-elseif-elseif */
	      } /* end for l */
	  } /* end for m */
	  /* clear positions from "idxposu" */
	  for (m=0; m<cntu; m++) {
	      l=idxlstu[m];
	      idxposu[l]=0;
	  } /* end for m */
#ifdef PRINT_CHECK
	  mexPrintf("DGNLldu2bldu: lower triangular diagonal block\n");fflush(stdout);
	  prD=mxGetPr(D_matrix);
	  for (m=0; m<=j-i; m++) {
	      for (l=0; l<=j-i; l++) {
		  mexPrintf("%8.1le",prD[m+l*(j-i+1)]);
	      }
	      mexPrintf("\n");fflush(stdout);
	  }
#endif
#ifdef PRINT_CHECK
	  mexPrintf("DGNLldu2bldu: sub-diagonal block\n");fflush(stdout);
	  prL=mxGetPr(L_matrix);
	  for (m=0; m<cntu; m++) {
	      for (l=0; l<=j-i; l++) {
		  mexPrintf("%8.1le",prL[m+l*cntu]);
	      }
	      mexPrintf("\n");fflush(stdout);
	  }
#endif
#ifdef PRINT_CHECK
	  for (m=0; m<n; m++) {
	      if (idxposu[m]) {
		 mexPrintf("DGNLldu2bldu: idxpos[%d]=%d !=0 !!!\n",m+1,idxposu[m]);fflush(stdout);
	    }
	  }
#endif
	  /* build UT_{i:j,i:j}^{-T}UT_{j+1:n,i:j}^T using BLAS function DTRSV */
	  if (cntu && j>i) {
	     prL=mxGetPr(L_matrix);
	     prD=mxGetPr(D_matrix);
	     m=j-i+1;
	     /* prD is lower triangular, we have to solve with the transpose and it
		has unit diagonal part:
		-> "L", "T", "U"
		Furthermore, its size and its leading dimension is j-i+1
		prL is a cnt x (j-i+1) matrix, but we have to use its transpose which
		requires an index jump of cnt rather than 1
	     */
	     for (l=0; l<cntu; l++)
	         dtrsv_("L", "T", "U", &m, prD, &m, prL+l, &cntu, 1,1,1);
#ifdef PRINT_CHECK
	     mexPrintf("DGNLldu2bldu: sub-diagonal block after DTRSV\n");fflush(stdout);
	     prL=mxGetPr(L_matrix);
	     for (m=0; m<cntu; m++) {
	         for (l=0; l<=j-i; l++) {
		     mexPrintf("%8.1le",prL[m+l*cntu]);
		 }
		 mexPrintf("\n");fflush(stdout);
	     }
#endif
	  } /* end if cnt & j>i */

	  /* set each field in output structure */
	  mxSetFieldByNumber(block_column, (mwIndex)0, 2, L_matrix);
	  mxSetFieldByNumber(block_column, (mwIndex)0, 3, D_matrix);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: block_column.L/D set\n");fflush(stdout);
#endif


	  /* assign block column to cell array BUT */
	  mxSetCell(BUT,(mwIndex)k,block_column);
#ifdef PRINT_INFO
	  mexPrintf("DGNLldu2bldu: BUT{%d} set\n",k+1);fflush(stdout);
#endif



          k=k+1;
          i=j+1;

    } /* end while i<n */

    dims[0]=k;
    plhs[0]=mxCreateCellArray((mwSize)1, dims);
    plhs[1]=mxCreateCellArray((mwSize)1, dims);
    plhs[2]=mxCreateCellArray((mwSize)1, dims);
    for (m=0; m<k; m++) {
        block_column=mxGetCell(BL,(mwIndex)m);
        mxSetCell(plhs[0],(mwIndex)m,block_column);
        mxSetCell(BL,(mwIndex)m,NULL);

        block_column=mxGetCell(BD,(mwIndex)m);
        mxSetCell(plhs[1],(mwIndex)m,block_column);
        mxSetCell(BD,(mwIndex)m,NULL);

        block_column=mxGetCell(BUT,(mwIndex)m);
        mxSetCell(plhs[2],(mwIndex)m,block_column);
        mxSetCell(BUT,(mwIndex)m,NULL);
    } /* end for m */


    /* release memory */
    mxDestroyArray(BL);
    mxDestroyArray(BD);
    mxDestroyArray(BUT);
    free(idxlst);
    free(idxpos);
    free(idxlstu);
    free(idxposu);

#ifdef PRINT_INFO
    mexPrintf("DGNLldu2bldu: memory released\n");fflush(stdout);
#endif

    return;
}
