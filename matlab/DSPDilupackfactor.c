/* ========================================================================== */
/* === AMGfactor mexFunction ================================================ */
/* ========================================================================== */

/*
    Usage:

    Return the structure 'options' and preconditioner 'PREC' for ILUPACK V2.5
    
    Example:

    % for initializing parameters
    [PREC, options, rcomflag,S,tv] = DSPDilupackfactor(A,options,PRE,tv);



    Authors:

	Matthias Bollhoefer, TU Braunschweig

    Date:

	February 24, 2015. ILUPACK V2.5.

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

#define MAX_FIELDS 100
#define SIZE_PNAMES         22
#define SIZE_PPNAMES        23
#define SIZE_PNAMES_OMPROOT 12
#define SIZE_PPNAMES_OMPTAB  4
/* #define PRINT_INFO */

/* #define PRINT_INFO_OMP */
/* #define PRINT_INFO_OMP1 */


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
    Dmat A;
    DAMGlevelmat *PRE, *current, *pre;
    SAMGlevelmat *SPRE, *scurrent, *spre;
    DILUPACKparam *param;
    integer n, nlev, ispartial, buffldl;
    int  tv_exists, tv_field;

    const char **fnames;
    const char *pnames[]= {"n","nB", "L","D","U", "E","F", "rowscal","colscal", "p","invq",
			   "param","ptr", "isreal","isdefinite","issymmetric","ishermitian",
			   "issingle", "A_H", "errorL", "errorU", "errorS"};
    const char *ppnames[]= {"n","nB", "L","D","U", "E","F", "rowscal","colscal", "p","invq",
			    "A_H", "errorL", "errorU", "errorS", "ispartial","A",
			    "isreal","isdefinite","issymmetric","ishermitian",
			    "issingle","p_local"};
    const char *pnames_omproot[]= {"n","p","invq","param","ptr",
				   "isreal","isdefinite","issymmetric","ishermitian",
				   "issingle","ompparts","omptab"};

    const char *pnames_omptab[]= {"tree","chld","brth","hght"};


    const mwSize  *dims;
    const mwSize  mydims[]={1,1};
    mxClassID  *classIDflags; 
    mxArray    *tmp, *fout, *PRE_input, *tv_input, *A_input, *options_input,
               *PRE_output, *options_output, *S_output, *tv_output, 
               *ompparts, *ppre, *pomptab;
    char       *pdata, *input_buf, *output_buf;
    mwSize     nnz, ndim, buflen, nompparts;
    mwIndex    jstruct, *irs, *jcs;
    int        ifield, jfield, status, nfields,  ierr,i,j,k,l,m,iomp;
    integer    *ibuff, *istack,jj;
    size_t     sizebuf, mrows, ncols;
    double     dbuf, *A_valuesR, *convert, *sr, *pr;
    float      *spr;
    mwIndex    *A_ja,                 /* row indices of input matrix A */
               *A_ia;                 /* column pointers of input matrix A */
    

    if (nrhs!=2 && nrhs!=4)
       mexErrMsgTxt("Two/four input arguments required.");
    else if (nlhs!=2 && nlhs!=5)
       mexErrMsgTxt("wrong number of output arguments.");
    else if (!mxIsStruct(prhs[1]))
       mexErrMsgTxt("Second input must be a structure.");
    else if (!mxIsNumeric(prhs[0]))
       mexErrMsgTxt("First input must be a matrix.");

    /* The first input must be a square matrix.*/
    A_input = (mxArray *) prhs [0] ;
    /* get size of input matrix A */
    mrows = mxGetM (A_input) ;
    ncols = mxGetN (A_input) ;
    nnz = mxGetNzmax(A_input);
    if (mrows!=ncols) {
       mexErrMsgTxt("First input must be a square matrix.");
    }
    if (!mxIsSparse (A_input))
    {
        mexErrMsgTxt ("ILUPACK: input matrix must be in sparse format.") ;
    }



    /* copy input matrix to sparse row format */
    A.nc=A.nr=mrows;
    A.ia=(integer *) MAlloc((size_t)(A.nc+1)*sizeof(integer),"DSPDilupackfactor");
    A.ja=(integer *) MAlloc((size_t)nnz     *sizeof(integer),"DSPDilupackfactor");
    A.a =(double *)  MAlloc((size_t)nnz     *sizeof(double), "DSPDilupackfactor");


    A_ja         = (mwIndex *) mxGetIr(A_input);
    A_ia         = (mwIndex *) mxGetJc(A_input);
    A_valuesR    = (double *)  mxGetPr(A_input);

    /* -------------------------------------------------------------------- */
    /* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
    /*     notation.                                                        */
    /* -------------------------------------------------------------------- */
    
    /*
    for (i = 0 ; i < ncols ; i++)
      for (j = A_ia[i] ; j < A_ia[i+1] ; j++)
	printf("i=%d ja=%d  A.real=%e\n", i+1,  A_ja[j]+1, A_valuesR[j]);
    */

    A.ia[0]=1;
    for (i = 0 ; i < ncols ; i++) {
        A.ia[i+1]=A.ia[i];
	for (j = A_ia[i] ; j < A_ia[i+1] ; j++) {
	    k=A_ja[j];
	    if (k>=i) {
	       l=A.ia[i+1]-1;
	       A.ja[l]=k+1;
	       A.a [l]=A_valuesR[j];
	       A.ia[i+1]=l+2;
	    }
	}
    }

    /*
    for (i = 0 ; i < A.nr ; i++)
      for (j = A.ia[i]-1 ; j < A.ia[i+1]-1 ; j++)
	  printf("i=%d ja=%d  A.real=%e\n", i+1,  A.ja[j], A.a[j]);
    */

    param=(DILUPACKparam *)MAlloc((size_t)sizeof(DILUPACKparam),"DSPDilupackfactor:param");
    DSPDAMGinit(&A,param);

    /* Get second input arguments */
    options_input=(mxArray *)prhs[1];
    nfields = mxGetNumberOfFields(options_input);

    /* Allocate memory  for storing classIDflags */
    classIDflags = (mxClassID *) mxCalloc((size_t)nfields, (size_t)sizeof(mxClassID));
    
    /* allocate memory  for storing pointers */
    fnames = mxCalloc((size_t)nfields, (size_t)sizeof(*fnames));

    /* Get field name pointers */
    for (ifield = 0; ifield<nfields; ifield++) {
        fnames[ifield] = mxGetFieldNameByNumber(options_input,ifield);
    }

    /* import data */
    tv_exists=0;
    tv_field=-1;
    for (ifield = 0; ifield < nfields; ifield++) {
	tmp = mxGetFieldByNumber(options_input,0,ifield);
	classIDflags[ifield] = mxGetClassID(tmp); 

	ndim = mxGetNumberOfDimensions(tmp);
	dims = mxGetDimensions(tmp);

	/* Create string/numeric array */
	if (classIDflags[ifield] == mxCHAR_CLASS) {
	   /* Get the length of the input string. */
	   buflen = (mxGetM(tmp) * mxGetN(tmp)) + 1;

	   /* Allocate memory for input and output strings. */
	   input_buf = (char *) mxCalloc((size_t)buflen, (size_t)sizeof(char));

	   /* Copy the string data from tmp into a C string 
	      input_buf. */
	   status = mxGetString(tmp, input_buf, buflen);
	   
	   if (!strcmp("amg",fnames[ifield])) {
              if (strcmp(param->amg,input_buf)) {
		 param->amg=(char *)MAlloc((size_t)buflen*sizeof(char),
					   "ilupackfactor");
		 strcpy(param->amg,input_buf);
	      }
	   }
	   else if (!strcmp("presmoother",fnames[ifield])) {
              if (strcmp(param->presmoother,input_buf)) {
		 param->presmoother=(char *)MAlloc((size_t)buflen*sizeof(char),
						   "ilupackfactor");
		 strcpy(param->presmoother,input_buf);
	      }
	   }
	   else if (!strcmp("postsmoother",fnames[ifield])) {
              if (strcmp(param->postsmoother,input_buf)) {
		 param->postsmoother=(char *)MAlloc((size_t)buflen*sizeof(char),
						    "ilupackfactor");
		 strcpy(param->postsmoother,input_buf);
	      }
	   }
	   else if (!strcmp("typecoarse",fnames[ifield])) {
              if (strcmp(param->typecoarse,input_buf)) {
		 param->typecoarse=(char *)MAlloc((size_t)buflen*sizeof(char),
						  "ilupackfactor");
		 strcpy(param->typecoarse,input_buf);
	      }
	   }
	   else if (!strcmp("typetv",fnames[ifield])) {
              if (strcmp(param->typetv,input_buf)) {
		 param->typetv=(char *)MAlloc((size_t)buflen*sizeof(char),
					      "ilupackfactor");
		 strcpy(param->typetv,input_buf);
	      }
	      /* 'static' or 'dynamic' test vector */
	      if (strcmp("none",input_buf)) tv_exists=-1;
	   }
	   else if (!strcmp("FCpart",fnames[ifield])) {
              if (strcmp(param->FCpart,input_buf)) {
		 param->FCpart=(char *)MAlloc((size_t)buflen*sizeof(char),
					      "ilupackfactor");
		 strcpy(param->FCpart,input_buf);
	      }
	   }
	   else if (!strcmp("solver",fnames[ifield])) {
              if (strcmp(param->solver,input_buf)) {
		 param->solver=(char *)MAlloc((size_t)buflen*sizeof(char),
					      "ilupackfactor");
		 strcpy(param->solver,input_buf);
	      }
	   }
	   else if (!strcmp("ordering",fnames[ifield])) {
              if (strcmp(param->ordering,input_buf)) {
	         param->ordering=(char *)MAlloc((size_t)buflen*sizeof(char),
						"ilupackfactor");
		 strcpy(param->ordering,input_buf);
	      }
	   }
	   else {
	      /* mexPrintf("%s ignored\n",fnames[ifield]);fflush(stdout); */
	   }
	} 
	else {
	   if (!strcmp("elbow",fnames[ifield])) {
	      param->elbow=*mxGetPr(tmp);
	   }
	   else if (!strcmp("lfilS",fnames[ifield])) {
	      param->lfilS=*mxGetPr(tmp);
	   }
	   else if (!strcmp("lfil",fnames[ifield])) {
	      param->lfil=*mxGetPr(tmp);
	   }
	   else if (!strcmp("maxit",fnames[ifield])) {
	      param->maxit=*mxGetPr(tmp);
	   }
	   else if (!strcmp("droptolS",fnames[ifield])) {
	      param->droptolS=*mxGetPr(tmp);
	   }
	   else if (!strcmp("droptolc",fnames[ifield])) {
	      param->droptolc=*mxGetPr(tmp);
	   }
	   else if (!strcmp("droptol",fnames[ifield])) {
	      param->droptol=*mxGetPr(tmp);
	   }
	   else if (!strcmp("condest",fnames[ifield])) {
	      param->condest=*mxGetPr(tmp);
	   }
	   else if (!strcmp("restol",fnames[ifield])) {
	      param->restol=*mxGetPr(tmp);
	   }
	   else if (!strcmp("npresmoothing",fnames[ifield])) {
	      param->npresmoothing=*mxGetPr(tmp);
	   }
	   else if (!strcmp("npostmoothing",fnames[ifield])) {
	      param->npostsmoothing=*mxGetPr(tmp);
	   }
	   else if (!strcmp("ncoarse",fnames[ifield])) {
	      param->ncoarse=*mxGetPr(tmp);
	   }
	   else if (!strcmp("matching",fnames[ifield])) {
	      param->matching=*mxGetPr(tmp);
	   }
	   else if (!strcmp("nrestart",fnames[ifield])) {
	      param->nrestart=*mxGetPr(tmp);
	   }
	   else if (!strcmp("damping",fnames[ifield])) {
	      param->damping=*mxGetPr(tmp);
	   }
	   else if (!strcmp("contraction",fnames[ifield])) {
	      param->contraction=*mxGetPr(tmp);
	   }
	   else if (!strcmp("tv",fnames[ifield])) {
	      tv_field=ifield;
	   }
	   else if (!strcmp("mixedprecision",fnames[ifield])) {
	      param->mixedprecision=*mxGetPr(tmp);
	   }
	   else if (!strcmp("coarsereduce",fnames[ifield])) {
	      if (*mxGetPr(tmp)!=0.0)
		 param->flags|=COARSE_REDUCE;
	      else
		 param->flags&=~COARSE_REDUCE;
	   }
	   else if (!strcmp("decoupleconstraints",fnames[ifield])) {
	      if (*mxGetPr(tmp)>0.0)
		 param->flags|=DECOUPLE_CONSTRAINTSHH;
	      else 
		 if (*mxGetPr(tmp)<0.0)
		    param->flags|=DECOUPLE_CONSTRAINTS;
		 else
		    param->flags&=~(DECOUPLE_CONSTRAINTS|DECOUPLE_CONSTRAINTSHH);
	   }
	   else if (!strcmp("nthreads",fnames[ifield])) {
	      param->nthreads=*mxGetPr(tmp);
	   }
	   else if (!strcmp("loadbalancefactor",fnames[ifield])) {
	      param->loadbalancefactor=*mxGetPr(tmp);
	   }
	   else {
	     /* mexPrintf("%s ignored\n",fnames[ifield]);fflush(stdout); */
	   }
	}
    }
    if (param->droptolS>0.125*param->droptol) {
       mexPrintf("!!! ILUPACK Warning !!!\n");
       mexPrintf("`param.droptolS' is recommended to be one order of magnitude less than `param.droptol'\n");
    } 

    if (tv_exists && tv_field>=0 && 
        (nrhs==2   || (nrhs==4 && mxIsNumeric(prhs[2])))) {
       tmp=mxGetFieldByNumber(options_input,0,tv_field);
       param->tv=mxGetPr(tmp);
    }
    mxFree(fnames);


    PRE=(DAMGlevelmat *)MAlloc((size_t)sizeof(DAMGlevelmat),"DSPDilupackfactor");

#ifdef PRINT_INFO
    mexPrintf("DSPDilupackfactor: factorize matrix\n");fflush(stdout);
#endif
    if (nrhs==4) {
       
       /* at least the second time that we call this routine */
       if (!mxIsNumeric(prhs[2])) {
	  /* import pointer to the preconditioner */
	  PRE_input = (mxArray*) prhs [2] ;
	  /* get number of levels of input preconditioner structure `PREC' */
          /* nlev=mxGetN(PRE_input); */
	  nfields = mxGetNumberOfFields(PRE_input);
	  /* allocate memory  for storing pointers */
	  fnames = mxCalloc((size_t)nfields, (size_t)sizeof(*fnames));
	  for (ifield = 0; ifield < nfields; ifield++) {
	      fnames[ifield] = mxGetFieldNameByNumber(PRE_input,ifield);
	      /* check whether `PREC.ptr' exists */
	      if (!strcmp("ptr",fnames[ifield])) {
		 /* field `ptr' */
		 tmp = mxGetFieldByNumber(PRE_input,0,ifield);
		 pdata = mxGetData(tmp);
		 memcpy(&PRE, pdata, (size_t)sizeof(size_t));
	      }
	      else if (!strcmp("param",fnames[ifield])) {
	         /* field `param' */
	 	 tmp = mxGetFieldByNumber(PRE_input,0,ifield);
		 pdata = mxGetData(tmp);
		 memcpy(&param, pdata, (size_t)sizeof(size_t));
	      }
	  }
	  mxFree(fnames);


	  /* import pointer to the test vector */
	  tv_input=(mxArray*)prhs[3] ;
	  /* size of the test vector */
	  j=mxGetM(tv_input);
	  pr=mxGetPr(tv_input);
	  memcpy(param->tv,pr,j*sizeof(double));

	  /* remap A if necessary */
	  if (A.nr==param->mstack[0].nr) {
	    param->mstack[0].ia=A.ia;
	    param->mstack[0].ja=A.ja;
	    param->mstack[0].a =A.a;
	    /* mexPrintf("original matrix remapped\n");fflush(stdout); */
	  }
       }
    }
#ifdef PRINT_INFO
    /* printf("options.isdefinite    %d\n",param->isdefinite); */
    printf("options.matching         %d\n",param->matching);
    printf("options.ordering         %s\n",param->ordering);
    printf("options.droptol          %12.4le\n",param->droptol);
    printf("options.droptolS         %12.4le\n",param->droptolS);
    printf("options.droptolc         %12.4le\n",param->droptolc);
    printf("options.condest          %12.4le\n",param->condest);
    printf("options.restol           %12.4le\n",param->restol);
    printf("options.maxit            %d\n",param->maxit);
    printf("options.elbow            %12.4le\n",param->elbow);
    printf("options.lfil             %d\n",param->lfil);
    printf("options.lfilS            %d\n",param->lfilS);
    printf("options.typetv           %s\n",param->typetv);
    printf("options.amg              %s\n",param->amg);
    printf("options.npresmoothing    %d\n",param->npresmoothing);
    printf("options.npostsmoothing   %d\n",param->npostsmoothing);
    printf("options.ncoarse          %d\n",param->ncoarse);
    printf("options.presmoother      %s\n",param->presmoother);
    printf("options.postsmoother     %s\n",param->postsmoother);
    printf("options.FCpart           %s\n",param->FCpart);
    printf("options.typecoarse       %s\n",param->typecoarse);
    printf("options.solver           %s\n",param->solver);
    printf("options.damping          %12.4le\n",param->damping);
    printf("options.contraction      %12.4le\n",param->contraction);
    printf("options.nrestart         %d\n",param->nrestart);
    printf("options.mixedprecision   %d\n",param->mixedprecision);
    printf("options.niter            %d\n",param->niter);
    printf("options.nthreads         %d\n",param->nthreads);
    printf("options.loadbalancefactor%d\n",param->loadbalancefactor);
#endif

    ierr=DSPDAMGfactor(&A, PRE, param);
#ifdef PRINT_INFO
    mexPrintf("DSPDilupackfactor: matrix factored\n");fflush(stdout);
    if (param->rcomflag!=0) {
       mexPrintf("DSPDilupackfactor: request for reverse communication\n");fflush(stdout);
    }
#endif

    if (nlhs==5 && ierr==0) {
       plhs[2]=mxCreateDoubleMatrix((mwSize)1, (mwSize)1, mxREAL);
       pr = mxGetPr(plhs[2]);
       *pr=param->rcomflag;
	
       /* mexPrintf("rcomflag set\n");fflush(stdout); */

       if (param->rcomflag!=0) {
	  nnz=param->A.ia[param->A.nr]-1;
	  /* mexPrintf("extract sparse matrix %ld,%ld,%ld\n",param->A.nr,param->A.nr,nnz);fflush(stdout); */
	  plhs[3]=mxCreateSparse((mwSize)param->A.nr,(mwSize)param->A.nc, nnz, mxREAL);
	  S_output=plhs[3];

	  sr  = (double *)  mxGetPr(S_output);
	  irs = (mwIndex *) mxGetIr(S_output);
	  jcs = (mwIndex *) mxGetJc(S_output);

	  k=0;
	  for (i=0; i<param->A.nr; i++) {
	      jcs[i]=k;
	      /* strict lower triangular part */
	      for (j=param->A.ia[i]-1; j<param->A.ia[i+1]-1; j++) {
		  irs[k] =param->A.ja[j]-1;
		  sr[k++]=param->A.a[j];
	      }
	  } /* end for i */
	  jcs[i]=k;

	  /* mexPrintf("extract test vector\n");fflush(stdout);*/
	  /* output test vector */
	  plhs[4]=mxCreateDoubleMatrix((mwSize)A.nr,(mwSize)1, mxREAL);
	  tv_output=plhs[4];
	  pr = mxGetPr(tv_output);
	  for (i=0; i<A.nr; i++) 
	      *pr++=param->tv[i];
       }
       else {
	  plhs[3]=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	  plhs[4]=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
       }
    }



    if (ierr) {
       nlhs=0;
       /* finally release memory of the preconditioner */
       DSPDAMGdelete(&A, PRE, param);
       free(A.ia);
       free(A.ja);
       free(A.a);
       free(PRE);
       free(param);
    }

    
    switch (ierr) {
    case  0: /* perfect! */
      break;
    case -1: /* Error. input matrix may be wrong.
		(The elimination process has generated a
		row in L or U whose length is .gt.  n.) */
      mexErrMsgTxt("ILUPACK error, data may be wrong.");
    case -2: /* The matrix L overflows the array alu */
      mexErrMsgTxt("memory overflow, please increase `options.elbow' and retry");
    case -3: /* The matrix U overflows the array alu */
      mexErrMsgTxt("memory overflow, please increase `options.elbow' and retry");
    case -4: /* Illegal value for lfil */
      mexErrMsgTxt("Illegal value for `options.lfil'");
    case -5: /* zero row encountered */
      mexErrMsgTxt("zero row encountered, please reduce `options.droptol'");
    case -6: /* zero column encountered */
      mexErrMsgTxt("zero column encountered, please reduce `options.droptol'");
    case -7: /* buffers too small */
      mexErrMsgTxt("memory overflow, please increase `options.elbow' and retry");
    default: /* zero pivot encountered at step number ierr */
      mexErrMsgTxt("zero pivot encountered, please reduce `options.droptol'");
    } /* end switch */
    if (ierr) {
      plhs[0]=NULL;
      return;
    }



    /* prepare a struct matrices for output */
    nfields = mxGetNumberOfFields(options_input);

    /* allocate memory  for storing pointers */
    fnames = mxCalloc((size_t)nfields, (size_t)sizeof(*fnames));
    /* Get field name pointers */
    for (ifield=0; ifield<nfields; ifield++) {
         fnames[ifield] = mxGetFieldNameByNumber(options_input,ifield);
    }

    plhs[1] = mxCreateStructMatrix((mwSize)1,(mwSize)1, nfields, fnames);
    if (plhs[1]==NULL)
       mexErrMsgTxt("Could not create structure mxArray");
    options_output=plhs[1];

    /* export data */
    for (ifield = 0; ifield<nfields; ifield++) {
         tmp = mxGetFieldByNumber(options_input,0,ifield);
	 classIDflags[ifield] = mxGetClassID(tmp); 

	 ndim = mxGetNumberOfDimensions(tmp);
	 dims = mxGetDimensions(tmp);

	 /* Create string/numeric array */
	 if (classIDflags[ifield] == mxCHAR_CLASS) {
	    if (!strcmp("amg",fnames[ifield])) {
	       output_buf = (char *) mxCalloc((size_t)strlen(param->amg)+1, (size_t)sizeof(char));
	       strcpy(output_buf,param->amg);
	       fout = mxCreateString(output_buf);
	   }
	   else if (!strcmp("presmoother",fnames[ifield])) {
	      output_buf = (char *) mxCalloc((size_t)strlen(param->presmoother)+1, (size_t)sizeof(char));
	      strcpy(output_buf,param->presmoother);
	      fout = mxCreateString(output_buf);
	   }
	   else if (!strcmp("postsmoother",fnames[ifield])) {
	      output_buf = (char *) mxCalloc((size_t)strlen(param->postsmoother)+1, (size_t)sizeof(char));
	      strcpy(output_buf,param->postsmoother);
	      fout = mxCreateString(output_buf);
	   }
	   else if (!strcmp("typecoarse",fnames[ifield])) {
	      output_buf = (char *) mxCalloc((size_t)strlen(param->typecoarse)+1, (size_t)sizeof(char));
	      strcpy(output_buf,param->typecoarse);
	      fout = mxCreateString(output_buf);
	   }
	   else if (!strcmp("typetv",fnames[ifield])) {
	      output_buf = (char *) mxCalloc((size_t)strlen(param->typetv)+1, (size_t)sizeof(char));
	      strcpy(output_buf,param->typetv);
	      fout = mxCreateString(output_buf);
	   }
	   else if (!strcmp("FCpart",fnames[ifield])) {
	      output_buf = (char *) mxCalloc((size_t)strlen(param->FCpart)+1, (size_t)sizeof(char));
	      strcpy(output_buf,param->FCpart);
	      fout = mxCreateString(output_buf);
	   }
	   else if (!strcmp("solver",fnames[ifield])) {
	      output_buf = (char *) mxCalloc((size_t)strlen(param->solver)+1, (size_t)sizeof(char));
	      strcpy(output_buf, param->solver);
	      fout = mxCreateString(output_buf);
	   }
	   else if (!strcmp("ordering",fnames[ifield])) {
	      output_buf = (char *) mxCalloc((size_t)strlen(param->ordering)+1, (size_t)sizeof(char));
	      strcpy(output_buf, param->ordering);
	      fout = mxCreateString(output_buf);
	   }
	   else {
	      /* Get the length of the input string. */
	      buflen = (mxGetM(tmp) * mxGetN(tmp)) + 1;

	      /* Allocate memory for input and output strings. */
	      input_buf  = (char *) mxCalloc((size_t)buflen, (size_t)sizeof(char));
	      output_buf = (char *) mxCalloc((size_t)buflen, (size_t)sizeof(char));
	      
	      /* Copy the string data from tmp into a C string 
		 input_buf. */
	      status = mxGetString(tmp, input_buf, buflen);
	      
	      sizebuf = buflen*sizeof(char);
	      memcpy(output_buf, input_buf, sizebuf);
	      fout = mxCreateString(output_buf);
	   }
	} 
	else {
	   fout = mxCreateNumericArray((mwSize)ndim, dims, 
				       classIDflags[ifield], mxREAL);
	   pdata = mxGetData(fout);

	   sizebuf = mxGetElementSize(tmp);
	   if (!strcmp("elbow",fnames[ifield])) {
	      dbuf=param->elbow;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("lfilS",fnames[ifield])) {
	      dbuf=param->lfilS;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("lfil",fnames[ifield])) {
	      dbuf=param->lfil;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("maxit",fnames[ifield])) {
	      dbuf=param->maxit;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("droptolc",fnames[ifield])) {
	      dbuf=param->droptolc;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("droptolS",fnames[ifield])) {
	      dbuf=param->droptolS;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("droptol",fnames[ifield])) {
	      dbuf=param->droptol;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("condest",fnames[ifield])) {
	      dbuf=param->condest;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("restol",fnames[ifield])) {
	      dbuf=param->restol;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("npresmoothing",fnames[ifield])) {
	      dbuf=param->npresmoothing;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("npostmoothing",fnames[ifield])) {
	      dbuf=param->npostsmoothing;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("ncoarse",fnames[ifield])) {
	      dbuf=param->ncoarse;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("matching",fnames[ifield])) {
	      dbuf=param->matching;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("nrestart",fnames[ifield])) {
	      dbuf=param->nrestart;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("damping",fnames[ifield])) {
	      dbuf=param->damping;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("contraction",fnames[ifield])) {
	      dbuf=param->contraction;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("mixedprecision",fnames[ifield])) {
	      dbuf=param->mixedprecision;
	      memcpy(pdata, &dbuf, (size_t)sizebuf);
	   }
	   else if (!strcmp("nthreads",fnames[ifield])) {
	      dbuf=param->nthreads;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else if (!strcmp("loadbalancefactor",fnames[ifield])) {
	      dbuf=param->loadbalancefactor;
	      memcpy(pdata, &dbuf, sizebuf);
	   }
	   else {
	      memcpy(pdata, mxGetData(tmp), sizebuf);
	   }
	}


	/* Set each field in options output structure */
	mxSetFieldByNumber(options_output, (mwIndex)0, ifield, fout);
    }


    mxFree(fnames);
    mxFree(classIDflags);

#ifdef PRINT_INFO_OMP
    mexPrintf("structure param extracted\n");fflush(stdout);
#endif

    /* single-threaded case */
    if (param->nthreads==1) {
       plhs[0] = mxCreateStructMatrix((mwSize)1,(mwSize)PRE->nlev, SIZE_PNAMES, pnames);
       if (plhs[0]==NULL)
	   mexErrMsgTxt("Could not create structure mxArray\n");
       PRE_output=plhs[0];


       current=PRE;
       if (PRE->issingle) {
	  SPRE=(SAMGlevelmat *)PRE;
	  scurrent=SPRE;
       }
       n=A.nr;
       ibuff=  (integer *)MAlloc((size_t)n*sizeof(integer),"DSPDilupackfactor:ibuff");
       istack =(integer *)MAlloc((size_t)n*sizeof(integer),"DSPDilupackfactor:istack");

       convert=(double *) MAlloc((size_t)n*sizeof(double), "DSPDilupackfactor:convert");

       for (jstruct=0; jstruct<PRE->nlev; jstruct++) {
      
	   /* 1. field `n' */
	   ifield=0;

	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);
	
	   if (PRE->issingle) 
	      *pr=scurrent->n;
	   else
	      *pr=current->n;
       
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 2. field `nB' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);

	   if (PRE->issingle) 
	      *pr=scurrent->nB;
	   else
	      *pr=current->nB;
       
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 3. field `L' */
	   ++ifield;
	   /*
	   if (((PRE->issingle)?scurrent->LU.ja:current->LU.ja)!=NULL)
	      mexPrintf("before %d\n",(PRE->issingle)?scurrent->LU.ja[scurrent->nB]:current->LU.ja[current->nB]);
	   */
	   if (param->rcomflag==0) {
	      /* switched to full-matrix processing? */
	      if (jstruct==PRE->nlev-1 && ((PRE->issingle)?scurrent->LU.ja:current->LU.ja)==NULL) {

		 if (PRE->issingle) {
		    fout=mxCreateDoubleMatrix((mwSize)scurrent->nB,(mwSize)scurrent->nB, mxREAL);
		    /* position of the diagonal entry inside the factorization */
		    for (i=0; i<scurrent->nB; i++) 
		        ibuff[i]=i*scurrent->nB-(i*(i-1))/2;
		    sr=mxGetPr(fout);
		    spr=scurrent->LU.a;
		 }
		 else {
		    fout=mxCreateDoubleMatrix((mwSize)current->nB, (mwSize)current->nB,  mxREAL);
		    /* position of the diagonal entry inside the factorization */
		    for (i=0; i<current->nB; i++) 
		        ibuff[i]=i*current->nB-(i*(i-1))/2;
		    sr=mxGetPr(fout);
		    pr=current->LU.a;
		 }


		 for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
		     /* init strict upper triangular part with zeros */
		     for (j=0; j<i; j++) {
		         *sr++=0;
		     }

		     /* additional permutation? */
#ifndef USE_LAPACK_DRIVER
		     l=((PRE->issingle)?scurrent->LU.ia[i]:current->LU.ia[i])-1;
#else
		     l=i;
#endif

		     /* extract diagonal entry */
		     if (PRE->issingle) {
#ifndef USE_LAPACK_DRIVER
		       dbuf=1.0/(double)spr[ibuff[l]];
#else
		       dbuf=(double)spr[ibuff[l]];
#endif
		     }
		     else {
#ifndef USE_LAPACK_DRIVER
		        dbuf=1.0/pr[ibuff[l]];
#else
			dbuf=pr[ibuff[l]];
#endif
		     }

		     *sr++=dbuf*dbuf;


		     /* extract strict lower triangular part */
		     for (j=i+1; j<((PRE->issingle)?scurrent->nB:current->nB); j++) {

		         /* additional permutation? */
#ifndef USE_LAPACK_DRIVER
		         k=((PRE->issingle)?scurrent->LU.ia[j]:current->LU.ia[j])-1;
#else
			 k=j;
#endif

			 /* position of the associated off-diagonal entry a_{ij} */
			 m=k-l;
			 if (m>0)
			    m=ibuff[l]+m;
			 else
			    m=ibuff[k]-m;
			 if (PRE->issingle) 
			    *sr++=(double)spr[m]*dbuf;
			 else
			    *sr++=pr[m]*dbuf;
		     }
		 }	   

		 /*	   
		 for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
#ifndef USE_LAPACK_DRIVER
		     l=((PRE->issingle)?scurrent->LU.ia[i]:current->LU.ia[i])-1;
#else
		     l=i;
#endif
		     for (j=0; j<i; j++) {
#ifndef USE_LAPACK_DRIVER
		         k=((PRE->issingle)?scurrent->LU.ia[j]:current->LU.ia[j])-1;
#else
			 k=j;
#endif
			 m=k-l;
			 if (m>0)
			    m=ibuff[l]+m;
			 else
			    m=ibuff[k]-m;
			 *sr++=pr[m];

		     }
#ifndef USE_LAPACK_DRIVER
		     *sr++=1.0/pr[ibuff[l]];
#else
		     *sr++=pr[ibuff[l]];
#endif
		     for (j=i+1; j<((PRE->issingle)?scurrent->nB:current->nB); j++) {
		         *sr++=0;
		     }
		 }	   
		 */

		 /* set each field in preconditioner output structure */
		 mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	      }
	      else {
		 if (PRE->issingle)
		    nnz=scurrent->LU.nnz+1-scurrent->LU.ja[0]+scurrent->nB;
		 else
		    nnz=current->LU.nnz+1-current->LU.ja[0]+current->nB;

		 if (param->flags&COARSE_REDUCE) {
		    if (PRE->issingle)
		       fout=mxCreateSparse((mwSize)scurrent->nB,(mwSize)scurrent->nB, nnz, mxREAL);
		    else
		       fout=mxCreateSparse((mwSize)current->nB, (mwSize)current->nB,  nnz, mxREAL);
		 }
		 else {
		    if (PRE->issingle) {
		       fout=mxCreateSparse((mwSize)scurrent->n,(mwSize)scurrent->nB, nnz, mxREAL);
		       buffldl=scurrent->LU.ja[scurrent->nB];
		       scurrent->LU.ja[scurrent->nB]=scurrent->LU.nnz+1;
		    }
		    else {
		       fout=mxCreateSparse((mwSize)current->n, (mwSize)current->nB,  nnz, mxREAL);
		       buffldl=current->LU.ja[current->nB];
		       current->LU.ja[current->nB]=current->LU.nnz+1;
		    }
		 }

		 /*
		 if (((PRE->issingle)?scurrent->LU.ja:current->LU.ja)!=NULL)
		    mexPrintf("intermediate %d\n",(PRE->issingle)?scurrent->LU.ja[scurrent->nB]:current->LU.ja[current->nB]);
		 */

		 sr =(double *) mxGetPr(fout);
		 irs=(mwIndex *)mxGetIr(fout);
		 jcs=(mwIndex *)mxGetJc(fout);
	      
		 k=0;
		 if (PRE->issingle) {
		    for (i=0; i<scurrent->nB; i++) {
		        /* extract diagonal entry */
		        jcs[i]=k;
			irs[k]=i;
			sr[k++]=1.0/scurrent->LU.a[i];

			j =scurrent->LU.ja[i]-1;
			jj=scurrent->LU.ja[i+1]-scurrent->LU.ja[i];
			Sqsort(scurrent->LU.a+j,scurrent->LU.ja+j,istack,&jj);

			for (j=scurrent->LU.ja[i]-1; j<scurrent->LU.ja[i+1]-1; j++) {
			    irs[k] =scurrent->LU.ja[j]-1;
			    sr[k++]=scurrent->LU.a[j];
			}
		    }
		 }
		 else {
		    for (i=0; i<current->nB; i++) {
		        /* extract diagonal entry */
		        jcs[i]=k;
			irs[k]=i;
			sr[k++]=1.0/current->LU.a[i];

			j =current->LU.ja[i]-1;
			jj=current->LU.ja[i+1]-current->LU.ja[i];
			Dqsort(current->LU.a+j,current->LU.ja+j,istack,&jj);

			for (j=current->LU.ja[i]-1; j<current->LU.ja[i+1]-1; j++) {
			    irs[k] =current->LU.ja[j]-1;
			    sr[k++]=current->LU.a[j];
			}
		    }
		 }
		 jcs[i]=k;

		 if (!(param->flags&COARSE_REDUCE)) {
		    if (PRE->issingle)
		       scurrent->LU.ja[scurrent->nB]=buffldl;
		    else
		       current->LU.ja[current->nB]=buffldl;
		 }

		 /*
		 if (((PRE->issingle)?scurrent->LU.ja:current->LU.ja)!=NULL)
		    mexPrintf("final %d\n",(PRE->issingle)?scurrent->LU.ja[scurrent->nB]:current->LU.ja[current->nB]);
		 */
		 /* set each field in preconditioner output structure */
		 mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	      }
	   }
	   else {
	      fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	      /* set each field in preconditioner output structure */
	      mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	   }


	   /* 4. field `D' */
	   ++ifield;
	   if (param->rcomflag==0) {
	      if (PRE->issingle)
		 fout=mxCreateSparse((mwSize)scurrent->nB,(mwSize)scurrent->nB, (mwSize)scurrent->nB, mxREAL);
	      else
		 fout=mxCreateSparse((mwSize)current->nB, (mwSize)current->nB,  (mwSize)current->nB, mxREAL);

	      sr =(double *) mxGetPr(fout);
	      irs=(mwIndex *)mxGetIr(fout);
	      jcs=(mwIndex *)mxGetJc(fout);
	   
	      for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
		  jcs[i]=i;
		  irs[i]=i;
	      }
	      jcs[i]=i;
	   
	      /* switched to full-matrix processing */
	      if (jstruct==PRE->nlev-1 &&
		 ((PRE->issingle)?scurrent->LU.ja:current->LU.ja)==NULL) {

		 if (PRE->issingle)
		    spr=scurrent->LU.a;
		 else
		    pr=current->LU.a;
		 for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
#ifndef USE_LAPACK_DRIVER
		     l=((PRE->issingle)?scurrent->LU.ia[i]:current->LU.ia[i])-1;
		     dbuf=1.0/((PRE->issingle)?(double)spr[ibuff[l]]:pr[ibuff[l]]);
#else
		     l=i;
		     dbuf=((PRE->issingle)?(double)spr[ibuff[l]]:pr[ibuff[l]]);
#endif
		     sr[i]=1.0/(dbuf*dbuf);
		 }	   
	      }
	      else {
		 for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
#ifndef USE_LAPACK_DRIVER
		     sr[i]=1.0/((PRE->issingle)?scurrent->LU.a[i]:current->LU.a[i]);
#else
		     sr[i]=((PRE->issingle)?scurrent->LU.a[i]:current->LU.a[i]);
#endif
		 }
	      }
	      /* set each field in preconditioner output structure */
	      mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	   }
	   else {
	      fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	      /* set each field in preconditioner output structure */
	      mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	   }


	
	   /* 5. field `U' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	   
	   
	   
	   /* 6. field `E', E=F^T */
	   ++ifield;
	   if (param->rcomflag==0) {
	      if (jstruct<PRE->nlev-1){ 
		
		 if (param->flags&COARSE_REDUCE) {
		    if (PRE->issingle) {
		       nnz=scurrent->F.ia[scurrent->nB]-1;
		       fout=mxCreateSparse((mwSize)n-scurrent->nB,(mwSize)scurrent->nB, nnz, mxREAL);
		    }
		    else {
		       nnz=current->F.ia[current->nB]-1;
		       fout=mxCreateSparse((mwSize)n-current->nB, (mwSize)current->nB,  nnz, mxREAL);
		    }
		    sr =(double *) mxGetPr(fout);
		    irs=(mwIndex *)mxGetIr(fout);
		    jcs=(mwIndex *)mxGetJc(fout);
		 
		    k=0;
		    for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
		        jcs[i]=k;
			if (PRE->issingle) {

			   j =scurrent->F.ia[i]-1;
			   jj=scurrent->F.ia[i+1]-scurrent->F.ia[i];
			   Sqsort(scurrent->F.a+j,scurrent->F.ja+j,istack,&jj);

			   for (j=scurrent->F.ia[i]-1; j<scurrent->F.ia[i+1]-1; j++) {
			       irs[k] =scurrent->F.ja[j]-1;
			       sr[k++]=scurrent->F.a[j];
			   }
			}
			else {

			   j =current->F.ia[i]-1;
			   jj=current->F.ia[i+1]-current->F.ia[i];
			   Dqsort(current->F.a+j,current->F.ja+j,istack,&jj);

			   for (j=current->F.ia[i]-1; j<current->F.ia[i+1]-1; j++) {
			       irs[k] =current->F.ja[j]-1;
			       sr[k++]=current->F.a[j];
			   }
			}
		    }
		    jcs[i]=k;
		 }
		 else {
		    nnz=0;
		    if (PRE->issingle)
		       fout=mxCreateSparse((mwSize)n-scurrent->nB,(mwSize)scurrent->nB, nnz, mxREAL);
		    else
		       fout=mxCreateSparse((mwSize)n-current->nB, (mwSize)current->nB,  nnz, mxREAL);
		 }
		 /* set each field in preconditioner output structure */
		 mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	      }
	   }
	   else {
	      fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	      /* set each field in preconditioner output structure */
	      mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	   }


	   /* 7. field `F' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 8. field `rowscal' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
	   
	   if (param->rcomflag==0) {
	      if (PRE->issingle) {
		 pr = mxGetPr(fout);
		 for (i=0;i<n; i++)
		     pr[i]=(double)scurrent->rowscal[i];
	      }
	      else {
		 pdata = mxGetData(fout);
		 memcpy(pdata, current->rowscal, (size_t)n*sizeof(double));
	      }
	   }
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 9. save level size to field `colscal' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
	
	   if (param->rcomflag==0) {
	      if (PRE->issingle) {
		 pr = mxGetPr(fout);
		 for (i=0;i<n; i++)
		     pr[i]=(double)scurrent->colscal[i];
	      }
	      else {
		 pdata = mxGetData(fout);
		 memcpy(pdata, current->colscal, (size_t)n*sizeof(double));
	      }
	   }
	
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 10. field `p' */
	   ++ifield;
	   if (param->rcomflag==0) {
	      fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
	      pdata = mxGetData(fout);
	   
	      if (PRE->issingle) {
		 for (i=0;i<n; i++)
		     convert[i]=scurrent->p[i];
	      }
	      else {
		 for (i=0;i<n; i++)
		     convert[i]=current->p[i];
	      }
	      memcpy(pdata, convert, (size_t)n*sizeof(double));
	
	      /* set each field in preconditioner output structure */
	      mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	   }
	   else {
	      fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	      /* set each field in preconditioner output structure */
	      mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	   }

	   
	   /* 11. field `invq' */
	   ++ifield;
	   if (param->rcomflag==0) {
	      fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
	      pdata = mxGetData(fout);
	   
	      if (PRE->issingle) {
		 for (i=0;i<n; i++)
		     convert[i]=scurrent->invq[i];
	      }
	      else {
		 for (i=0;i<n; i++)
		     convert[i]=current->invq[i];
	      }
	      memcpy(pdata, convert, (size_t)n*sizeof(double));
	   
	      /* set each field in preconditioner output structure */
	      mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	   }
	   else {
	      fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	      /* set each field in preconditioner output structure */
	      mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);
	   }


	   /* 12. field `param' */
	   ++ifield;
	   fout=mxCreateNumericArray((mwSize)1,mydims, mxUINT64_CLASS, mxREAL);
	   pdata = mxGetData(fout);
	
	   memcpy(pdata, &param, (size_t)sizeof(size_t));
	
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 13. field `ptr' */
	   ++ifield;
	   fout=mxCreateNumericArray((mwSize)1,mydims, mxUINT64_CLASS, mxREAL);
	   pdata = mxGetData(fout);
	
	   memcpy(pdata, &PRE, (size_t)sizeof(size_t));
	
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 14. save real property to field `isreal' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);

	   *pr=1;
	   
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 15. save positive definite property to field `isdefinite' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);

	   *pr=1;
	   
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 16. save symmetry property to field `issymmetric' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);

	   *pr=1;
       
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);


	   
	   /* 17. save trivial Hermitian property to field `ishermitian' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);

	   *pr=1;
       
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 18. save trivial Hermitian property to field `issingle' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);

	   if (PRE->issingle)
	      *pr=scurrent->issingle;
	   else
	      *pr=current->issingle;
       
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, jstruct, ifield, fout);



	   /* 19. save coarse grid system `A_H' */
	   ++ifield;
	   if (jstruct>=PRE->nlev-1) {
	      fout=mxCreateSparse((mwSize)0,(mwSize)0,(mwSize)0, mxREAL);
	      /*
	        mexPrintf("level %d, no A_H available\n",jstruct+1);fflush(stdout);
	      */
	   }
	   else if (param->ipar[16]&DISCARD_MATRIX) {
	      if (PRE->issingle)
		 fout=mxCreateSparse((mwSize)0,(mwSize)0,(mwSize)0, mxREAL);
	      else
		 fout=mxCreateSparse((mwSize)0, (mwSize)0, (mwSize)0, mxREAL);
	      /*
		mexPrintf("level %d, coarse grid system A_H discarded\n",jstruct+1);fflush(stdout);
	      */
	   }
	   else {
	      /* switched to full-matrix processing? */
	      if (jstruct==PRE->nlev-2 &&
		  ((PRE->issingle)?scurrent->next->LU.ja:current->next->LU.ja)==NULL) {
		 fout=mxCreateSparse((mwSize)0,(mwSize)0,(mwSize)0, mxREAL);

		 /*
		   mexPrintf("level %d, switch to full-matrix processing on the next level\n",jstruct+1);fflush(stdout);
		 */
	      } /* end if-else-if jstruct==PRE->nlev-2 */
	      else {
		 if (PRE->issingle)
		    nnz=scurrent->next->A.ia[scurrent->next->A.nr]-1;
		 else
		    nnz=current->next->A.ia[current->next->A.nr]-1;

		 /*
		   fout=mxCreateSparse((mwSize)0,(mwSize)0,(mwSize)0, mxREAL);
		 */
	      
		 if (PRE->issingle)
		    fout=mxCreateSparse((mwSize)scurrent->next->A.nr,(mwSize)scurrent->next->A.nc,nnz, mxREAL);
		 else
		    fout=mxCreateSparse((mwSize)current->next->A.nr, (mwSize)current->next->A.nc, nnz, mxREAL);
  
		 /*
		   mexPrintf("level %d, %d,%d,%d\n",jstruct+1,current->next->A.nr,current->next->A.nc,nnz);fflush(stdout);
		 */
		 sr =(double *) mxGetPr(fout);
		 irs=(mwIndex *)mxGetIr(fout);
		 jcs=(mwIndex *)mxGetJc(fout);
	      
		 k=0;
		 if (PRE->issingle) {
		    for (i=0; i<scurrent->next->A.nr; i++) {
		        jcs[i]=k;

			j =scurrent->next->A.ia[i]-1;
			jj=scurrent->next->A.ia[i+1]-scurrent->next->A.ia[i];
			Sqsort(scurrent->next->A.a+j,scurrent->next->A.ja+j,istack,&jj);

			for (j=scurrent->next->A.ia[i]-1; j<scurrent->next->A.ia[i+1]-1; j++) {
			    irs[k] =scurrent->next->A.ja[j]-1;
			    sr[k++]=scurrent->next->A.a[j];
			}
		    }
		 }
		 else {
		    for (i=0; i<current->next->A.nr; i++) {
		        jcs[i]=k;

			j =current->next->A.ia[i]-1;
			jj=current->next->A.ia[i+1]-current->next->A.ia[i];
			Dqsort(current->next->A.a+j,current->next->A.ja+j,istack,&jj);

			for (j=current->next->A.ia[i]-1; j<current->next->A.ia[i+1]-1; j++) {
			    irs[k] =current->next->A.ja[j]-1;
			    sr[k++]=current->next->A.a[j];
			}
		    }
		 }
		 jcs[i]=k;
	      }
	   }
	   /* set each field in preconditioner output structure */
	   /*
	     pr=(double *) mxGetPr(fout);
	     j=mxGetM(fout); k=mxGetN(fout);
	     mexPrintf("!!!level %d, %d,%d\n",jstruct+1,j,k);fflush(stdout);
	   */
	   mxSetFieldByNumber(PRE_output, (mwIndex)jstruct, ifield, fout);



	   /* 20. save error in L to field `errorL' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);
	   if (PRE->issingle)
	      *pr=scurrent->errorL;     
	   else
	      *pr=current->errorL;     
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, (mwIndex)jstruct, ifield, fout);
	   
	   /* 21. save error in U to field `errorU' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);
	   if (PRE->issingle)
	      *pr=scurrent->errorU;
	   else
	      *pr=current->errorU;
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, (mwIndex)jstruct, ifield, fout);

	   /* 22. save error in S to field `errorS' */
	   ++ifield;
	   fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	   pr = mxGetPr(fout);
	   if (PRE->issingle)
	      *pr=scurrent->errorS;
	   else
	      *pr=current->errorS;
	   /* set each field in preconditioner output structure */
	   mxSetFieldByNumber(PRE_output, (mwIndex)jstruct, ifield, fout);



	   if (PRE->issingle) {
	      n-=scurrent->nB;
	      scurrent=scurrent->next;
	   }
	   else {
	      n-=current->nB;
	      current=current->next;
	   }
       }
    }
    else {
       /* multi-threaded case */

#ifdef PRINT_INFO_OMP
       mexPrintf("start multi-threaded case\n");fflush(stdout);
#endif

       plhs[0] = mxCreateStructMatrix((mwSize)1,(mwSize)1, SIZE_PNAMES_OMPROOT, pnames_omproot);
       if (plhs[0]==NULL)
	   mexErrMsgTxt("Could not create structure mxArray\n");
       PRE_output=plhs[0];


       /* store total size of the system */
       n=A.nr;

       /* auxiliary buffers */
       ibuff=  (integer *)MAlloc((size_t)n*sizeof(integer),"DSPDilupackfactor:ibuff");
       istack =(integer *)MAlloc((size_t)n*sizeof(integer),"DSPDilupackfactor:istack");
       convert=(double *) MAlloc((size_t)n*sizeof(double), "DSPDilupackfactor:convert");

       /* mixed precision case */
       if (PRE->issingle) {
	  SPRE=(SAMGlevelmat *)PRE;
       }


       /* -------------------------------------- */
       /* ----- extract global information ----- */

       /*  1. field `n' */
       ifield=0;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
       pr = mxGetPr(fout);
	
       if (PRE->issingle) 
	  *pr=SPRE->n;
       else
	  *pr=PRE->n;
       
       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, fout);


       /*  2. field `p' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       if (param->rcomflag==0) {
	  fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
	  pdata = mxGetData(fout);
	  
	  if (PRE->issingle) {
	     for (i=0;i<n; i++)
	         convert[i]=SPRE->p[i];
	  }
	  else {
	     for (i=0;i<n; i++)
	         convert[i]=PRE->p[i];
	  }
	  memcpy(pdata, convert, (size_t)n*sizeof(double));
	
	  /* set each field in preconditioner output structure */
	  mxSetFieldByNumber(PRE_output, 0, ifield, fout);
       }
       else {
	  fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	  /* set each field in preconditioner output structure */
	  mxSetFieldByNumber(PRE_output, 0, ifield, fout);
       }

	   
       /*  3. field `invq' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       if (param->rcomflag==0) {
	  fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
	  pdata = mxGetData(fout);
	   
	  if (PRE->issingle) {
	     for (i=0;i<n; i++)
	         convert[i]=SPRE->invq[i];
	  }
	  else {
	     for (i=0;i<n; i++)
	         convert[i]=PRE->invq[i];
	      }
	  memcpy(pdata, convert, (size_t)n*sizeof(double));
	   
	  /* set each field in preconditioner output structure */
	  mxSetFieldByNumber(PRE_output, 0, ifield, fout);
       }
       else {
	  fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	  /* set each field in preconditioner output structure */
	  mxSetFieldByNumber(PRE_output, 0, ifield, fout);
       }


       /*  4. field `param' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       fout=mxCreateNumericArray((mwSize)1,mydims, mxUINT64_CLASS, mxREAL);
       pdata = mxGetData(fout);
	
       memcpy(pdata, &param, (size_t)sizeof(size_t));
	
       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, fout);



       /*  5. field `ptr' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       fout=mxCreateNumericArray((mwSize)1,mydims, mxUINT64_CLASS, mxREAL);
       pdata = mxGetData(fout);
	
       memcpy(pdata, &PRE, (size_t)sizeof(size_t));
	
       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, fout);



       /*  6. save real property to field `isreal' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
       pr = mxGetPr(fout);

       *pr=1;
	   
       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, fout);



       /*  7. save positive definite property to field `isdefinite' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
       pr = mxGetPr(fout);

       *pr=1;
	   
       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, fout);



       /*  8. save symmetry property to field `issymmetric' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
       pr = mxGetPr(fout);

       *pr=1;
       
       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, fout);


	   
       /*  9. save trivial Hermitian property to field `ishermitian' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
       pr = mxGetPr(fout);

       *pr=1;
       
       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, fout);



       /* 10. save trivial Hermitian property to field `issingle' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
       pr = mxGetPr(fout);

       if (PRE->issingle)
	  *pr=SPRE->issingle;
       else
	  *pr=PRE->issingle;
       
       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, fout);


       /* 11. field `ompparts' */
       ++ifield;
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s",ifield+1,pnames_omproot[ifield]);
#endif
       if (PRE->issingle)
	  nompparts=(mwSize)SPRE->nompparts;
       else
	  nompparts=(mwSize)PRE->nompparts;


#ifdef PRINT_INFO_OMP
       mexPrintf(", %d tasks\n",nompparts);fflush(stdout);
#endif
       /* --- temporarily end extract global information --- */
       /* -------------------------------------------------- */





       /* ------------------------------------------------- */
       /* ----- extract parallel multilevel ILU tasks ----- */

       /* create as many cell array entries as partial multilevel ILUs are present */
       ompparts=mxCreateCellArray((mwSize)1,&nompparts);

       for (iomp=0; iomp<nompparts; iomp++) {
#ifdef PRINT_INFO_OMP
	   mexPrintf("%2d. field %s{%d}",ifield+1,pnames_omproot[ifield],iomp+1);
#endif
	   if (PRE->issingle) {
	      /* local root of the i-th multilevel ILU */
	      spre    =&(SPRE->ompparts[iomp]);
	      scurrent=spre;
	      /* number of levels */
	      nlev    =spre->nlev;
	      /* size of the current partial system */
	      n       =spre->n;
	      ispartial=spre->ispartial;
	   }
	   else {
	      /* local root of the iomp-th multilevel ILU */
	      pre     =&(PRE->ompparts[iomp]);
	      current =pre;
	      /* number of levels */
	      nlev    =pre->nlev;
	      /* size of the current partial system */
	      n       =pre->n;
	      ispartial=pre->ispartial;
	   }
	   ppre = mxCreateStructMatrix((mwSize)1,(mwSize)nlev, SIZE_PPNAMES, ppnames);


#ifdef PRINT_INFO_OMP
	   mexPrintf(", %d levels\n",nlev);fflush(stdout);
#endif
	   for (jstruct=0; jstruct<nlev; jstruct++) {
      
	       /* 1. field `n' */
	       jfield=0;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);
	
	       if (PRE->issingle) 
		  *pr=scurrent->n;
	       else
		  *pr=current->n;
       
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, jstruct, jfield, fout);



	       /* 2. field `nB' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);

	       if (PRE->issingle) 
		  *pr=scurrent->nB;
	       else
		  *pr=current->nB;
       
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, jstruct, jfield, fout);



	       /* 3. field `L' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       if (param->rcomflag==0) {
		  /* switched to full-matrix processing */
		  if (jstruct==nlev-1 && ((PRE->issingle)?scurrent->LU.ja:current->LU.ja)==NULL && !ispartial) {
		   
		     if (PRE->issingle) {
		        fout=mxCreateDoubleMatrix((mwSize)scurrent->nB,(mwSize)scurrent->nB, mxREAL);
			/* position of the diagonal entry inside the factorization */
			for (i=0; i<scurrent->nB; i++) 
			    ibuff[i]=i*scurrent->nB-(i*(i-1))/2;
			sr=mxGetPr(fout);
			pr=(double *)scurrent->LU.a;
		     }
		     else {
		        fout=mxCreateDoubleMatrix((mwSize)current->nB, (mwSize)current->nB,  mxREAL);
			/* position of the diagonal entry inside the factorization */
		        for (i=0; i<current->nB; i++) 
		            ibuff[i]=i*current->nB-(i*(i-1))/2;
			sr=mxGetPr(fout);
			pr=current->LU.a;
		     }

		     for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
		         /* init strict upper triangular part with zeros */
		         for (j=0; j<i; j++) {
			     *sr++=0;
			 }

			 /* additional permutation? */
#ifndef USE_LAPACK_DRIVER
			 l=((PRE->issingle)?scurrent->LU.ia[i]:current->LU.ia[i])-1;
#else
			 l=i;
#endif

			 /* extract diagonal entry */
#ifndef USE_LAPACK_DRIVER
			 dbuf=1.0/pr[ibuff[l]];
#else
			 dbuf=pr[ibuff[l]];
#endif
			 *sr++=dbuf*dbuf;


			 /* extract strict lower triangular part */
			 for (j=i+1; j<((PRE->issingle)?scurrent->nB:current->nB); j++) {

			     /* additional permutation? */
#ifndef USE_LAPACK_DRIVER
			     k=((PRE->issingle)?scurrent->LU.ia[j]:current->LU.ia[j])-1;
#else
			     k=j;
#endif

			     /* position of the associated off-diagonal entry a_{ij} */
			     m=k-l;
			     if (m>0)
			        m=ibuff[l]+m;
			     else
			        m=ibuff[k]-m;
			     *sr++=pr[m]*dbuf;

			 }
		     }	  
 
		     /*
		     for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
#ifndef USE_LAPACK_DRIVER
		         l=((PRE->issingle)?scurrent->LU.ia[i]:current->LU.ia[i])-1;
#else
			 l=i;
#endif
			 for (j=0; j<i; j++) {
#ifndef USE_LAPACK_DRIVER
			     k=((PRE->issingle)?scurrent->LU.ia[j]:current->LU.ia[j])-1;
#else
			     k=j;
#endif
			     m=k-l;
			     if (m>0)
			        m=ibuff[l]+m;
			     else
			        m=ibuff[k]-m;
			     *sr++=pr[m];

			 }
#ifndef USE_LAPACK_DRIVER
			 *sr++=1.0/pr[ibuff[l]];
#else
			 *sr++=pr[ibuff[l]];
#endif
			 for (j=i+1; j<((PRE->issingle)?scurrent->nB:current->nB); j++) {
			     *sr++=0;
			 }
		     }	   
		     */

		     /* set each field in (partial) preconditioner output structure */
		     mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
		  }
		  else if (jstruct==nlev-1 && ispartial) {
		     /* no L on the final level if the factorization is partial */
		     fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
		     mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
		  }
		  else {
		     if (PRE->issingle)
		        nnz=scurrent->LU.nnz+1-scurrent->LU.ja[0]+scurrent->nB;
		     else
		        nnz=current->LU.nnz+1-current->LU.ja[0]+current->nB;

		     if (param->flags&COARSE_REDUCE) {
		        if (PRE->issingle)
			   fout=mxCreateSparse((mwSize)scurrent->nB,(mwSize)scurrent->nB, nnz, mxREAL);
			else
			   fout=mxCreateSparse((mwSize)current->nB, (mwSize)current->nB,  nnz, mxREAL);
		     }
		     else {
		        if (PRE->issingle) {
			   fout=mxCreateSparse((mwSize)scurrent->n,(mwSize)scurrent->nB, nnz, mxREAL);
			   buffldl=scurrent->LU.ja[scurrent->nB];
			   scurrent->LU.ja[scurrent->nB]=scurrent->LU.nnz+1;
			}
			else {
			   fout=mxCreateSparse((mwSize)current->n, (mwSize)current->nB,  nnz, mxREAL);
			   buffldl=current->LU.ja[current->nB];
			   current->LU.ja[current->nB]=current->LU.nnz+1;
			}
		     }
	      
		     sr =(double *) mxGetPr(fout);
		     irs=(mwIndex *)mxGetIr(fout);
		     jcs=(mwIndex *)mxGetJc(fout);
		     
		     k=0;
		     if (PRE->issingle) {
		        for (i=0; i<scurrent->nB; i++) {
			    /* extract diagonal entry */
			    jcs[i]=k;
			    irs[k]=i;
			    sr[k++]=1.0/scurrent->LU.a[i];

			    j =scurrent->LU.ja[i]-1;
			    jj=scurrent->LU.ja[i+1]-scurrent->LU.ja[i];
			    Sqsort(scurrent->LU.a+j,scurrent->LU.ja+j,istack,&jj);

			    for (j=scurrent->LU.ja[i]-1; j<scurrent->LU.ja[i+1]-1; j++) {
			        irs[k] =scurrent->LU.ja[j]-1;
				sr[k++]=scurrent->LU.a[j];
			    }
			}
		     }
		     else {
		        for (i=0; i<current->nB; i++) {
			    /* extract diagonal entry */
			    jcs[i]=k;
			    irs[k]=i;
			    sr[k++]=1.0/current->LU.a[i];

			    j =current->LU.ja[i]-1;
			    jj=current->LU.ja[i+1]-current->LU.ja[i];
			    Dqsort(current->LU.a+j,current->LU.ja+j,istack,&jj);
			    
			    for (j=current->LU.ja[i]-1; j<current->LU.ja[i+1]-1; j++) {
			        irs[k] =current->LU.ja[j]-1;
				sr[k++]=current->LU.a[j];
			    }
			}
		     }
		     jcs[i]=k;
	      
		     if (!(param->flags&COARSE_REDUCE)) {
		        if (PRE->issingle)
			   scurrent->LU.ja[scurrent->nB]=buffldl;
			else
			   current->LU.ja[current->nB]=buffldl;
		     }

		     /* set each field in (partial) preconditioner output structure */
		     mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
		  }
	       }
	       else {
		  fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
		  /* set each field in (partial) preconditioner output structure */
		  mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	       } /* if-else rcomflag==0 */


	       /* 4. field `D' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       if (param->rcomflag==0) {
		  if (PRE->issingle)
		     fout=mxCreateSparse((mwSize)scurrent->nB,(mwSize)scurrent->nB, (mwSize)scurrent->nB, mxREAL);
		  else
		     fout=mxCreateSparse((mwSize)current->nB, (mwSize)current->nB,  (mwSize)current->nB, mxREAL);

		  sr =(double *) mxGetPr(fout);
		  irs=(mwIndex *)mxGetIr(fout);
		  jcs=(mwIndex *)mxGetJc(fout);
	   
		  for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
		      jcs[i]=i;
		      irs[i]=i;
		  }
		  jcs[i]=i;
	   
		  /* switched to full-matrix processing */
		  if (jstruct==nlev-1 &&
		      ((PRE->issingle)?scurrent->LU.ja:current->LU.ja)==NULL && !ispartial) {

		     if (PRE->issingle)
		        spr=scurrent->LU.a;
		     else
		        pr=current->LU.a;
		     for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
#ifndef USE_LAPACK_DRIVER
		         l=((PRE->issingle)?scurrent->LU.ia[i]:current->LU.ia[i])-1;
			 dbuf=1.0/((PRE->issingle)?(double)spr[ibuff[l]]:pr[ibuff[l]]);
#else
			 l=i;
			 dbuf=((PRE->issingle)?(double)spr[ibuff[l]]:pr[ibuff[l]]);
#endif
			 sr[i]=1.0/(dbuf*dbuf);
		     }	   
		  }
		  else if (jstruct==nlev-1 && ispartial) {
		     /* no D on the final level if the factorization is partial */
		     fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
		     /* mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout); */
		  }
		  else {
		     for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
#ifndef USE_LAPACK_DRIVER
		         sr[i]=1.0/((PRE->issingle)?scurrent->LU.a[i]:current->LU.a[i]);
#else
			 sr[i]=((PRE->issingle)?scurrent->LU.a[i]:current->LU.a[i]);
#endif
		     }
		  }
		  /* set each field in (partial) preconditioner output structure */
		  mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	       }
	       else {
		  fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
		  /* set each field in (partial) preconditioner output structure */
		  mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	       }

	       
	       
	       /* 5. field `U' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	   
	   
	   
	       /* 6. field `E', E=F^T */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       if (param->rcomflag==0) {
		  if (jstruct<nlev-1){ 
		
		     if (param->flags&COARSE_REDUCE) {
		        if (PRE->issingle) {
			   nnz=scurrent->F.ia[scurrent->nB]-1;
			   fout=mxCreateSparse((mwSize)n-scurrent->nB,(mwSize)scurrent->nB, nnz, mxREAL);
			}
			else { /* !PRE->issingle */
			   nnz=current->F.ia[current->nB]-1;
			   fout=mxCreateSparse((mwSize)n-current->nB, (mwSize)current->nB,  nnz, mxREAL);
			}
			sr =(double *) mxGetPr(fout);
			irs=(mwIndex *)mxGetIr(fout);
			jcs=(mwIndex *)mxGetJc(fout);
		 
			k=0;
			for (i=0; i<((PRE->issingle)?scurrent->nB:current->nB); i++) {
			    jcs[i]=k;
			    if (PRE->issingle) {

			       j =scurrent->F.ia[i]-1;
			       jj=scurrent->F.ia[i+1]-scurrent->F.ia[i];
			       Sqsort(scurrent->F.a+j,scurrent->F.ja+j,istack,&jj);

			       for (j=scurrent->F.ia[i]-1; j<scurrent->F.ia[i+1]-1; j++) {
				   irs[k] =scurrent->F.ja[j]-1;
				   sr[k++]=scurrent->F.a[j];
			       }
			    }
			    else {

			       j =current->F.ia[i]-1;
			       jj=current->F.ia[i+1]-current->F.ia[i];
			       Dqsort(current->F.a+j,current->F.ja+j,istack,&jj);

			       for (j=current->F.ia[i]-1; j<current->F.ia[i+1]-1; j++) {
				   irs[k] =current->F.ja[j]-1;
				   sr[k++]=current->F.a[j];
			       }
			    }
			}
			jcs[i]=k;
		     }
		     else {
		        nnz=0;
			if (PRE->issingle)
			   fout=mxCreateSparse((mwSize)n-scurrent->nB,(mwSize)scurrent->nB, nnz, mxREAL);
			else
			   fout=mxCreateSparse((mwSize)n-current->nB, (mwSize)current->nB,  nnz, mxREAL);
		     }
		     /* set each field in (partial) preconditioner output structure */
		     mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
		  }
	       }
	       else {
		  fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
		  /* set each field in (partial) preconditioner output structure */
		  mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	       }


	       /* 7. field `F' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);



	       /* 8. field `rowscal' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
	   
	       if (param->rcomflag==0) {
		  if (PRE->issingle) {
		     pr = mxGetPr(fout);
		     if (scurrent->rowscal==NULL) {
		        for (i=0;i<n; i++)
			    pr[i]=1.0;
		     }
		     else {
		        for (i=0;i<n; i++)
			    pr[i]=(double)scurrent->rowscal[i];
		     }
		  }
		  else {
		     if (current->rowscal==NULL) {
		        pr = mxGetPr(fout);
		        for (i=0;i<n; i++)
			    pr[i]=1.0;
		     }
		     else {
		        pdata = mxGetData(fout);
		        memcpy(pdata, current->rowscal, (size_t)n*sizeof(double));
		     }
		  }
	       }
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);



	       /* 9. save level size to field `colscal' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
	
	       if (param->rcomflag==0) {
		  if (PRE->issingle) {
		     pr = mxGetPr(fout);
		     if (scurrent->colscal==NULL)
		        for (i=0;i<n; i++)
			    pr[i]=1.0;
		     else
		        for (i=0;i<n; i++)
			    pr[i]=(double)scurrent->colscal[i];
		  }
		  else {
		    if (current->colscal==NULL) {
		       pr = mxGetPr(fout);
		       for (i=0;i<n; i++)
			   pr[i]=1.0;
		    }
		    else {
		       pdata = mxGetData(fout);
		       memcpy(pdata, current->colscal, (size_t)n*sizeof(double));
		    }
		  }
	       }
	
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);



	       /* 10. field `p' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       if (param->rcomflag==0) {
		  fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
		  pdata = mxGetData(fout);
	   
		  if (PRE->issingle) {
		     if (scurrent->p==NULL)
		        for (i=0;i<n; i++)
			    convert[i]=i+1;
		     else
		        for (i=0;i<n; i++)
			    convert[i]=scurrent->p[i];
		  }
		  else {
		     if (current->p==NULL)
		        for (i=0;i<n; i++)
			    convert[i]=i+1;
		     else
		        for (i=0;i<n; i++)
			    convert[i]=current->p[i];
		  }
		  memcpy(pdata, convert, (size_t)n*sizeof(double));
	
		  /* set each field in (partial) preconditioner output structure */
		  mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	       }
	       else {
		  fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
		  /* set each field in (partial) preconditioner output structure */
		  mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	       }

	   
	       /* 11. field `invq' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       if (param->rcomflag==0) {
		  fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)n, mxREAL);
		  pdata = mxGetData(fout);
	   
		  if (PRE->issingle) {
		     if (scurrent->invq==NULL)
		        for (i=0;i<n; i++)
			    convert[i]=i+1;
		     else
		        for (i=0;i<n; i++)
			    convert[i]=scurrent->invq[i];
		  }
		  else {
		     if (current->invq==NULL)
		        for (i=0;i<n; i++)
			    convert[i]=i+1;
		     else
		        for (i=0;i<n; i++)
			    convert[i]=current->invq[i];
		  }
		  memcpy(pdata, convert, (size_t)n*sizeof(double));
	   
		  /* set each field in (partial) preconditioner output structure */
		  mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	       }
	       else {
		  fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
		  /* set each field in (partial) preconditioner output structure */
		  mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	       }


	       /* 12. save coarse grid system `A_H' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       if (jstruct>=nlev-1) {
#ifdef PRINT_INFO_OMP1
		  mexPrintf("%2d. field %s{%d}(%d).%s...if\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		  fout=mxCreateSparse((mwSize)0,(mwSize)0,(mwSize)0, mxREAL);
	       }
	       else if (param->ipar[16]&DISCARD_MATRIX) {
#ifdef PRINT_INFO_OMP1
		  mexPrintf("%2d. field %s{%d}(%d).%s...else if\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		  if (PRE->issingle)
		     fout=mxCreateSparse((mwSize)n-scurrent->nB,(mwSize)n-scurrent->nB,(mwSize)0, mxREAL);
		  else
		     fout=mxCreateSparse((mwSize)n-current->nB, (mwSize)n-current->nB, (mwSize)0, mxREAL);
	       }
	       else {
#ifdef PRINT_INFO_OMP1
		  mexPrintf("%2d. field %s{%d}(%d).%s...else\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		  /* switched to full-matrix processing */
		  if (jstruct==nlev-2 &&
		      ((PRE->issingle)?scurrent->next->LU.ja:current->next->LU.ja)==NULL) {
#ifdef PRINT_INFO_OMP1
		     mexPrintf("%2d. field %s{%d}(%d).%s...else...if\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		     fout=mxCreateSparse((mwSize)0,(mwSize)0,(mwSize)0, mxREAL);
		  }
		  else {
#ifdef PRINT_INFO_OMP1
		     mexPrintf("%2d. field %s{%d}(%d).%s...else...else\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		     nnz=0;
		     if (PRE->issingle) {
		        if (scurrent->next->A.ia!=NULL)
			   nnz=scurrent->next->A.ia[scurrent->next->A.nr]-1;
		     }
		     else {
		        if (current->next->A.ia!=NULL)
			   nnz=current->next->A.ia[current->next->A.nr]-1;
		     }

#ifdef PRINT_INFO_OMP1
		     mexPrintf("%2d. field %s{%d}(%d).%s...else...else2\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		     /*
		       mexPrintf("level %d, %d,%d,%d\n",jstruct+1,current->next->A.nr,current->next->A.nc,nnz);fflush(stdout);
		       fout=mxCreateSparse((mwSize)0,(mwSize)0,(mwSize)0, mxREAL);
		     */
	      
		     if (PRE->issingle)
		        fout=mxCreateSparse((mwSize)scurrent->next->A.nr,(mwSize)scurrent->next->A.nc,nnz, mxREAL);
		     else
		        fout=mxCreateSparse((mwSize)current->next->A.nr, (mwSize)current->next->A.nc, nnz, mxREAL);
  
#ifdef PRINT_INFO_OMP1
		     mexPrintf("%2d. field %s{%d}(%d).%s...else...else3\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		     sr =(double *) mxGetPr(fout);
		     irs=(mwIndex *)mxGetIr(fout);
		     jcs=(mwIndex *)mxGetJc(fout);
	      
#ifdef PRINT_INFO_OMP1
		     mexPrintf("%2d. field %s{%d}(%d).%s...else...else4\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		     k=0;
		     if (PRE->issingle) {
#ifdef PRINT_INFO_OMP1
		        mexPrintf("%2d. field %s{%d}(%d).%s...else...else...issingle\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		        for (i=0; i<scurrent->next->A.nr; i++) {
			    jcs[i]=k;

			    j =scurrent->next->A.ia[i]-1;
			    jj=scurrent->next->A.ia[i+1]-scurrent->next->A.ia[i];
			    Sqsort(scurrent->next->A.a+j,scurrent->next->A.ja+j,istack,&jj);

			    for (j=scurrent->next->A.ia[i]-1; j<scurrent->next->A.ia[i+1]-1; j++) {
			        irs[k] =scurrent->next->A.ja[j]-1;
				sr[k++]=scurrent->next->A.a[j];
			    }
			}
		     }
		     else {
#ifdef PRINT_INFO_OMP1
		        mexPrintf("%2d. field %s{%d}(%d).%s...else...else...isdouble\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
		        for (i=0; i<current->next->A.nr; i++) {
			    jcs[i]=k;

			    j =current->next->A.ia[i]-1;
			    jj=current->next->A.ia[i+1]-current->next->A.ia[i];
			    Dqsort(current->next->A.a+j,current->next->A.ja+j,istack,&jj);
			    
			    for (j=current->next->A.ia[i]-1; j<current->next->A.ia[i+1]-1; j++) {
			        irs[k] =current->next->A.ja[j]-1;
				sr[k++]=current->next->A.a[j];
			    }
			}
		     }
		     jcs[i]=k;
		  }
	       }

#ifdef PRINT_INFO_OMP1
	       mexPrintf("%2d. field %s{%d}(%d).%s...set field\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	       
	       
	       
	       /* 13. save error in L to field `errorL' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);
	       if (PRE->issingle)
		  *pr=scurrent->errorL;     
	       else
		  *pr=current->errorL;     
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);
	   
	       /* 14. save error in U to field `errorU' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);
	       if (PRE->issingle)
		  *pr=scurrent->errorU;
	       else
		  *pr=current->errorU;
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);

	       /* 15. save error in S to field `errorS' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);
	       if (PRE->issingle)
		  *pr=scurrent->errorS;
	       else
		  *pr=current->errorS;
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);


	       /* 16. save partial property to field `ispartial' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);

	       if (PRE->issingle)
		  *pr=scurrent->ispartial;
	       else
		  *pr=current->ispartial;
       
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);


	       /* 17. save partial Schur complement to field `A' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       j=0;
	       if (PRE->issingle) {
		  if (scurrent->ispartial && jstruct==nlev-1)
		     j=-1;
	       }
	       else {
		  if (current->ispartial && jstruct==nlev-1)
		     j=-1;
	       }

	       if (j) {
		  if (PRE->issingle)
		     nnz=scurrent->A.ia[scurrent->A.nr]-1;
		  else
		     nnz= current->A.ia[ current->A.nr]-1;

		  if (PRE->issingle)
		     fout=mxCreateSparse((mwSize)scurrent->A.nr,(mwSize)scurrent->A.nc,nnz, mxREAL);
		  else
		     fout=mxCreateSparse((mwSize) current->A.nr,(mwSize) current->A.nc,nnz, mxREAL);
  
		  sr =(double *) mxGetPr(fout);
		  irs=(mwIndex *)mxGetIr(fout);
		  jcs=(mwIndex *)mxGetJc(fout);
	      
		  k=0;
		  if (PRE->issingle) {
		     for (i=0; i<scurrent->A.nr; i++) {
		         jcs[i]=k;

			 j =scurrent->A.ia[i]-1;
			 jj=scurrent->A.ia[i+1]-scurrent->A.ia[i];
			 Sqsort(scurrent->A.a+j,scurrent->A.ja+j,istack,&jj);

			 for (j=scurrent->A.ia[i]-1; j<scurrent->A.ia[i+1]-1; j++) {
			     irs[k] =scurrent->A.ja[j]-1;
			     sr[k++]=scurrent->A.a[j];
			 } /* end for j */
		     } /* end for i */
		  } /* end if */
		  else {
		     for (i=0; i<current->A.nr; i++) {
		         jcs[i]=k;

			 j =current->A.ia[i]-1;
			 jj=current->A.ia[i+1]-current->A.ia[i];
			 Dqsort(current->A.a+j,current->A.ja+j,istack,&jj);
			    
			 for (j=current->A.ia[i]-1; j<current->A.ia[i+1]-1; j++) {
			     irs[k] =current->A.ja[j]-1;
			     sr[k++]=current->A.a[j];
			 } /* end for j */
		     } /* end for i */
		  } /* end if-else PRE->issingle */
		  jcs[i]=k;
	       }
	       else {
		  /* no `A' on the final level if the factorization is partial */
		  fout=mxCreateDoubleMatrix((mwSize)0,(mwSize)0, mxREAL);
	       } /* if-else */

	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);



	       /* 18. save property to field `isreal' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);

	       *pr=1;
	   
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);


	       /* 19. save property to field `isdefinite' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);

	       *pr=1;
	   
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);


	       /* 20. save property to field `issymmetric' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);

	       *pr=1;
	   
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);


	       /* 21. save property to field `ishermitian' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);

	       *pr=1;
	   
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);


	       /* 22. save property to field `issingle' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)1, mxREAL);
	       pr = mxGetPr(fout);

	       if (PRE->issingle)
		  *pr=1;
	       else
		  *pr=0;
	   
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);


	       /* 23. save property to field `p_local' */
	       ++jfield;
#ifdef PRINT_INFO_OMP
	       mexPrintf("%2d. field %s{%d}(%d).%s\n",ifield+1,pnames_omproot[ifield],iomp+1,jstruct+1,ppnames[jfield]);fflush(stdout);
#endif
	       if (PRE->issingle) {
		  if (scurrent->p_local==NULL)
		     i=0;
		  else
		     i=scurrent->n;
	       }
	       else {
		  if (current->p_local==NULL)
		     i=0;
		  else
		     i=current->n;
	       }
	       fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)i, mxREAL);
	       pr = mxGetPr(fout);
	       
	       if (PRE->issingle) {
		  for (j=0; j<i; j++)
		      pr[j]=scurrent->p_local[j];
	       }
	       else {
		  for (j=0; j<i; j++)
		      pr[j]=current->p_local[j];
	       }
	   
	       /* set each field in (partial) preconditioner output structure */
	       mxSetFieldByNumber(ppre, (mwIndex)jstruct, jfield, fout);



	       if (PRE->issingle) {
		  n-=scurrent->nB;
		  scurrent=scurrent->next;
	       }
	       else {
		  n-=current->nB;
		  current=current->next;
	       }
	   } /* end for jstruct */

	   
	   /* set cell component parallel preconditioner output structure */
	   mxSetCell(ompparts, (mwIndex)iomp, ppre); 

       } /* end for iomp */
       /* --- END extract parallel multilevel ILU tasks --- */
       /* ------------------------------------------------- */






       /* ---------------------------------------------- */
       /* --- continue to extract global information --- */


       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, ompparts);



       /* 12. field `omptab' */
       ++ifield;      
#ifdef PRINT_INFO_OMP
       mexPrintf("%2d. field %s\n",ifield+1,pnames_omproot[ifield]);fflush(stdout);
#endif
       pomptab=mxCreateStructMatrix((mwSize)1,(mwSize)1, SIZE_PPNAMES_OMPTAB, pnames_omptab);
       if (PRE->issingle) {
	  l=SPRE->omptab.dimL;
       }
       else {
	  l=PRE->omptab.dimL;
       }


          /* 12.1. field `treetab' */
	  jfield=0;
#ifdef PRINT_INFO_OMP
	  mexPrintf("%2d.%d field %s.%s\n",ifield+1,jfield+1,pnames_omproot[ifield],pnames_omptab[jfield]);fflush(stdout);
#endif
	  fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)nompparts, mxREAL);
	  pr = mxGetPr(fout);
	
	  if (PRE->issingle) 
	     for (i=0; i<nompparts; i++)
	         pr[i]=SPRE->omptab.tree[i]+1;
	  else
	     for (i=0; i<nompparts; i++)
	         pr[i]=PRE->omptab.tree[i]+1;

	  /* set each field in omptab output structure */
	  mxSetFieldByNumber(pomptab, 0, jfield, fout);


	  /* 12.2. field `chldtab' */
	  ++jfield;
#ifdef PRINT_INFO_OMP
	  mexPrintf("%2d.%d field %s.%s\n",ifield+1,jfield+1,pnames_omproot[ifield],pnames_omptab[jfield]);fflush(stdout);
#endif
	  fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)nompparts, mxREAL);
	  pr = mxGetPr(fout);
	
	  if (PRE->issingle) 
	     for (i=0; i<nompparts; i++)
	         pr[i]=SPRE->omptab.chld[i]+1;
	  else
	     for (i=0; i<nompparts; i++)
	         pr[i]=PRE->omptab.chld[i]+1;

	  /* set each field in omptab output structure */
	  mxSetFieldByNumber(pomptab, 0, jfield, fout);


	  /* 12.3. field `brthtab' */
	  ++jfield;
#ifdef PRINT_INFO_OMP
	  mexPrintf("%2d.%d field %s.%s\n",ifield+1,jfield+1,pnames_omproot[ifield],pnames_omptab[jfield]);fflush(stdout);
#endif
	  fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)nompparts, mxREAL);
	  pr = mxGetPr(fout);
	
	  if (PRE->issingle) 
	     for (i=0; i<nompparts; i++)
	         pr[i]=SPRE->omptab.brth[i]+1;
	  else
	     for (i=0; i<nompparts; i++)
	         pr[i]=PRE->omptab.brth[i]+1;

	  /* set each field in omptab output structure */
	  mxSetFieldByNumber(pomptab, 0, jfield, fout);


	  /* 12.4. field `hghttab' */
	  ++jfield;
#ifdef PRINT_INFO_OMP
	  mexPrintf("%2d.%d field %s.%s\n",ifield+1,jfield+1,pnames_omproot[ifield],pnames_omptab[jfield]);fflush(stdout);
#endif
	  fout=mxCreateDoubleMatrix((mwSize)1,(mwSize)nompparts, mxREAL);
	  pr = mxGetPr(fout);
	
	  if (PRE->issingle) 
	     for (i=0; i<nompparts; i++)
	         pr[i]=SPRE->omptab.hght[i];
	  else
	     for (i=0; i<nompparts; i++)
	         pr[i]=PRE->omptab.hght[i];


	  /* set each field in pomptab output structure */
	  mxSetFieldByNumber(pomptab, 0, jfield, fout);


       /* set each field in preconditioner output structure */
       mxSetFieldByNumber(PRE_output, 0, ifield, pomptab);



       /* --- END extract global information --- */
       /* -------------------------------------- */


    }


    free(A.ia);
    free(A.ja);
    free(A.a);

    free(ibuff);
    free(istack);
    free(convert);
    
    return;
}

