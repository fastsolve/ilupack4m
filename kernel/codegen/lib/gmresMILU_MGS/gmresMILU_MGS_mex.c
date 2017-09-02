/*
 * gmresMILU_MGS_mex.c
 *
 * Auxiliary code for mexFunction of gmresMILU_MGS
 *
 * C source code generated by m2c.
 * %#m2c options:40a307bea14e989f7d5adbfe2a9c68b7
 *
 */

#include "mex.h"
#if !defined(MATLAB_MEX_FILE) && defined(printf)
#undef printf
#endif
/* Include the C header file generated by codegen in lib mode */
#include "gmresMILU_MGS.h"
#include "m2c.c"

/* Include declaration of some helper functions. */
#include "lib2mex_helper.c"


static void marshallin_const_struct0_T(struct0_T *pStruct, const mxArray *mx, const char *mname) {
    mxArray             *sub_mx;

    if (!mxIsStruct(mx))
        M2C_error("marshallin_const_struct0_T:WrongType",
            "Input argument %s has incorrect data type; struct is expected.", mname);
    if (!mxGetField(mx, 0, "row_ptr"))
        M2C_error("marshallin_const_struct0_T:WrongType",
            "Input argument %s is missing the field row_ptr.", mname);
    if (!mxGetField(mx, 0, "col_ind"))
        M2C_error("marshallin_const_struct0_T:WrongType",
            "Input argument %s is missing the field col_ind.", mname);
    if (!mxGetField(mx, 0, "val"))
        M2C_error("marshallin_const_struct0_T:WrongType",
            "Input argument %s is missing the field val.", mname);
    if (!mxGetField(mx, 0, "nrows"))
        M2C_error("marshallin_const_struct0_T:WrongType",
            "Input argument %s is missing the field nrows.", mname);
    if (!mxGetField(mx, 0, "ncols"))
        M2C_error("marshallin_const_struct0_T:WrongType",
            "Input argument %s is missing the field ncols.", mname);
    if (mxGetNumberOfFields(mx) > 5)
        M2C_warn("marshallin_const_struct0_T:ExtraFields",
            "Extra fields in %s and are ignored.", mname);

    sub_mx = mxGetField(mx, 0, "row_ptr");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongInputType",
            "Input argument row_ptr has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongSizeOfInputArg",
            "Dimension 2 of row_ptr should equal 1.");
    pStruct->row_ptr = mxMalloc(sizeof(emxArray_int32_T));
    init_emxArray((emxArray__common*)(pStruct->row_ptr), 1);
    alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pStruct->row_ptr), "row_ptr", 1);

    sub_mx = mxGetField(mx, 0, "col_ind");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongInputType",
            "Input argument col_ind has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongSizeOfInputArg",
            "Dimension 2 of col_ind should equal 1.");
    pStruct->col_ind = mxMalloc(sizeof(emxArray_int32_T));
    init_emxArray((emxArray__common*)(pStruct->col_ind), 1);
    alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pStruct->col_ind), "col_ind", 1);

    sub_mx = mxGetField(mx, 0, "val");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxDOUBLE_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongInputType",
            "Input argument val has incorrect data type; double is expected.");
    if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongSizeOfInputArg",
            "Dimension 2 of val should equal 1.");
    pStruct->val = mxMalloc(sizeof(emxArray_real_T));
    init_emxArray((emxArray__common*)(pStruct->val), 1);
    alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pStruct->val), "val", 1);

    sub_mx = mxGetField(mx, 0, "nrows");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongInputType",
            "Input argument nrows has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(sub_mx) != 1)
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongSizeOfInputArg",
            "Argument nrows should be a scalar.");
    pStruct->nrows = *(int32_T*)mxGetData(sub_mx);

    sub_mx = mxGetField(mx, 0, "ncols");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongInputType",
            "Input argument ncols has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(sub_mx) != 1)
        mexErrMsgIdAndTxt("marshallin_const_struct0_T:WrongSizeOfInputArg",
            "Argument ncols should be a scalar.");
    pStruct->ncols = *(int32_T*)mxGetData(sub_mx);
}
static void destroy_struct0_T(struct0_T *pStruct) {

    free_emxArray((emxArray__common*)(pStruct->row_ptr));
    mxFree(pStruct->row_ptr);

    free_emxArray((emxArray__common*)(pStruct->col_ind));
    mxFree(pStruct->col_ind);

    free_emxArray((emxArray__common*)(pStruct->val));
    mxFree(pStruct->val);


}


static void marshallin_const_struct2_T(struct2_T *pStruct, const mxArray *mx, const char *mname) {
    mxArray             *sub_mx;

    if (!mxIsStruct(mx))
        M2C_error("marshallin_const_struct2_T:WrongType",
            "Input argument %s has incorrect data type; struct is expected.", mname);
    if (!mxGetField(mx, 0, "col_ptr"))
        M2C_error("marshallin_const_struct2_T:WrongType",
            "Input argument %s is missing the field col_ptr.", mname);
    if (!mxGetField(mx, 0, "row_ind"))
        M2C_error("marshallin_const_struct2_T:WrongType",
            "Input argument %s is missing the field row_ind.", mname);
    if (!mxGetField(mx, 0, "val"))
        M2C_error("marshallin_const_struct2_T:WrongType",
            "Input argument %s is missing the field val.", mname);
    if (!mxGetField(mx, 0, "nrows"))
        M2C_error("marshallin_const_struct2_T:WrongType",
            "Input argument %s is missing the field nrows.", mname);
    if (!mxGetField(mx, 0, "ncols"))
        M2C_error("marshallin_const_struct2_T:WrongType",
            "Input argument %s is missing the field ncols.", mname);
    if (mxGetNumberOfFields(mx) > 5)
        M2C_warn("marshallin_const_struct2_T:ExtraFields",
            "Extra fields in %s and are ignored.", mname);

    sub_mx = mxGetField(mx, 0, "col_ptr");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongInputType",
            "Input argument col_ptr has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongSizeOfInputArg",
            "Dimension 2 of col_ptr should equal 1.");
    pStruct->col_ptr = mxMalloc(sizeof(emxArray_int32_T));
    init_emxArray((emxArray__common*)(pStruct->col_ptr), 1);
    alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pStruct->col_ptr), "col_ptr", 1);

    sub_mx = mxGetField(mx, 0, "row_ind");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongInputType",
            "Input argument row_ind has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongSizeOfInputArg",
            "Dimension 2 of row_ind should equal 1.");
    pStruct->row_ind = mxMalloc(sizeof(emxArray_int32_T));
    init_emxArray((emxArray__common*)(pStruct->row_ind), 1);
    alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pStruct->row_ind), "row_ind", 1);

    sub_mx = mxGetField(mx, 0, "val");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxDOUBLE_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongInputType",
            "Input argument val has incorrect data type; double is expected.");
    if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongSizeOfInputArg",
            "Dimension 2 of val should equal 1.");
    pStruct->val = mxMalloc(sizeof(emxArray_real_T));
    init_emxArray((emxArray__common*)(pStruct->val), 1);
    alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pStruct->val), "val", 1);

    sub_mx = mxGetField(mx, 0, "nrows");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongInputType",
            "Input argument nrows has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(sub_mx) != 1)
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongSizeOfInputArg",
            "Argument nrows should be a scalar.");
    pStruct->nrows = *(int32_T*)mxGetData(sub_mx);

    sub_mx = mxGetField(mx, 0, "ncols");
    if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongInputType",
            "Input argument ncols has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(sub_mx) != 1)
        mexErrMsgIdAndTxt("marshallin_const_struct2_T:WrongSizeOfInputArg",
            "Argument ncols should be a scalar.");
    pStruct->ncols = *(int32_T*)mxGetData(sub_mx);
}
static void destroy_struct2_T(struct2_T *pStruct) {

    free_emxArray((emxArray__common*)(pStruct->col_ptr));
    mxFree(pStruct->col_ptr);

    free_emxArray((emxArray__common*)(pStruct->row_ind));
    mxFree(pStruct->row_ind);

    free_emxArray((emxArray__common*)(pStruct->val));
    mxFree(pStruct->val);


}


static void marshallin_const_emxArray_struct1_T(emxArray_struct1_T *pEmx, const mxArray *mx, const char *mname, const int dim) {
    mxArray             *sub_mx;
    int                  i;
    int                  nElems = mxGetNumberOfElements(mx);;

    if (!mxIsStruct(mx))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s has incorrect data type; struct is expected.", mname);
    if (!mxGetField(mx, 0, "p"))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s is missing the field p.", mname);
    if (!mxGetField(mx, 0, "q"))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s is missing the field q.", mname);
    if (!mxGetField(mx, 0, "rowscal"))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s is missing the field rowscal.", mname);
    if (!mxGetField(mx, 0, "colscal"))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s is missing the field colscal.", mname);
    if (!mxGetField(mx, 0, "L"))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s is missing the field L.", mname);
    if (!mxGetField(mx, 0, "U"))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s is missing the field U.", mname);
    if (!mxGetField(mx, 0, "d"))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s is missing the field d.", mname);
    if (!mxGetField(mx, 0, "negE"))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s is missing the field negE.", mname);
    if (!mxGetField(mx, 0, "negF"))
        M2C_error("marshallin_const_emxArray_struct1_T:WrongType",
            "Input argument %s is missing the field negF.", mname);
    if (mxGetNumberOfFields(mx) > 9)
        M2C_warn("marshallin_const_emxArray_struct1_T:ExtraFields",
            "Extra fields in %s and are ignored.", mname);

    alloc_emxArray_from_mxArray(mx, (emxArray__common*)pEmx, mname, dim, sizeof(struct1_T));
    for (i=0; i<nElems; ++i) {
        sub_mx = mxGetField(mx, i, "p");
        if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongInputType",
                "Input argument p has incorrect data type; int32 is expected.");
        if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongSizeOfInputArg",
                "Dimension 2 of p should equal 1.");
        pEmx->data[i].p = mxMalloc(sizeof(emxArray_int32_T));
        init_emxArray((emxArray__common*)(pEmx->data[i].p), 1);
        alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pEmx->data[i].p), "p", 1);

        sub_mx = mxGetField(mx, i, "q");
        if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxINT32_CLASS)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongInputType",
                "Input argument q has incorrect data type; int32 is expected.");
        if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongSizeOfInputArg",
                "Dimension 2 of q should equal 1.");
        pEmx->data[i].q = mxMalloc(sizeof(emxArray_int32_T));
        init_emxArray((emxArray__common*)(pEmx->data[i].q), 1);
        alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pEmx->data[i].q), "q", 1);

        sub_mx = mxGetField(mx, i, "rowscal");
        if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxDOUBLE_CLASS)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongInputType",
                "Input argument rowscal has incorrect data type; double is expected.");
        if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongSizeOfInputArg",
                "Dimension 2 of rowscal should equal 1.");
        pEmx->data[i].rowscal = mxMalloc(sizeof(emxArray_real_T));
        init_emxArray((emxArray__common*)(pEmx->data[i].rowscal), 1);
        alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pEmx->data[i].rowscal), "rowscal", 1);

        sub_mx = mxGetField(mx, i, "colscal");
        if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxDOUBLE_CLASS)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongInputType",
                "Input argument colscal has incorrect data type; double is expected.");
        if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongSizeOfInputArg",
                "Dimension 2 of colscal should equal 1.");
        pEmx->data[i].colscal = mxMalloc(sizeof(emxArray_real_T));
        init_emxArray((emxArray__common*)(pEmx->data[i].colscal), 1);
        alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pEmx->data[i].colscal), "colscal", 1);

        sub_mx = mxGetField(mx, i, "L");
        if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxSTRUCT_CLASS)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongInputType",
                "Input argument L has incorrect data type; struct is expected.");
        if (mxGetNumberOfElements(sub_mx) != 1)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongSizeOfInputArg",
                "Argument L should be a scalar.");
        marshallin_const_struct2_T(&pEmx->data[i].L, sub_mx, "L");

        sub_mx = mxGetField(mx, i, "U");
        if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxSTRUCT_CLASS)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongInputType",
                "Input argument U has incorrect data type; struct is expected.");
        if (mxGetNumberOfElements(sub_mx) != 1)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongSizeOfInputArg",
                "Argument U should be a scalar.");
        marshallin_const_struct2_T(&pEmx->data[i].U, sub_mx, "U");

        sub_mx = mxGetField(mx, i, "d");
        if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxDOUBLE_CLASS)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongInputType",
                "Input argument d has incorrect data type; double is expected.");
        if (mxGetNumberOfElements(sub_mx) && mxGetDimensions(sub_mx)[1] != 1) 
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongSizeOfInputArg",
                "Dimension 2 of d should equal 1.");
        pEmx->data[i].d = mxMalloc(sizeof(emxArray_real_T));
        init_emxArray((emxArray__common*)(pEmx->data[i].d), 1);
        alias_mxArray_to_emxArray(sub_mx, (emxArray__common *)(pEmx->data[i].d), "d", 1);

        sub_mx = mxGetField(mx, i, "negE");
        if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxSTRUCT_CLASS)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongInputType",
                "Input argument negE has incorrect data type; struct is expected.");
        if (mxGetNumberOfElements(sub_mx) != 1)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongSizeOfInputArg",
                "Argument negE should be a scalar.");
        marshallin_const_struct0_T(&pEmx->data[i].negE, sub_mx, "negE");

        sub_mx = mxGetField(mx, i, "negF");
        if (mxGetNumberOfElements(sub_mx) && mxGetClassID(sub_mx) != mxSTRUCT_CLASS)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongInputType",
                "Input argument negF has incorrect data type; struct is expected.");
        if (mxGetNumberOfElements(sub_mx) != 1)
            mexErrMsgIdAndTxt("marshallin_const_emxArray_struct1_T:WrongSizeOfInputArg",
                "Argument negF should be a scalar.");
        marshallin_const_struct0_T(&pEmx->data[i].negF, sub_mx, "negF");
    }
}
static void destroy_emxArray_struct1_T(emxArray_struct1_T *pEmx) {
    int                  nElems = numElements(pEmx->numDimensions, pEmx->size);
    int                  i;

    for (i=0; i<nElems; ++i) {
        free_emxArray((emxArray__common*)(pEmx->data[i].p));
        mxFree(pEmx->data[i].p);

        free_emxArray((emxArray__common*)(pEmx->data[i].q));
        mxFree(pEmx->data[i].q);

        free_emxArray((emxArray__common*)(pEmx->data[i].rowscal));
        mxFree(pEmx->data[i].rowscal);

        free_emxArray((emxArray__common*)(pEmx->data[i].colscal));
        mxFree(pEmx->data[i].colscal);

        destroy_struct2_T(&pEmx->data[i].L);
        destroy_struct2_T(&pEmx->data[i].U);
        free_emxArray((emxArray__common*)(pEmx->data[i].d));
        mxFree(pEmx->data[i].d);

        destroy_struct0_T(&pEmx->data[i].negE);
        destroy_struct0_T(&pEmx->data[i].negF);
    }
    free_emxArray((emxArray__common *)pEmx);
}


static void __gmresMILU_MGS_api(mxArray **plhs, const mxArray ** prhs) {
    struct0_T            A;
    emxArray_real_T      b;
    emxArray_struct1_T   M;
    int32_T              restart;
    real64_T             rtol;
    int32_T              maxit;
    emxArray_real_T      x0;
    int32_T              verbose;
    int32_T              nthreads;
    emxArray_real_T      x;
    int32_T             *flag;
    int32_T             *iter;
    emxArray_real_T      resids;

    /* Marshall in inputs and preallocate outputs */
    if (mxGetNumberOfElements(prhs[0]) && mxGetClassID(prhs[0]) != mxSTRUCT_CLASS)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongInputType",
            "Input argument A has incorrect data type; struct is expected.");
    if (mxGetNumberOfElements(prhs[0]) != 1)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongSizeOfInputArg",
            "Argument A should be a scalar.");
    marshallin_const_struct0_T(&A, prhs[0], "A");

    if (mxGetNumberOfElements(prhs[1]) && mxGetClassID(prhs[1]) != mxDOUBLE_CLASS)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongInputType",
            "Input argument b has incorrect data type; double is expected.");
    if (mxGetNumberOfElements(prhs[1]) && mxGetDimensions(prhs[1])[1] != 1) 
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongSizeOfInputArg",
            "Dimension 2 of b should equal 1.");
    alias_mxArray_to_emxArray(prhs[1], (emxArray__common *)(&b), "b", 1);

    if (mxGetNumberOfElements(prhs[2]) && mxGetClassID(prhs[2]) != mxSTRUCT_CLASS)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongInputType",
            "Input argument M has incorrect data type; struct is expected.");
    if (mxGetNumberOfElements(prhs[2]) && mxGetDimensions(prhs[2])[1] != 1) 
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongSizeOfInputArg",
            "Dimension 2 of M should equal 1.");
    marshallin_const_emxArray_struct1_T(&M, prhs[2], "M", 1);

    if (mxGetNumberOfElements(prhs[3]) && mxGetClassID(prhs[3]) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongInputType",
            "Input argument restart has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(prhs[3]) != 1)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongSizeOfInputArg",
            "Argument restart should be a scalar.");
    restart = *(int32_T*)mxGetData(prhs[3]);

    if (mxGetNumberOfElements(prhs[4]) && mxGetClassID(prhs[4]) != mxDOUBLE_CLASS)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongInputType",
            "Input argument rtol has incorrect data type; double is expected.");
    if (mxGetNumberOfElements(prhs[4]) != 1)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongSizeOfInputArg",
            "Argument rtol should be a scalar.");
    rtol = *(real64_T*)mxGetData(prhs[4]);

    if (mxGetNumberOfElements(prhs[5]) && mxGetClassID(prhs[5]) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongInputType",
            "Input argument maxit has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(prhs[5]) != 1)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongSizeOfInputArg",
            "Argument maxit should be a scalar.");
    maxit = *(int32_T*)mxGetData(prhs[5]);

    if (mxGetNumberOfElements(prhs[6]) && mxGetClassID(prhs[6]) != mxDOUBLE_CLASS)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongInputType",
            "Input argument x0 has incorrect data type; double is expected.");
    if (mxGetNumberOfElements(prhs[6]) && mxGetDimensions(prhs[6])[1] != 1) 
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongSizeOfInputArg",
            "Dimension 2 of x0 should equal 1.");
    alias_mxArray_to_emxArray(prhs[6], (emxArray__common *)(&x0), "x0", 1);

    if (mxGetNumberOfElements(prhs[7]) && mxGetClassID(prhs[7]) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongInputType",
            "Input argument verbose has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(prhs[7]) != 1)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongSizeOfInputArg",
            "Argument verbose should be a scalar.");
    verbose = *(int32_T*)mxGetData(prhs[7]);

    if (mxGetNumberOfElements(prhs[8]) && mxGetClassID(prhs[8]) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongInputType",
            "Input argument nthreads has incorrect data type; int32 is expected.");
    if (mxGetNumberOfElements(prhs[8]) != 1)
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongSizeOfInputArg",
            "Argument nthreads should be a scalar.");
    nthreads = *(int32_T*)mxGetData(prhs[8]);
    init_emxArray((emxArray__common*)(&x), 1);

    flag = mxMalloc(sizeof(int32_T));

    iter = mxMalloc(sizeof(int32_T));
    init_emxArray((emxArray__common*)(&resids), 1);

    /* Invoke the target function */
    gmresMILU_MGS(&A, &b, &M, restart, rtol, maxit, &x0, verbose, nthreads, &x, flag, iter, &resids);

    /* Deallocate input and marshall out function outputs */
    destroy_struct0_T(&A);
    free_emxArray((emxArray__common*)(&b));
    destroy_emxArray_struct1_T(&M);
    /* Nothing to be done for restart */
    /* Nothing to be done for rtol */
    /* Nothing to be done for maxit */
    free_emxArray((emxArray__common*)(&x0));
    /* Nothing to be done for verbose */
    /* Nothing to be done for nthreads */
    plhs[0] = move_emxArray_to_mxArray((emxArray__common*)(&x), mxDOUBLE_CLASS);
    mxFree(x.size);
    plhs[1] = move_scalar_to_mxArray(flag, mxINT32_CLASS);
    plhs[2] = move_scalar_to_mxArray(iter, mxINT32_CLASS);
    plhs[3] = move_emxArray_to_mxArray((emxArray__common*)(&resids), mxDOUBLE_CLASS);
    mxFree(resids.size);

}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Temporary copy for mex outputs. */
    mxArray *outputs[4] = {NULL, NULL, NULL, NULL};
    int i;
    int nOutputs = (nlhs < 1 ? 1 : nlhs);

    if (nrhs == 9) {
        if (nlhs > 4)
            mexErrMsgIdAndTxt("gmresMILU_MGS:TooManyOutputArguments",
                "Too many output arguments for entry-point gmresMILU_MGS.\n");
        /* Call the API function. */
        __gmresMILU_MGS_api(outputs, prhs);
    }
    else
        mexErrMsgIdAndTxt("gmresMILU_MGS:WrongNumberOfInputs",
            "Incorrect number of input variables for entry-point gmresMILU_MGS.");

    /* Copy over outputs to the caller. */
    for (i = 0; i < nOutputs; ++i) {
        plhs[i] = outputs[i];
    }
}
