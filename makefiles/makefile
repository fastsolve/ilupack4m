
STARTDIR:=$(PWD)/..
MYSTARTDIR=$(STARTDIR)

TARGET=MATLAB

MATCHING=-D_MUMPS_MATCHING_
MUMPSLIB=-lmumps

include makefile.include_$(TARGET)

# where is your "mex" script located?
VPATH=$(STARTDIR)/matlab
MEXDIR=$(STARTDIR)/matlab/ilupack

DOBJECTS=$(MEXDIR)/partitionilupackmetisn.$(EXT)\
         $(MEXDIR)/symilupackmetisn.$(EXT)\
         $(MEXDIR)/symilupackmetise.$(EXT)\
         $(MEXDIR)/symilupackcamd.$(EXT)\
         $(MEXDIR)/symmwmilupackmetisn.$(EXT)\
         $(MEXDIR)/symmwmilupackmetise.$(EXT)\
         $(MEXDIR)/symmwmilupackamd.$(EXT)\
         $(MEXDIR)/symmwmilupackmmd.$(EXT)\
         $(MEXDIR)/symmwmilupackrcm.$(EXT)\
         $(MEXDIR)/symilupackcamd.$(EXT)\
         $(MEXDIR)/symmwmilupackmetisnsp.$(EXT)\
         $(MEXDIR)/symmwmilupackmetisesp.$(EXT)\
         $(MEXDIR)/symmwmilupackamdsp.$(EXT)\
         $(MEXDIR)/symmwmilupackrcmsp.$(EXT)\
         $(MEXDIR)/mwmilupackamd.$(EXT)\
         $(MEXDIR)/mwmilupackmetisn.$(EXT)\
         $(MEXDIR)/mwmilupackmetise.$(EXT)\
         $(MEXDIR)/mwmilupackmmd.$(EXT)\
         $(MEXDIR)/mwmilupackrcm.$(EXT)\
         $(MEXDIR)/mwmilupacknull.$(EXT)\
         $(MEXDIR)/DSYMilupacksqmr.$(EXT)\
         $(MEXDIR)/DSYMildlfactor.$(EXT)\
         $(MEXDIR)/ZHERildlfactor.$(EXT)\
         $(MEXDIR)/ZSYMildlfactor.$(EXT)\
         $(MEXDIR)/DSYMselinv.$(EXT)\
         $(MEXDIR)/ZHERselinv.$(EXT)\
         $(MEXDIR)/ZSYMselinv.$(EXT)\
         $(MEXDIR)/DSYMselbinv.$(EXT)\
         $(MEXDIR)/ZSYMselbinv.$(EXT)\
         $(MEXDIR)/ZHERselbinv.$(EXT)\
         $(MEXDIR)/DSYMldl2bldl.$(EXT)\
         $(MEXDIR)/ZSYMldl2bldl.$(EXT)\
         $(MEXDIR)/ZHERldl2bldl.$(EXT)\
         $(MEXDIR)/DGNLldu2bldu.$(EXT)\
         $(MEXDIR)/ZGNLldu2bldu.$(EXT)\
         $(MEXDIR)/Dinverse_aware.$(EXT)\
         $(MEXDIR)/Zinverse_aware.$(EXT)\
         $(MEXDIR)/Dgnl_inverse_aware.$(EXT)\
         $(MEXDIR)/Zgnl_inverse_aware.$(EXT)\
         $(MEXDIR)/DGNLselinv.$(EXT)\
         $(MEXDIR)/DGNLselbinv.$(EXT)\
         $(MEXDIR)/ZGNLselinv.$(EXT)\
         $(MEXDIR)/ZGNLselbinv.$(EXT)\
         $(MEXDIR)/DSYMLDLsol.$(EXT)\
         $(MEXDIR)/ZHERLDLsol.$(EXT)\
         $(MEXDIR)/ZSYMLDLsol.$(EXT)\
         $(MEXDIR)/DGNLLDUsol.$(EXT)\
         $(MEXDIR)/DGNLLDUtsol.$(EXT)\
         $(MEXDIR)/ZGNLLDUsol.$(EXT)\
         $(MEXDIR)/ZGNLLDUhsol.$(EXT)\
         $(MEXDIR)/DSPDilupackinit.$(EXT)\
         $(MEXDIR)/DSYMilupackinit.$(EXT)\
         $(MEXDIR)/DGNLilupackinit.$(EXT)\
         $(MEXDIR)/DSPDilupackfactor.$(EXT)\
         $(MEXDIR)/DSYMilupackfactor.$(EXT)\
         $(MEXDIR)/DGNLilupackfactor.$(EXT)\
         $(MEXDIR)/DSPDilupacksolver.$(EXT)\
         $(MEXDIR)/DSYMilupacksolver.$(EXT)\
         $(MEXDIR)/DGNLilupacksolver.$(EXT)\
         $(MEXDIR)/DGNLilupacktsolver.$(EXT)\
         $(MEXDIR)/DSPDilupackdelete.$(EXT)\
         $(MEXDIR)/DSYMilupackdelete.$(EXT)\
         $(MEXDIR)/DGNLilupackdelete.$(EXT)\
         $(MEXDIR)/DSYMSPDilupacksolver.$(EXT)\
         $(MEXDIR)/DGNLSPDilupacksolver.$(EXT)\
         $(MEXDIR)/DGNLSYMilupacksolver.$(EXT)\
         $(MEXDIR)/DGNLSPDilupacktsolver.$(EXT)\
         $(MEXDIR)/DGNLSYMilupacktsolver.$(EXT)\
         $(MEXDIR)/Dloadhbo.$(EXT)\
         $(MEXDIR)/DSYMsavehbo.$(EXT)\
         $(MEXDIR)/DGNLsavehbo.$(EXT)\
         $(MEXDIR)/DSPDilupacksol.$(EXT)\
         $(MEXDIR)/DSYMilupacksol.$(EXT)\
         $(MEXDIR)/DGNLilupacksol.$(EXT)\
         $(MEXDIR)/DGNLilupacktsol.$(EXT)\
         $(MEXDIR)/DSYMSPDilupackconvert.$(EXT)

ZOBJECTS=$(MEXDIR)/ZSYMilupacksqmr.$(EXT)\
         $(MEXDIR)/ZHERilupacksqmr.$(EXT)\
         $(MEXDIR)/ZHPDilupackinit.$(EXT)\
         $(MEXDIR)/ZSYMilupackinit.$(EXT)\
         $(MEXDIR)/ZHERilupackinit.$(EXT)\
         $(MEXDIR)/ZGNLilupackinit.$(EXT)\
         $(MEXDIR)/ZHPDilupackfactor.$(EXT)\
         $(MEXDIR)/ZSYMilupackfactor.$(EXT)\
         $(MEXDIR)/ZHERilupackfactor.$(EXT)\
         $(MEXDIR)/ZGNLilupackfactor.$(EXT)\
         $(MEXDIR)/ZHPDilupacksolver.$(EXT)\
         $(MEXDIR)/ZSYMilupacksolver.$(EXT)\
         $(MEXDIR)/ZHERilupacksolver.$(EXT)\
         $(MEXDIR)/ZGNLilupacksolver.$(EXT)\
         $(MEXDIR)/ZGNLilupacktsolver.$(EXT)\
         $(MEXDIR)/ZGNLilupackhsolver.$(EXT)\
         $(MEXDIR)/ZHPDilupackdelete.$(EXT)\
         $(MEXDIR)/ZSYMilupackdelete.$(EXT)\
         $(MEXDIR)/ZHERilupackdelete.$(EXT)\
         $(MEXDIR)/ZGNLilupackdelete.$(EXT)\
         $(MEXDIR)/ZGNLHPDilupacksolver.$(EXT)\
         $(MEXDIR)/ZGNLHPDilupacktsolver.$(EXT)\
         $(MEXDIR)/ZGNLHPDilupackhsolver.$(EXT)\
         $(MEXDIR)/ZGNLSYMilupacksolver.$(EXT)\
         $(MEXDIR)/ZGNLSYMilupacktsolver.$(EXT)\
         $(MEXDIR)/ZGNLSYMilupackhsolver.$(EXT)\
         $(MEXDIR)/ZGNLHERilupacksolver.$(EXT)\
         $(MEXDIR)/ZGNLHERilupacktsolver.$(EXT)\
         $(MEXDIR)/ZGNLHERilupackhsolver.$(EXT)\
         $(MEXDIR)/ZHERHPDilupacksolver.$(EXT)\
         $(MEXDIR)/ZGNLDSPDilupacksolver.$(EXT)\
         $(MEXDIR)/ZGNLDSPDilupacktsolver.$(EXT)\
         $(MEXDIR)/ZGNLDSPDilupackhsolver.$(EXT)\
         $(MEXDIR)/ZGNLDSYMilupacksolver.$(EXT)\
         $(MEXDIR)/ZGNLDSYMilupacktsolver.$(EXT)\
         $(MEXDIR)/ZGNLDSYMilupackhsolver.$(EXT)\
         $(MEXDIR)/ZGNLDGNLilupacksolver.$(EXT)\
         $(MEXDIR)/ZGNLDGNLilupacktsolver.$(EXT)\
         $(MEXDIR)/ZGNLDGNLilupackhsolver.$(EXT)\
         $(MEXDIR)/ZHERDSPDilupacksolver.$(EXT)\
         $(MEXDIR)/ZHERDSYMilupacksolver.$(EXT)\
         $(MEXDIR)/ZSYMDSPDilupacksolver.$(EXT)\
         $(MEXDIR)/ZSYMDSYMilupacksolver.$(EXT)\
         $(MEXDIR)/ZHPDDSPDilupacksolver.$(EXT)\
         $(MEXDIR)/Zloadhbo.$(EXT)\
         $(MEXDIR)/ZGNLsavehbo.$(EXT)\
         $(MEXDIR)/ZHERsavehbo.$(EXT)\
         $(MEXDIR)/ZSYMsavehbo.$(EXT)\
         $(MEXDIR)/ZHPDilupacksol.$(EXT)\
         $(MEXDIR)/ZSYMilupacksol.$(EXT)\
         $(MEXDIR)/ZHERilupacksol.$(EXT)\
         $(MEXDIR)/ZGNLilupacksol.$(EXT)\
         $(MEXDIR)/ZGNLilupacktsol.$(EXT)\
         $(MEXDIR)/ZHERHPDilupackconvert.$(EXT)

OBJECTS=$(DOBJECTS) $(ZOBJECTS)
ILU_TOPFUNCS=generic_*.o Dpartition_metis_N.o Dperm_matching_null.o \
     Dqsortpair.o [DZ]readmtc.o [DZ]writemtc.o [DZC]symspd.o [DZ]symiluc.o \
     Zsymilucs.o ZsymAMGsolver_real_spd_precs.o Dilupackinit.o

all: $(OBJECTS)

clean:
	rm -f $(OBJECTS) *.o

.SUFFIXES: .c .$(EXT)

$(MEXDIR)/%.$(EXT): %.c
	$(CMEX) -O $(MATCHING) $(FORTRANNAMES) $(LONGINTEGER) -I../include -I../include/$(TARGET) \
	$(OPTS) $(OUTPUT) $(MEXDIR)/$*.$(EXT) \
	$< \
	-L$(STARTDIR)/lib/$(PLATFORM) -lilupack $(LAPACK)
	rm -f $*.o

$(DOBJECTS): ../lib/$(PLATFORM)/libilupack.so
$(ZOBJECTS): ../lib/$(PLATFORM)/libilupack.so

../lib/$(PLATFORM)/libilupack.so: ../lib/$(PLATFORM)/libilupack_mumps.a
	cd ../lib/$(PLATFORM) && ar -x libilupack_mumps.a && \
	objcopy --localize-symbol=dsymilucupdate_ Dsymiluc.o && \
	objcopy --localize-symbol=zherilucupdate_ Zsymiluc.o && \
	objcopy --localize-symbol=zsymilucupdate_ Zsymilucs.o && \
	gcc -shared -o libilupack.so $(ILU_TOPFUNCS) -L. \
	-lilupack_mumps -lmumps $(MUMPSLIB) \
	-lsparspak -lblaslike -lmetisomp -lmetis -lmetisomp \
	-lcamd -lamd  -lsuitesparseconfig -lm -lc -lgfortran -lgomp && \
	rm -f *.o

../lib/$(PLATFORM)/libilupack_mumps.a:
	mkdir -p ../lib/$(PLATFORM)
	curl -s -L https://raw.githubusercontent.com/hpdata/gdutil/master/gd_get_pub.py | python - -i 0ByTwsK5_Tl_PN09kZHdNZlRwSkk
	unzip ilupack05102016.zip
	mv ilupack/lib/$(PLATFORM)/* ../lib/$(PLATFORM)/
	rm -rf ilupack05102016.zip ilupack
