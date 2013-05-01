all: MLDOSMaxMinNOpt

export PETSC_DIR=/home/xdliang/MyLocal/petsc-dev
export PETSC_ARCH=arch-opt
export CavityLibPath=/home/xdliang/MyCodes/CavityPETScCodesDev_0/CodesForLibrary
export NloptLibPath=${HOME}/MyLocal/nlopt_0/.libs

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLDOSMaxMinNOpt mldosmaxminn.o 

MLDOSMaxMinNOpt: mldosmaxminn.o maxminnobjandconstraint.o ldosmaxminnkernel.o
	$(CLINKER) mldosmaxminn.o  maxminnobjandconstraint.o ldosmaxminnkernel.o -o MLDOSMaxMinNOpt ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libcavity.a $(PETSC_LIB)

mldosmaxminn: mldosmaxminn.o
	$(CLINKER) -Wall -g -o $@ mldosmaxminn.o 

maxminnobjandconstraint: maxminnobjandconstraint.o
	$(CLINKER) -Wall -g -o $@ maxminnobjandconstraint.o

ldosmaxminnkernel: ldosmaxminnkernel.o
	$(CLINKER) -Wall -g -o $@ ldosmaxminnkernel.o