all: MLDOSMaxMinNOpt

export PETSC_DIR=${HOME}/MyLocal/petsc-dev
export PETSC_ARCH=arch-opt
export CavityLibPath=../CodesForLibrary
export NloptLibPath=${HOME}/MyLocal/nlopt_0/.libs
export C_INCLUDE_PATH=${HOME}/MyLocal/include

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLDOSMaxMinNOpt mldosmaxminn.o 

MLDOSMaxMinNOpt: mldosmaxminn.o maxminnobjandconstraint.o ldosmaxminnkernel.o ldosmaxminneigensolver.o
	$(CLINKER) mldosmaxminn.o  maxminnobjandconstraint.o ldosmaxminnkernel.o ldosmaxminneigensolver.o -o MLDOSMaxMinNOpt ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libcavity.a $(PETSC_LIB)

mldosmaxminn: mldosmaxminn.o
	$(CLINKER) -Wall -g -o $@ mldosmaxminn.o 

maxminnobjandconstraint: maxminnobjandconstraint.o
	$(CLINKER) -Wall -g -o $@ maxminnobjandconstraint.o

ldosmaxminnkernel: ldosmaxminnkernel.o
	$(CLINKER) -Wall -g -o $@ ldosmaxminnkernel.o
ldosmaxminneigensolver: ldosmaxminneigensolver.o
	$(CLINKER) -Wall -g -o $@ ldosmaxminneigensolver.o