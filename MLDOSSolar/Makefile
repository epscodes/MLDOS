all: MLDOSSolarOpt

#include ${PETSC_DIR}/conf/variables
#include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common
CLEANFILES = MLDOSSolarOpt mldossolar.o 

MLDOSSolarOpt: mldossolar.o 
	$(CLINKER) mldossolar.o -o MLDOSSolarOpt $(PETSC_LIB) ${CavityLibPath}/libcavity.a ${NloptLibPath}/libnlopt.a ${SLEPC_LIB}

mldossolar: mldossolar.o
	$(CLINKER) -Wall -g -o $@ mldossolar.o 
