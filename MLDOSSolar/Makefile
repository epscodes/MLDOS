all: MLDOSSolarOpt

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLDOSSolarOpt mldossolar.o Resonator.h 

MLDOSSolarOpt: mldossolar.o 
	$(CLINKER) mldossolar.o -o MLDOSSolarOpt $(PETSC_LIB) ${CavityLibPath}/libcavity.a ${NloptLibPath}/libnlopt.a 

mldossolar: mldossolar.o
	$(CLINKER) -Wall -g -o $@ mldossolar.o 
