all: MLDOSPreOpt

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLDOSPreOpt mldospre.o 

MLDOSPreOpt: mldospre.o 
	$(CLINKER) mldospre.o -o MLDOSPreOpt  ${CavityLibPath}/libcavity.a ${NloptLibPath}/libnlopt.a $(PETSC_LIB)

mldospre: mldospre.o
	$(CLINKER) -Wall -g -o $@ mldospre.o 
