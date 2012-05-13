all: MLDOSOpt

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLDOSOpt mldos.o 

MLDOSOpt: mldos.o 
	$(CLINKER) mldos.o -o MLDOSOpt ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libcavity.a $(PETSC_LIB)

mldos: mldos.o
	$(CLINKER) -Wall -g -o $@ mldos.o 
