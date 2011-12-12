all: MLDOSOpt

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLDOSOpt mldos.o 

MLDOSOpt: mldos.o 
	$(CLINKER) mldos.o -o MLDOSOpt $(PETSC_LIB) ${CavityLibPath}/libcavity.a ${NloptLibPath}/libnlopt.a 

mldos: mldos.o
	$(CLINKER) -Wall -g -o $@ mldos.o 
