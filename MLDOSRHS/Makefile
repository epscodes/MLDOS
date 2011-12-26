all: MLDOSRHSOpt

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLDOSRHSOpt mldos.o 

MLDOSRHSOpt: mldosrhs.o 
	$(CLINKER) mldosrhs.o -o MLDOSRHSOpt $(PETSC_LIB) ${CavityLibPath}/libcavity.a ${NloptLibPath}/libnlopt.a 

mldosrhs: mldosrhs.o
	$(CLINKER) -Wall -g -o $@ mldosrhs.o 
