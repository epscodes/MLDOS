all: MLDOSPOLXYOpt

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLDOSPOLXYOpt mldospolxy.o 

MLDOSPOLXYOpt: mldospolxy.o 
	$(CLINKER) mldospolxy.o -o MLDOSPOLXYOpt $(PETSC_LIB) ${CavityLibPath}/libcavity.a ${NloptLibPath}/libnlopt.a 

mldospolxy: mldospolxy.o
	$(CLINKER) -Wall -g -o $@ mldospolxy.o 
