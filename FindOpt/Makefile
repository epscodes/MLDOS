all: FindOpt

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = FindOpt FindOptStruct.o 

FindOpt: FindOptStruct.o 
	$(CLINKER) FindOptStruct.o -o FindOpt $(PETSC_LIB) ${CavityLibPath}/libcavity.a ${NloptLibPath}/libnlopt.a

FindOptStruct: FindOptStruct.o
	$(CLINKER) -Wall -g -o $@ FindOptStruct.o 
