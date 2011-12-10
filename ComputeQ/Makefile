all: CompQ

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = CompQ ComputeQ.o 

CompQ: ComputeQ.o 
	$(CLINKER) ComputeQ.o -o CompQ $(PETSC_LIB) ${CavityLibPath}/libcavity.a ${NloptLibPath}/libnlopt.a  

ComputeQ: ComputeQ.o
	$(CLINKER) -Wall -g -o $@ ComputeQ.o 
