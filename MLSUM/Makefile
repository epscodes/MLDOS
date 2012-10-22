all: MLSUMCal

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLSUMCal mlsum.o 

MLSUMCal: mlsum.o 
	$(CLINKER) mlsum.o -o MLSUMCal ${CavityLibPath}/libcavity.a $(PETSC_LIB)

mlsum: mlsum.o
	$(CLINKER) -Wall -g -o $@ mlsum.o 
