all: AnalyzeOpt

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = AnalyzeOpt AnalyzeStructure.o 

AnalyzeOpt: AnalyzeStructure.o 
	$(CLINKER) AnalyzeStructure.o -o AnalyzeOpt $(PETSC_LIB) ${CavityLibPath}/libcavity.a ${NloptLibPath}/libnlopt.a 



AnalyzeStructure: AnalyzeStructure.o
	$(CLINKER) -Wall -g -o $@ AnalyzeStructure.o 
