all: MoperatorGeneral.o SourceGeneration.o PML.o Eps.o MathTools.o Output.o ldos.o  ResonatorSolver.o EigenSolver.o mympisetup.o ldoscal.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
#include ${SLEPC_DIR}/conf/slepc_common
CLEANFILES =  MoperatorGeneral.o SourceGeneration.o PML.o EPS.o MathTools.o Output.o ldos.o  ResonatorSolver.o EigenSolver.o  mympisetup.o ldoscal.o 

MoperatorGeneral: MoperatorGeneral.o
	$(CLINKER) -Wall -g  -o $@ MoperatorGeneral.o  
SourceGeneration: SourceGeneration.o
	$(CLINKER) -Wall -g -o $@ SourceGeneration.o  
PML:PML.o
	$(CLINKER) -Wall -g -o $@ PML.o  
Eps:Eps.o
	$(CLINKER) -Wall -g -o $@ Eps.o   
MathTools:MathTools.o
	$(CLINKER) -Wall -g -o $@ MathTools.o  
Output:Output.o
	$(CLINKER) -Wall -g -o $@ Output.o  
ldos:ldos.o
	$(CLINKER) -Wall -g -o $@ ldos.o  
ResonatorSolver:ResonatorSolver.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolver.o  
EigenSolver:EigenSolver.o
	$(CLINKER) -Wall -g -o $@ EigenSolver.o  
mympisetup:mympisetup.o
	$(CLINKER) -Wall -g -o $@ mympisetup.o  
ldoscal:ldoscal.o
	$(CLINKER) -Wall -g -o $@ ldoscal.o 