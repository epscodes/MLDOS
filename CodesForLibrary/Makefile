all: MoperatorGeneral.o SourceGeneration.o PML.o Eps.o MathTools.o Output.o ldos.o  ResonatorSolver.o EigenSolver.o ResonatorSolverRHS.o ResonatorSolverPOLXY.o ResonatorSolverSolar.o MoperatorGeneralBloch.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES =  MoperatorGeneral.o SourceGeneration.o PML.o EPS.o MathTools.o Output.o ldos.o  ResonatorSolver.o EigenSolver.o ResonatorSolverRHS.o ResonatorSolverPOLXY.o ResonatorSolverSolar.o MoperatorGeneralBloch.o

MoperatorGeneral: MoperatorGeneral.o
	$(CLINKER) -Wall -g  -o $@ MoperatorGeneral.o $(PETSC_LIB)
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
ResonatorSolverRHS:ResonatorSolverRHS.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverRHS.o
ResonatorSolverPOLXY:ResonatorSolverPOLXY.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverPOLXY.o
ResonatorSolverSolar:ResonatorSolverSolar.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverSolar.o
MoperatorGeneralBloch: MoperatorGeneralBloch.o
	$(CLINKER) -Wall -g  -o $@ MoperatorGeneralBloch.o


