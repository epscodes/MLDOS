all: MoperatorGeneral.o SourceGeneration.o PML.o Eps.o MathTools.o Output.o ldos.o  ResonatorSolver.o EigenSolver.o ResonatorSolverRHS.o ResonatorSolverPOLXY.o ResonatorSolverSolar.o MoperatorGeneralBloch.o MoperatorGeneralBloch2D.o ResonatorSolverSolar2D.o  mympisetup.o ResonatorSolverPre.o prefun.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
#include ${SLEPC_DIR}/conf/slepc_common
CLEANFILES =  MoperatorGeneral.o SourceGeneration.o PML.o EPS.o MathTools.o Output.o ldos.o  ResonatorSolver.o EigenSolver.o ResonatorSolverRHS.o ResonatorSolverPOLXY.o ResonatorSolverSolar.o MoperatorGeneralBloch.o  MoperatorGeneralBloch2D.o ResonatorSolverSolar2D.o SolarEigenvaluesSolver.o mympisetup.o  ResonatorSolverPre.o prefun.o

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
ResonatorSolverRHS:ResonatorSolverRHS.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverRHS.o
ResonatorSolverPOLXY:ResonatorSolverPOLXY.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverPOLXY.o
ResonatorSolverSolar:ResonatorSolverSolar.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverSolar.o
MoperatorGeneralBloch: MoperatorGeneralBloch.o
	$(CLINKER) -Wall -g  -o $@ MoperatorGeneralBloch.o
MoperatorGeneralBloch2D: MoperatorGeneralBloch2D.o
	$(CLINKER) -Wall -g  -o $@ MoperatorGeneralBloch2D.o
ResonatorSolverSolar2D:ResonatorSolverSolar2D.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverSolar2D.o
#SolarEigenvaluesSolver:SolarEigenvaluesSolver.o
#	$(CLINKER) -Wall -g -o $@ SolarEigenvaluesSolver.o
mympisetup:mympisetup.o
	$(CLINKER) -Wall -g -o $@ mympisetup.o
ResonatorSolverPre:ResonatorSolverPre.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverPre.o
prefun: prefun.o
	$(CLINKER) -Wall -g -o $@ prefun.o