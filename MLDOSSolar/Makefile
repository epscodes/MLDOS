all: MLDOSSolarOpt  

export PETSC_DIR=${HOME}/MyLocal/petsc/petsc-3.3-p7
export PETSC_ARCH=arch-opt
export CavityLibPath=../CodesForLibrary
export NloptLibPath=${HOME}/MyLocal/nlopt_0/.libs
export C_INCLUDE_PATH=${HOME}/MyLocal/include
export SLEPC_DIR=${HOME}/MyLocal/slepc/slepc-3.3-p4

#include ${PETSC_DIR}/conf/variables
#include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common
CLEANFILES = MLDOSSolarOpt mldossolar.o ResonatorSolverSolar.o ResonatorSolverSolar2D.o MoperatorGeneralBloch.o  MoperatorGeneralBloch2D.o SolarEigenvaluesSolver.o

MLDOSSolarOpt: mldossolar.o ResonatorSolverSolar.o MoperatorGeneralBloch.o MoperatorGeneralBloch2D.o ResonatorSolverSolar2D.o SolarEigenvaluesSolver.o
	$(CLINKER) mldossolar.o ResonatorSolverSolar.o ResonatorSolverSolar2D.o  MoperatorGeneralBloch.o MoperatorGeneralBloch2D.o SolarEigenvaluesSolver.o -o MLDOSSolarOpt ${NloptLibPath}/libnlopt.a   ${CavityLibPath}/libcavity.a $(PETSC_LIB) $(SLEPC_LIB)

mldossolar: mldossolar.o
	$(CLINKER) -Wall -g -o $@ mldossolar.o 

ResonatorSolverSolar:ResonatorSolverSolar.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverSolar.o  
MoperatorGeneralBloch: MoperatorGeneralBloch.o
	$(CLINKER) -Wall -g  -o $@ MoperatorGeneralBloch.o  
MoperatorGeneralBloch2D: MoperatorGeneralBloch2D.o
	$(CLINKER) -Wall -g  -o $@ MoperatorGeneralBloch2D.o  
ResonatorSolverSolar2D:ResonatorSolverSolar2D.o
	$(CLINKER) -Wall -g -o $@ ResonatorSolverSolar2D.o  
SolarEigenvaluesSolver:SolarEigenvaluesSolver.o
	$(CLINKER) -Wall -g -o $@ SolarEigenvaluesSolver.o