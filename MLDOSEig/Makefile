all: MLDOSEigCal

export PETSC_DIR=${HOME}/MyLocal/petsc/petsc-3.3-p7
export PETSC_ARCH=arch-opt
export CavityLibPath=../CodesForLibrary
export C_INCLUDE_PATH=${HOME}/MyLocal/include
export SLEPC_DIR=${HOME}/MyLocal/slepc/slepc-3.3-p4

#include ${PETSC_DIR}/conf/variables
#include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common

CLEANFILES = MLDOSEigCal mldoseig.o SlepcEigenSolver.o

MLDOSEigCal: mldoseig.o SlepcEigenSolver.o
	$(CLINKER) mldoseig.o SlepcEigenSolver.o -o MLDOSEigCal ${CavityLibPath}/libcavity.a $(PETSC_LIB) $(SLEPC_LIB)

mldoseig: mldoseig.o
	$(CLINKER) -Wall -g -o $@ mldoseig.o 

SlepcEigenSolver: SlepcEigenSolver.o
	$(CLINKER) -Wall -g -o $@ SlepcEigenSolver.o