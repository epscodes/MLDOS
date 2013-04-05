all: MLDOSSHGOpt

export PETSC_DIR=/home/xdliang/MyLocal/petsc-dev
export PETSC_ARCH=arch-opt
export CavityLibPath=/home/xdliang/MyCodes/CavityPETScCodesDev_0/CodesForLibrary
export NloptLibPath=${HOME}/MyLocal/nlopt_0/.libs

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = MLDOSSHGOpt mldosshg.o 

MLDOSSHGOpt: mldosshg.o shgobjandconstraint.o ldoskernel.o
	$(CLINKER) mldosshg.o  shgobjandconstraint.o ldoskernel.o -o MLDOSSHGOpt ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libcavity.a $(PETSC_LIB)

mldosshg: mldosshg.o
	$(CLINKER) -Wall -g -o $@ mldosshg.o 

shgobjandconstraint: shgobjandconstraint.o
	$(CLINKER) -Wall -g -o $@ shgobjandconstraint.o

ldoskernel: ldoskernel.o
	$(CLINKER) -Wall -g -o $@ ldoskernel.o