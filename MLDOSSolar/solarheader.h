#include <petsc.h>


// from MoperatorGeneralBloch.c
PetscErrorCode MoperatorGeneralBloch(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double hx, double hy, double hz, int bx[2], int by[2], int bz[2], double *muinv, int DimPeriod, double blochbc[3], Vec epsOmegasqr, Vec epsOmegasqri);

// from MoperatorGeneralBloch2D.c
int MoperatorGeneralBloch2D(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double hx, double hy, double hz, int bx[2], int by[2], int bz[2], double *muinv, int DimPeriod, double blochbc[3], Vec epsOmegasqr, Vec epsOmegasqri);

// from ResonatorSolverSolar.c
double ResonatorSolverSolar(int Mxyz,double *epsopt, double *grad, void *data);
PetscErrorCode SolarComputeKernel(Vec epsCurrent, Vec epsOmegasqr, Vec epsOmegasqri, double blochbc[3], double *kdlos, Vec kepsgrad);
double ldossolar(int numofvar, double *varopt, double *grad, void *data);

// from ResonatorSolverSolar2D.c
double ResonatorSolverSolar2D(int Mxyz,double *epsopt, double *grad, void *data);
PetscErrorCode SolarComputeKernel2D(Vec epsCurrent, Vec epsOmegasqr, Vec epsOmegasqri, double blochbc[3], double *kdlos, Vec kepsgrad);
double ldossolar2D(int numofvar, double *varopt, double *grad, void *data);

double boweightfun(double epsjloc);
double boweightfundev(double epsjloc);
double smoothstepfun(double x);
double smoothstepfundev(double x);

// from SolarEgienvaluesSolver.c
int SolarEigenvaluesSolver(Mat M, Vec epsCurrent, Vec epspmlQ, Mat D);


// from ResonatorSolverSolar2D.c
double ResonatorSolverSolar2D(int Mxyz,double *epsopt, double *grad, void *data);

int SolarComputeKernel2D(Vec epsCurrent, Vec epsOmegasqr, Vec epsOmegasqri, double blochbc[3], double *ptkldos, Vec kepsgrad);
