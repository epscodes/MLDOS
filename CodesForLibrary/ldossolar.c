#include <petsc.h>
#include <time.h>
#include "Resonator.h"

extern int count;
/*-----global variables for basic geometries-----*/
extern int Nx, Ny, Nz, Nxyz, BCPeriod, NJ, nkx, nky, nkz, nkxyz;
extern double hx, hy, hz, hxyz, omega, kxstep, kystep, kzstep, kxyzstep;
extern int bx[2], by[2], bz[2];
extern double *muinv;
extern int *JRandPos;
extern Vec epspmlQ, epsmedium, vR, epscoef;
extern Mat A, D;
extern IS from, to;
extern char filenameComm[PETSC_MAX_PATH_LEN];
extern double kxbase, kybase, kzbase;

#undef __FUNCT__ 
#define __FUNCT__ "ldossolar"
double ldossolar(int numofvar,double *varopt, double *grad, void *data)
{
PetscErrorCode ierr;
  Vec epsSReal, epsgrad; // create compatiable vectors with A.
  ierr = MatGetVecs(A,&epsSReal, &epsgrad); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr); 
 
  // copy epsopt to epsSReal;
  myfundataSolartype *ptmyfundata = (myfundataSolartype *)data;
  ierr=ArrayToVec(ptmyfundata->Sptepsinput,epsSReal); CHKERRQ(ierr);

  Vec epsCurrent; //epsCurrent is the current real epsilon everywhere; while epsC is calculated in ModifyMatDiagonals but with epsPML; here I need purely real epsilon;
  VecDuplicate(epsgrad,&epsCurrent); 
  ierr =MatMult(A, epsSReal,epsCurrent); CHKERRQ(ierr); 
  ierr = VecAXPY(epsCurrent,1.0,epsmedium); CHKERRQ(ierr);

  // Compute epsOmegasqr and epsOmegasqri;
  omega = varopt[0]; // now omega is degree of freedom;
  Vec epsOmegasqr, epsOmegasqri;
  VecDuplicate(epsgrad,&epsOmegasqr);
  VecDuplicate(epsgrad,&epsOmegasqri);
  ierr = VecPointwiseMult(epsOmegasqr, epsCurrent,epspmlQ); CHKERRQ(ierr);
  ierr = VecScale(epsOmegasqr, pow(omega,2)); CHKERRQ(ierr);
  ierr = MatMult(D,epsOmegasqr,epsOmegasqri); CHKERRQ(ierr);

  /*---------For each k, compute its ldos for all j------------------------*/
  int i, j, k;
  double ldos=0.0;
  
  for (i=0; i<nkx; i++)
    for (j=0; j<nky; j++)
      for (k =0; k<nkz; k++)
	{
	  double blochbc[3]={i*kxstep+kxbase*2*PI,j*kystep+kybase*2*PI,k*kzstep+kzbase*2*PI};
	  PetscPrintf(PETSC_COMM_WORLD,"Compute value at k-points (%f,%f,%f) \n", blochbc[0], blochbc[1], blochbc[2]);
	  double kldos;
	  SolarComputeKernel(epsCurrent, epsOmegasqr, epsOmegasqri, blochbc, &kldos, NULL);
	  ldos += kldos;
	}

  // take the average;
  ldos = ldos * kxyzstep;
  PetscPrintf(PETSC_COMM_WORLD,"---The ldos at omega  %.16e  is %.16e  (step %d) \n", omega,ldos, count);

  PetscPrintf(PETSC_COMM_WORLD,"---The average ldos at step %d  (including all currents) is %.16e \n", count,ldos); 
  PetscPrintf(PETSC_COMM_WORLD,"-------------------------------------------------------------- \n");

  if (grad)
    {
      PetscPrintf(PETSC_COMM_WORLD,"---Significantly wrong!!! Derivative is not provided! \n");
    }

  /*---Destroy Vectors *----*/
  ierr=VecDestroy(epsSReal);CHKERRQ(ierr);
  ierr=VecDestroy(epsgrad);CHKERRQ(ierr);
  ierr=VecDestroy(epsOmegasqr);CHKERRQ(ierr);
  ierr=VecDestroy(epsOmegasqri);CHKERRQ(ierr);
  ierr=VecDestroy(epsCurrent);CHKERRQ(ierr); 
 
  count++;
  return ldos;
}

