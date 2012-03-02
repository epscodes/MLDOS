#include <petsc.h>
#include <time.h>
#include "Resonator.h"

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

#undef __FUNCT__ 
#define __FUNCT__ "ldossolar"
double ldossolar(int numofvar,double *varopt, double *grad, void *data)
{
PetscErrorCode ierr;
  Vec epsSReal, epsgrad, vgrad; // create compatiable vectors with A.
  ierr = MatGetVecs(A,&epsSReal, &epsgrad); CHKERRQ(ierr);
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vgrad, "vgrad");CHKERRQ(ierr);

  VecSet(epsgrad,0.0);
  VecAssemblyBegin(epsgrad);
  VecAssemblyEnd(epsgrad);
 
  // copy epsopt to epsSReal;
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);

  Vec epsCurrent; //epsCurrent is the current real epsilon everywhere; while epsC is calculated in ModifyMatDiagonals but with epsPML; here I need purely real epsilon;
  VecDuplicate(epsgrad,&epsCurrent); 
  ierr =MatMult(A, epsSReal,epsCurrent); CHKERRQ(ierr); 
  ierr = VecAXPY(epsCurrent,1.0,epsmedium); CHKERRQ(ierr);

  // Compute epsOmegasqr and epsOmegasqri;
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
	  double blochbc[3]={i*kxstep,j*kystep,k*kzstep};
	  double kldos;
	  Vec kepsgrad;
	  VecDuplicate(epsgrad,&kepsgrad);
	  SolarComputeKernel(epsCurrent, epsOmegasqr, epsOmegasqri, blochbc, &kldos, kepsgrad);
	  ldos += kldos;
	  ierr=VecAXPY(epsgrad,1.0,kepsgrad); CHKERRQ(ierr);
	  ierr=VecDestroy(kepsgrad); CHKERRQ(ierr);
	}

  // take the average;
  ldos = ldos * kxyzstep;
  ierr=VecScale(epsgrad,kxyzstep); CHKERRQ(ierr);

  // set imaginary part of epsgrad = 0; ( we're only interested in real part;
  ierr = VecPointwiseMult(epsgrad,epsgrad,vR); CHKERRQ(ierr);

  // vgrad =A'*epsgrad; A' is the restriction matrix; Mapped to the small grid;
  ierr = MatMultTranspose(A,epsgrad,vgrad);CHKERRQ(ierr);   

  // copy vgrad (distributed vector) to a regular array grad;
  VecScatter scatter;
  Vec vgradlocal;
  ierr = VecCreateSeq(PETSC_COMM_SELF, Mxyz, &vgradlocal); CHKERRQ(ierr);
  ierr = VecToArray(vgrad,grad,scatter,from,to,vgradlocal,Mxyz);

  PetscPrintf(PETSC_COMM_WORLD,"---The average ldos at step %d  (including all currents) is %.16e \n", count,ldos); 
  PetscPrintf(PETSC_COMM_WORLD,"-------------------------------------------------------------- \n");

 
  /* Now store the epsilon at each step*/
  char buffer [100];

  int STORE=1;    
  if(STORE==1)
    {
      sprintf(buffer,"%.5depsSReal.m",count);
      OutputVec(PETSC_COMM_WORLD, epsSReal, filenameComm, buffer);      
    }

  /* Now store the epsilon which makes improvement */
  int cSTORE=0;
  if (cSTORE==1 && ldos > cldos)
    {
      sprintf(buffer,"%.5depsSRealMONT.m",ccount);
      OutputVec(PETSC_COMM_WORLD, epsSReal, filenameComm, buffer);
      cldos=ldos;
      ccount++;
    }

  /*---Destroy Vectors *----*/
  ierr=VecDestroy(epsSReal);CHKERRQ(ierr);
  ierr=VecDestroy(epsgrad);CHKERRQ(ierr);
  ierr=VecDestroy(epsOmegasqr);CHKERRQ(ierr);
  ierr=VecDestroy(epsOmegasqri);CHKERRQ(ierr);
  ierr=VecDestroy(vgrad);CHKERRQ(ierr);
  ierr=VecDestroy(vgradlocal);CHKERRQ(ierr);
  ierr=VecDestroy(epsCurrent);CHKERRQ(ierr); 
 
  count++;
  return ldos;






}
