#include <petsc.h>
#include <slepceps.h>
#include "Resonator.h"

extern Mat M, D, A;
extern Vec epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP;
extern int Nxyz;

extern char filenameComm[PETSC_MAX_PATH_LEN];

#undef _FUNCT_
#define _FUNCT_ "ComputeEigFun"
int ComputeEigFun(Vec epsCurrent, double omega)
{
  
  // update the diagonals of M Matrix;
  Mat Mone;
  MatDuplicate(M,MAT_COPY_VALUES,&Mone); // creat a copy of standar M (curl*muinv*curl);
  ModifyMatDiagonals(Mone, A, D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz,omega);
  
  PetscErrorCode ierr;
  ierr=PetscObjectSetName((PetscObject) Mone, "Mone");CHKERRQ(ierr);
  
  SlepcEigenSolver(Mone, epsCurrent, epspmlQ, D);

  MatDestroy(&Mone);
  return 0;

}

/*---this subrountie compute the eigenvalues lambda of the generalized eigenvalue problem (M-eps*omega_0^2) V = eps*lambda * V. In order to compute with mpb, user need to convert like this sqrt(lambda+omega_0^2)/(2*pi). */

#undef _FUNCT_
#define _FUNCT_ "SlepcEigenSolver"
int SlepcEigenSolver(Mat M, Vec epsCurrent, Vec epspmlQ, Mat D)
{

  PetscErrorCode ierr;
  EPS eps;
  PetscInt nconv;

  Mat B;
  int nrow, ncol;

  ierr=MatGetSize(M,&nrow, &ncol); CHKERRQ(ierr);

  ierr=MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nrow, ncol, 2, NULL, 2, NULL, &B); CHKERRQ(ierr);
  ierr=PetscObjectSetName((PetscObject)B, "epsmatrix"); CHKERRQ(ierr);
  

  if (D==PETSC_NULL)
    {  // for purely real epsC, no absorption;
      ierr=MatDiagonalSet(B,epsCurrent,INSERT_VALUES); CHKERRQ(ierr);
      MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
    }
  else
    {
      Vec epsC;
      VecDuplicate(epsCurrent, &epsC);
      ierr = VecPointwiseMult(epsC, epsCurrent,epspmlQ); CHKERRQ(ierr);
      MatSetTwoDiagonals(B, epsC, D, 1.0);
      VecDestroy(&epsC);
    }
  
/*
  ierr=PetscObjectSetName((PetscObject) epspmlQ, "epspmlQ"); CHKERRQ(ierr);
  OutputVec(PETSC_COMM_WORLD,epsCurrent,"MyVec","epsCurrent.m");
  OutputVec(PETSC_COMM_WORLD,epspmlQ,"MyVec","epspmlQ.m");
  OutputMat(PETSC_COMM_WORLD,M,"MyMat","M.m");
  OutputMat(PETSC_COMM_WORLD,B,"MyMat","B.m");
*/
  

  PetscPrintf(PETSC_COMM_WORLD,"!!!---computing eigenvalues---!!! \n");
  ierr=EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
  ierr=EPSSetOperators(eps, M, B); CHKERRQ(ierr);
  EPSSetFromOptions(eps);

  PetscLogDouble t1, t2, tpast;
  ierr = PetscTime(&t1);CHKERRQ(ierr);

  ierr=EPSSolve(eps); CHKERRQ(ierr);
  EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
  
  {
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    tpast = t2 - t1;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
      PetscPrintf(PETSC_COMM_SELF,"---The eigensolver time is %f s \n",tpast);
  }  

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of converged eigenpairs: %d\n\n",nconv);CHKERRQ(ierr);


  double *krarray, *kiarray, *errorarray;
  krarray = (double *) malloc(sizeof(double)*nconv);
  kiarray = (double *) malloc(sizeof(double)*nconv);
  errorarray =(double *) malloc(sizeof(double)*nconv);

  Vec xr, xi;
  ierr=MatGetVecs(M,PETSC_NULL,&xr); CHKERRQ(ierr);
  ierr=MatGetVecs(M,PETSC_NULL,&xi); CHKERRQ(ierr);
  ierr=PetscObjectSetName((PetscObject) xr, "xr"); CHKERRQ(ierr);
  ierr=PetscObjectSetName((PetscObject) xi, "xi"); CHKERRQ(ierr);
  int ni;
  for(ni=0; ni<nconv; ni++)
    {
      ierr=EPSGetEigenpair(eps, ni, krarray+ni, kiarray+ni, xr, xi);CHKERRQ(ierr);
      ierr = EPSComputeRelativeError(eps,ni,errorarray+ni);CHKERRQ(ierr);
      
      char bufferr[100], bufferi[100];
      sprintf(bufferr,"%.2dxr.m",ni+1);
      sprintf(bufferi,"%.2dxi.m",ni+1);

      OutputVec(PETSC_COMM_WORLD,xr,filenameComm,bufferr);
      OutputVec(PETSC_COMM_WORLD,xi,filenameComm,bufferi);
    }

  PetscPrintf(PETSC_COMM_WORLD, "Now print the eigenvalues: \n");
  for(ni=0; ni<nconv; ni++)
    PetscPrintf(PETSC_COMM_WORLD," %.12e%+.12ei,", krarray[ni], kiarray[ni]);

  PetscPrintf(PETSC_COMM_WORLD, "\n\nstart printing erros");

  for(ni=0; ni<nconv; ni++)
    PetscPrintf(PETSC_COMM_WORLD," %g,", errorarray[ni]);      

  PetscPrintf(PETSC_COMM_WORLD,"\n\n Finish EPS Solving !!! \n\n");

  /*-- destroy vectors and free space --*/
  EPSDestroy(&eps);
  MatDestroy(&B);
  VecDestroy(&xr);
  VecDestroy(&xi);

  free(krarray);
  free(kiarray);
  free(errorarray);

  PetscFunctionReturn(0);
}
