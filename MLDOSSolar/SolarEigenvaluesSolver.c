#include <petsc.h>
#include "slepceps.h"
#include "Resonator.h"

extern double omega;

/*---this subrountie compute the eigenvalues lambda of the generalized eigenvalue problem (M-eps*omega_0^2) V = eps*lambda * V. In order to compute with mpb, user need to convert like this sqrt(lambda+omega_0^2)/(2*pi). */

#undef _FUNCT_
#define _FUNCT_ "SolarEigenvalueSolver"
int SolarEigenvaluesSolver(Mat M, Vec epsCurrent, Vec epspmlQ, Mat D)
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


  PetscPrintf(PETSC_COMM_WORLD,"!!!---computing eigenvalues---!!! \n");
  ierr=EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
  ierr=EPSSetOperators(eps, M, B); CHKERRQ(ierr);
  //ierr=EPSSetProblemType(eps,EPS_PGNHEP);CHKERRQ(ierr);
  EPSSetFromOptions(eps);

  PetscLogDouble t1, t2, tpast;
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);

  ierr=EPSSolve(eps); CHKERRQ(ierr);
  EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
  
  {
    ierr = PetscGetTime(&t2);CHKERRQ(ierr);
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

  int ni;
  for(ni=0; ni<nconv; ni++)
    {
      ierr=EPSGetEigenpair(eps,ni, krarray+ni,kiarray+ni,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr = EPSComputeRelativeError(eps,ni,errorarray+ni);CHKERRQ(ierr);
      ierr=EPSComputeRelativeError(eps, ni, errorarray+ni );
    }

  PetscPrintf(PETSC_COMM_WORLD, "Now print the eigenvalues: \n");
  for(ni=0; ni<nconv; ni++)
    PetscPrintf(PETSC_COMM_WORLD," %g%+gi,", krarray[ni], kiarray[ni]);

  PetscPrintf(PETSC_COMM_WORLD, "\n\nNow print the normalized eigenvalues: \n");
  for(ni=0; ni<nconv; ni++)
    PetscPrintf(PETSC_COMM_WORLD," %g%+gi,", sqrt(krarray[ni]+pow(omega,2))/(2*PI),kiarray[ni]);


  PetscPrintf(PETSC_COMM_WORLD, "\n\nstart printing erros");

  for(ni=0; ni<nconv; ni++)
    PetscPrintf(PETSC_COMM_WORLD," %g,", errorarray[ni]);      

  PetscPrintf(PETSC_COMM_WORLD,"\n\n Finish EPS Solving !!! \n\n");

  /*-- destroy vectors and free space --*/
  EPSDestroy(&eps);
  MatDestroy(&B);

  free(krarray);
  free(kiarray);
  free(errorarray);

  PetscFunctionReturn(0);
}
