#include <petsc.h>
#include <time.h>
#include "Resonator.h"

extern int count;
extern int its;
extern int maxit;

/*--------global variables---------*/
extern int Nx, Ny, Nz, Nxyz;
extern double hx, hy, hz, hxyz, omega;
extern Vec epspmlQ, epsC, epsCi, epsP, x, b, weightedJ, epsSReal;
extern Mat A, D, M;
extern KSP ksp;

/*-------global variables for Job 3 -----*/
extern double omegacur;
extern double ldoscenter;

#undef __FUNCT__ 
#define __FUNCT__ "ldosdiff"
double ldosdiff(int numofvar, double *varopt, double *grad, void *data)
{
  double ldosnew;
  ldosnew = ldos(numofvar,varopt,grad,data);
  ldosnew = fabs(ldosnew-ldoscenter);
  PetscPrintf(PETSC_COMM_WORLD,"---The relative difference  is %.16f  \n",(ldosnew/ldoscenter));
  return ldosnew;
}


#undef __FUNCT__ 
#define __FUNCT__ "ldos"
double ldos(int numofvar,double *varopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;
  
  double omegasqrdiff;
  omegasqrdiff=pow(varopt[0],2)-pow(omegacur,2);
  omegacur = varopt[0]; //update omegacur to current omega;

  // Update the diagonals of M Matrix;
  ModifyMatDiagonalsForOmega(M, A,D, epsSReal, epspmlQ, epsC, epsCi, epsP, Nxyz,omegasqrdiff);   
  
  #if 1
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  #endif
  /*-----------------KSP Solving------------------*/ 

#if 1 
  if (its> ceil(maxit*0.8) || count< 2 )
    {
      PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern, LU is redone! \n");
      ierr = KSPSetOperators(ksp,M,M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  else
    {ierr = KSPSetOperators(ksp,M,M,SAME_PRECONDITIONER);CHKERRQ(ierr);}
   ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
   ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %D----\n ",its);CHKERRQ(ierr);
#endif

   // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
#if 1
  {
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    if(its>(maxit-2))
      {
	PetscPrintf(PETSC_COMM_WORLD,"Too many iterations needed! Recomputing \n");
	ierr = KSPSetOperators(ksp,M,M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %D----\n ",its);CHKERRQ(ierr);
     }
  }
#endif

//Print kspsolving information

#if 1
  double norm;
  Vec xdiff;
  ierr=VecDuplicate(x,&xdiff);CHKERRQ(ierr);
  ierr = MatMult(M,x, xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
  //ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"---Norm of error %A, Kryolv Iterations %D----\n ",norm,its);CHKERRQ(ierr);    
  ierr=VecDestroy(xdiff);CHKERRQ(ierr);
#endif

  /*--------------Finish KSP Solving---------------*/
#if 1
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0)
  PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);
#endif   

  double l; //ldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  ierr = VecDot(x,weightedJ,&l);


  l = -1.0*l*hxyz*omegacur/omega; // omegacur/omega is due to the i*omega*J in the RHS.
  PetscPrintf(PETSC_COMM_WORLD,"---The ldos at omega  %.16e  is %.16e  (step %d) \n", omegacur,l,count);
  
  /*-----take care of the gradient-------*/
  if (grad) {
      PetscPrintf(PETSC_COMM_WORLD,"---Significantly wrong!!! Derivative is not provided! \n");
  }

  count++;
  return l;
}
