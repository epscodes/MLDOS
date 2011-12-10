#include <petsc.h>
#include <time.h>
#include "Resonator.h"

extern int count;
extern int its;
extern int maxit;

#undef __FUNCT__ 
#define __FUNCT__ "ldosdiff"
double ldosdiff(int numofvar, double *varopt, double *grad, void *data)
{
  double ldosnew, lcenter;
  ldosnew = ldos(numofvar,varopt,grad,data);
  myfundatatypeq *ptmyfundata = (myfundatatypeq *) data;
  lcenter = ptmyfundata->Ssparedouble;
  ldosnew =  fabs(ldosnew-lcenter);
  PetscPrintf(PETSC_COMM_WORLD,"---The relative difference  is %.16f  \n",(ldosnew/lcenter));
  return ldosnew;
}


#undef __FUNCT__ 
#define __FUNCT__ "ldos"
double ldos(int numofvar,double *varopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;
  
  myfundatatypeq *ptmyfundata = (myfundatatypeq *) data;
  
  int Nx = ptmyfundata->SNx;
  int Ny = ptmyfundata->SNy;
  int Nz = ptmyfundata->SNz;
  double hx = ptmyfundata->Shx;
  double hy = ptmyfundata->Shy;
  double hz = ptmyfundata->Shz;
  double omega = ptmyfundata->Somega;
  KSP ksp = ptmyfundata->Sksp;
  Vec epspmlQ = ptmyfundata->SepspmlQ;
  Vec epsC = ptmyfundata->SepsC;
  Vec epsCi = ptmyfundata->SepsCi;
  Vec epsP = ptmyfundata->SepsP;
  Vec x = ptmyfundata->Sx;
  Vec b = ptmyfundata->Sb;
  Vec weightedJ = ptmyfundata->SweightedJ;
  Vec epsSReal = ptmyfundata->SepsSReal;

  Mat A = ptmyfundata->SA;
  Mat D = ptmyfundata->SD;
  Mat M = ptmyfundata->SM;

  int Nxyz = Nx*Ny*Nz;
 
  // copy omegaopt to omega; !!! IMPORTANT CHANGE!!!

  //spareptdouble is the address to store previous omega;
  double *spareptdouble = ptmyfundata->Sspareptdouble;

  double omegasqrdiff;
  omegasqrdiff=pow(varopt[0],2)-pow(spareptdouble[0],2);

  spareptdouble[0] = varopt[0]; //update spareptdouble to current omega;

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

  double hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;
  l = -1.0*l*hxyz*spareptdouble[0]/omega; // spareptdouble[0]/omega is due to the i*omega*J in the RHS.
  PetscPrintf(PETSC_COMM_WORLD,"---The ldos at omega  %.16e  is %.16e  (step %d) \n", *spareptdouble,l,count);
  

  /*-----take care of the gradient-------*/
  if (grad) {
      PetscPrintf(PETSC_COMM_WORLD,"---Significantly wrong!!! Derivative is not provided! \n");
  }

  count++;
  return l;
}
