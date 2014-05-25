#include <petsc.h>
#include <time.h>
#include "Resonator.h"

extern int count;
extern int its;
extern int maxit;
extern int ccount;
extern double cldos;

/*global varaible */
extern int Nx, Ny, Nz, Nxyz;
extern double hx, hy, hz, hxyz,omega, Qabs;
extern KSP ksp;
extern Vec epspmlQ, epsmedium, epsC, epsCi, epsP, x, b, weightedJ, vR, epsSReal;
extern Mat A, D, M;
extern char filenameComm[PETSC_MAX_PATH_LEN];

/*global variables for grad*/
extern Vec epscoef, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, tmpc;
extern  IS from, to;
extern VecScatter scatter;
 
/*global variables for pre */
extern int withpre;
extern Vec epsLast, xLast, xWLastSqr, weight;
extern double ldosLast;

#undef __FUNCT__ 
#define __FUNCT__ "ResonatorSolverPre"
double ResonatorSolverPre(int Mxyz,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;
  
  // copy epsopt to epsSReal;
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);
 
  // Update the diagonals of M Matrix;
  ModifyMatDiagonals(M, A,D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz,omega);
   
  #if 1
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = PetscTime(&t1);CHKERRQ(ierr);
  #endif
  /*-----------------KSP Solving------------------*/ 

#if 1 
  if (its> 15 || count< 15 )
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
	ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %D---\n ",its);CHKERRQ(ierr);
     }
  }
#endif


  //Print kspsolving information
  //int its;
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
  ierr = PetscTime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0)
  PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);
#endif   


  double ldos; //ldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  ierr = VecDot(x,weightedJ,&ldos);
  ldos = -1.0*ldos*hxyz;
  PetscPrintf(PETSC_COMM_WORLD,"---The current ldos at step %d is %.16e \n", count,ldos);
  PetscPrintf(PETSC_COMM_WORLD,"-------------------------------------------------------------- \n");

  if (withpre==1)
    {      
      
      ldosLast = ldos; //update ldosLast;
      VecCopy(epsSReal, epsLast); //update epsLast
      VecPointwiseMult(tmp,x,vR); //tmpa stores realpart of x;
      MatMultTranspose(A,tmp, xLast); // xLast store x in a small grid;
      
      /* 
      VecPointwiseMult(tmp,x,x);
      VecPointwiseMult(tmp,tmp,weight);
      VecPointwiseMult(tmp,tmp,vR);
      MatMultTranspose(A,tmp,xWLastSqr); // xWLast store xweight in a small grid;

      */
      // x^2 in complex sense; xWLastsqr=x.^2*weight; furhter need factor(1+1i/Qabs) TODO;
       CmpVecProd(x,x,tmp,D,0,tmpa,tmpb);
      CmpVecProd(tmp,weight,tmpc,D,0,tmpa,tmpb);
      CmpVecScale(tmpc,tmp,1.0,1.0/Qabs,D,tmpa);
      VecPointwiseMult(tmpc,tmp,vR);
      MatMultTranspose(A,tmpc,xWLastSqr);

    }





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
 


  /*-----take care of the gradient-------*/
  if (grad) {
 
   /* Adjoint-Method tells us Mtran*lambba =J -> x = i*omega/weight*conj(lambda);  therefore the derivative is Re(x^2*weight*i*omega*(1+i/Qabs)*epspml) = Re(x^2*epscoef) ; here, I omit two minus signs: one is M'*lam= -j; the other is -Re(***). minus minus is a plus.*/
   int aconj=0;
   CmpVecProd(x,epscoef,tmp,D,aconj,tmpa,tmpb);
   CmpVecProd(x,tmp,epsgrad,D,aconj,tmpa,tmpb);   
   VecScale(epsgrad,hxyz); // the factor hxyz handle both 2D and 3D;

   // set imaginary part of epsgrad = 0; ( we're only interested in real part;
   ierr = VecPointwiseMult(epsgrad,epsgrad,vR); CHKERRQ(ierr);

   // vgrad =A'*epsgrad; A' is the restriction matrix; Mapped to the small grid;
   ierr = MatMultTranspose(A,epsgrad,vgrad);CHKERRQ(ierr);   

// copy vgrad (distributed vector) to a regular array grad;
   ierr = VecToArray(vgrad,grad,scatter,from,to,vgradlocal,Mxyz);
  }

  count++;

  return ldos;
}











