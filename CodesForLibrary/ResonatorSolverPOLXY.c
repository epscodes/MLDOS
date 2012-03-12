#include <petsc.h>
#include <time.h>
#include "Resonator.h"

extern int count;
extern int itsx;
extern int itsy;
extern int maxit;
extern int ccount;
extern double cldos;

#undef __FUNCT__ 
#define __FUNCT__ "ResonatorSolverPOLXY"
double ResonatorSolverPOLXY(int Mxyz,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;
  
  myfundataPOLXYtype *ptmyfundata = (myfundataPOLXYtype *) data;
  
  int Nx = ptmyfundata->SNx;
  int Ny = ptmyfundata->SNy;
  int Nz = ptmyfundata->SNz;
  double hx = ptmyfundata->Shx;
  double hy = ptmyfundata->Shy;
  double hz = ptmyfundata->Shz;
  double omega = ptmyfundata->Somega;
  KSP kspx = ptmyfundata->Skspx;
  KSP kspy = ptmyfundata->Skspy;
  Vec epspmlQ = ptmyfundata->SepspmlQ;
  Vec epsmedium = ptmyfundata->Sepsmedium;
  Vec epsC = ptmyfundata->SepsC;
  Vec epsCi = ptmyfundata->SepsCi;
  Vec epsPx = ptmyfundata->SepsPx;
  Vec epsPy = ptmyfundata->SepsPy;
  Vec solx = ptmyfundata->Ssolx;
  Vec soly = ptmyfundata->Ssoly;
  Vec bx = ptmyfundata->Sbx;
  Vec by = ptmyfundata->Sby;
  Vec weightedJx = ptmyfundata->SweightedJx;
  Vec weightedJy = ptmyfundata->SweightedJy;

  Vec vR = ptmyfundata->SvR;
  Vec epsSReal = ptmyfundata->SepsSReal;
  Mat A = ptmyfundata->SA;
  Mat D = ptmyfundata->SD;
  Mat MatX = ptmyfundata->SMatX;
  Mat MatY = ptmyfundata->SMatY;
  
  char *filenameComm = ptmyfundata->SfilenameComm;

  int Nxyz = Nx*Ny*Nz;
  double hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;

  // copy epsopt to epsSReal;
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);
 

  // Update the diagonals of M Matrix;
  ModifyMatDiagonals(MatX, A,D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsPx, Nxyz,omega);

  ModifyMatDiagonals(MatY, A,D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsPy, Nxyz,omega);
  
  #if 1
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
  #endif
  /*-----------------KSP Solving------------------*/ 

#if 1 
  if (itsx> 15 || count< 15 )
    {
      PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern for x-direction, LU is redone! \n");
      ierr = KSPSetOperators(kspx,MatX,MatX,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  else
    {
      ierr = KSPSetOperators(kspx,MatX,MatX,SAME_PRECONDITIONER);CHKERRQ(ierr);
    }

  if (itsy> 15 || count< 15 )
    {
      PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern for y-direction, LU is redone! \n");
      ierr = KSPSetOperators(kspy,MatY,MatY,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  else
    {
      ierr = KSPSetOperators(kspy,MatY,MatY,SAME_PRECONDITIONER);CHKERRQ(ierr);
    }



   ierr = KSPSolve(kspx,bx,solx);CHKERRQ(ierr); 
   ierr = KSPGetIterationNumber(kspx,&itsx);CHKERRQ(ierr);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step (x-direciton) current is %D----\n ",itsx); CHKERRQ(ierr);

  ierr = KSPSolve(kspy,by,soly);CHKERRQ(ierr);  
  ierr = KSPGetIterationNumber(kspy,&itsy);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step (y-direciton) current is %D----\n ",itsy); CHKERRQ(ierr);

#endif

   // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
#if 1
  {
    ierr = KSPGetIterationNumber(kspx,&itsx);CHKERRQ(ierr);
   
    if(itsx>(maxit-2))
      {
	PetscPrintf(PETSC_COMM_WORLD,"Too many iterations (x-direction) needed! Recomputing \n");
	ierr = KSPSetOperators(kspx,MatX,MatX,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSolve(kspx,bx,solx);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(kspx,&itsx);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations (x-direction) in this step is %D---\n ",itsx);CHKERRQ(ierr);
     }
    
    ierr = KSPGetIterationNumber(kspy,&itsy);CHKERRQ(ierr);
    if (itsy>(maxit-2))
      {
	PetscPrintf(PETSC_COMM_WORLD,"Too many iterations (y-direction) needed! Recomputing \n");
	ierr = KSPSetOperators(kspy,MatY,MatY,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSolve(kspy,by,soly);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(kspy,&itsy);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations (y-direction) in this step is %D---\n ",itsy);CHKERRQ(ierr);
     }

  }
#endif


  //Print kspsolving information
  //int its;

#if 0
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


  double ldos, ldosx, ldosy; //ldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  ierr = VecDot(solx,weightedJx,&ldosx);
  ldosx = -1.0*ldosx*hxyz;
  ierr = VecDot(soly,weightedJy,&ldosy);
  ldosy = -1.0*ldosy*hxyz;  
  ldos = ldosx + ldosy;

  PetscPrintf(PETSC_COMM_WORLD,"---The current ldosx at step %d is %.16e \n", count,ldosx);
  PetscPrintf(PETSC_COMM_WORLD,"---The current ldosy at step %d is %.16e \n", count,ldosy);

 PetscPrintf(PETSC_COMM_WORLD,"---The current ldos at step %d is %.16e \n", count,ldos);
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
 



  /*-----take care of the gradient-------*/
  if (grad) {
   Vec epscoef = ptmyfundata->Sepscoef;
   Vec epsgrad = ptmyfundata->Sepsgrad;
   Vec vgrad = ptmyfundata->Svgrad;
   Vec vgradlocal = ptmyfundata->Svgradlocal;
   Vec tmp = ptmyfundata->Stmp;
   Vec tmpa = ptmyfundata->Stmpa;
   Vec tmpb = ptmyfundata->Stmpb;
   VecScatter scatter = ptmyfundata->Sscatter;
   IS from = ptmyfundata->Sfrom;
   IS to = ptmyfundata->Sto;

#if 1
   /* Adjoint-Method tells us Mtran*lambba =J -> x = i*omega/weight*conj(lambda);  therefore the derivative is Re(x^2*weight*i*omega*(1+i/Qabs)*epspml) = Re(x^2*epscoef) ; here, I omit two minus signs: one is M'*lam= -j; the other is -Re(***). minus minus is a plus.*/
   
   Vec epsgradx, epsgrady;
   VecDuplicate(epsgrad,&epsgradx);
   VecDuplicate(epsgrad,&epsgrady);

   int aconj=0;
   CmpVecProd(solx,epscoef,tmp,D,aconj,tmpa,tmpb);
   CmpVecProd(solx,tmp,epsgradx,D,aconj,tmpa,tmpb);   
   VecScale(epsgradx,hxyz); // the factor hxyz handle both 2D and 3D;

   CmpVecProd(soly,epscoef,tmp,D,aconj,tmpa,tmpb);
   CmpVecProd(soly,tmp,epsgrady,D,aconj,tmpa,tmpb);   
   VecScale(epsgrady,hxyz); // the factor hxyz handle both 2D and 3D;

   VecAXPBYPCZ(epsgrad,1.0,1.0,0.0,epsgradx,epsgrady);

   ierr = VecDestroy(epsgradx); CHKERRQ(ierr);
   ierr = VecDestroy(epsgrady); CHKERRQ(ierr);
#endif


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











