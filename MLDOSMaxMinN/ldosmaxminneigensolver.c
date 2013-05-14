#include <petsc.h>
#include <time.h>
#include "Resonator.h"
#include "maxminnheader.h"

/*global varaible */
extern int Nx, Ny, Nz, Nxyz;
extern double hx, hy, hz, hxyz;
extern Vec epspmlQ, epsmedium, epsC, epsCi, epsP, x, vR, epsSReal, tmp, tmpa, tmpb;
extern Mat M, A, D;
extern char filenameComm[PETSC_MAX_PATH_LEN];

extern KSP ksp;
extern PC pc;

extern int Nj;
extern int sameomega;

//variable needed b, weightedJ, omega,



#undef __FUNCT__ 
#define __FUNCT__ "MaxMinNEigenSolver"
PetscErrorCode MaxMinNEigenSolver(int Linear, int Eig, int maxeigit, void *data)
{
  PetscErrorCode ierr;

  myfundatatypemaxminn *ptmyfundata= (myfundatatypemaxminn *) data;

  int idj = ptmyfundata->Sidj;
  double omega= ptmyfundata->Somega;
  Vec b = ptmyfundata->Sb;
  Vec weightedJ = ptmyfundata->SweightedJ;

  Vec diagB = tmp; // just use tmp's space; Be caution! 
  
  // Update the diagonals of M Matrix;
  Mat Mone;
  MatDuplicate(M,MAT_COPY_VALUES,&Mone); // creat a copy of standar M (curl*muinv*curl);
  VecSet(epsP, 0.0); // set previous espP=0;
  ModifyMatDiagonals(Mone, A,D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz,omega);
  
   
  int its;
  PetscLogDouble t1, t2, tpast;
  int rank;
  
  if (Linear==1)
    {      
      ierr = PetscGetTime(&t1);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,Mone,Mone,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations for linear solver is %d----\n ",its);CHKERRQ(ierr);
      
      ierr = PetscGetTime(&t2);CHKERRQ(ierr);
      tpast = t2 - t1;
     
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if(rank==0)
	PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);	  
      double ldos; //ldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  
      ierr = VecDot(x,weightedJ,&ldos);


  ldos = -1.0*ldos*hxyz;
 
  PetscPrintf(PETSC_COMM_WORLD,"---The los by linear solver is %.16e \n",ldos);

  /*output the vectors !!!!!!!!!!!!!!!!!!!!!!!!!!!*/
  OutputVec(PETSC_COMM_WORLD, x,filenameComm, "linearx.m");
  PetscPrintf(PETSC_COMM_WORLD,"-------------------------------------------------------------- \n");
    }

  if (Eig==1)
    {
 // now epsP is the current epsilon; epspmlQ=epspml*(1+1i/Qabs); I am going to use diagB to represent the diagonal part of Matrix B;
  int aconj=0;
  VecAYPX(diagB,0.0,epsP); // diagB should be just eps*pml*(1+i/Q); 

  int i;  
  for (i=1;i<maxeigit;i++)
    {      
      ierr = PetscGetTime(&t1);CHKERRQ(ierr);

      if(i==1)
	{ 
	  if (Linear==1)
	    {ierr = KSPSetOperators(ksp,Mone,Mone,SAME_PRECONDITIONER);CHKERRQ(ierr); CHKERRQ(ierr);}
	  else
	    { ierr = KSPSetOperators(ksp,Mone,Mone,SAME_NONZERO_PATTERN); CHKERRQ(ierr);}	  
	  VecAYPX(epsC,0.0,b); // copy b to epsC;
	  CmpVecProd(diagB,epsC,b,D,aconj,tmpa,tmpb); // b=B*b;
	  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
	  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %d----\n ",its);CHKERRQ(ierr);
	}
      else
	{
	  // print the current eigenvalues;
	  RayleighQuotient(Mone,diagB, x, b, vR, D, tmpa, tmpb, Nxyz,i);
	  KSPSetOperators(ksp,Mone,Mone,SAME_PRECONDITIONER);CHKERRQ(ierr);
	  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
	  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %d----\n ",its);CHKERRQ(ierr);


           #if 1
	  double norm;
	  Vec xdiff;
	  ierr=VecDuplicate(x,&xdiff);CHKERRQ(ierr);
	  ierr = MatMult(Mone,x, xdiff);CHKERRQ(ierr);
	  ierr = VecAXPY(xdiff,-1.0,b);CHKERRQ(ierr);
	  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
	  //ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"---Norm of error %g, Kryolv Iterations %d----\n ",norm,its);CHKERRQ(ierr);    
	  ierr=VecDestroy(&xdiff);CHKERRQ(ierr);
          #endif

	}

      ierr = PetscGetTime(&t2);CHKERRQ(ierr);
      tpast = t2 - t1;
     
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if(rank==0)
	PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);
    	 
    }
  OutputVec(PETSC_COMM_WORLD, x,filenameComm, "eigx.m");

    }

 PetscFunctionReturn(0);
}

/* same as the one in EigenSolver.c */

#undef __FUNCT__ 
#define __FUNCT__ "RayleighQuotient"
PetscErrorCode RayleighQuotient(Mat M, Vec diagB, Vec x, Vec b, Vec vR, Mat D, Vec tmpa, Vec tmpb, int Nxyz, int i)
{
  PetscErrorCode ierr;
  
  int aconj=0;
  Vec Ax, Bx, Rex, Imx, ReAx, ImAx, ReBx, ImBx;

  ierr = VecDuplicate(x,&Ax);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&Bx);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&Rex);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&Imx);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ReAx);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ImAx);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ReBx);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ImBx);CHKERRQ(ierr);

  // scale x;
  double normx;
  VecNorm(x,NORM_2,&normx);
  PetscPrintf(PETSC_COMM_WORLD,"---- normx is %.16e---- \n",normx);
  VecScale(x,1.0/normx);

  ierr = MatMult(M, x, Ax); CHKERRQ(ierr);  // compute Ax
  CmpVecProd(diagB,x,Bx,D,aconj,tmpa,tmpb); // compute Bx


  ierr=VecPointwiseMult(Rex, vR, x); CHKERRQ(ierr);
  ierr=VecPointwiseMult(ReAx, vR, Ax); CHKERRQ(ierr);
  ierr=VecPointwiseMult(ReBx, vR, Bx); CHKERRQ(ierr);

  MatMult(D, x, tmpa);
  VecScale(tmpa,-1.0);
  VecPointwiseMult(Imx, tmpa, vR); 

  MatMult(D, Ax, tmpa);
  VecScale(tmpa,-1.0);
  VecPointwiseMult(ImAx, tmpa, vR); 

  MatMult(D, Bx, tmpa);
  VecScale(tmpa,-1.0);
  VecPointwiseMult(ImBx, tmpa, vR);
  
  double numr,numi, denr,deni, vala,valb;
  
  VecDot(Rex,ReAx, &vala);
  VecDot(Imx,ImAx, &valb);
  numr = vala - valb;

  VecDot(Rex,ImAx, &vala);
  VecDot(Imx,ReAx, &valb);
  numi = vala + valb;

  
  VecDot(Rex,ReBx, &vala);
  VecDot(Imx,ImBx, &valb);
  denr = vala - valb;

  VecDot(Rex,ImBx, &vala);
  VecDot(Imx,ReBx, &valb);
  deni = vala + valb;
  
  double Rel, Iml, normsqrden;

  normsqrden = pow(denr,2) + pow(deni,2);
  
  Rel = (numr * denr + numi * deni)/normsqrden;
  Iml = (numi * denr - numr * deni)/normsqrden;
  //update b=Bx;
  VecAYPX(b,0.0,Bx);

  PetscPrintf(PETSC_COMM_WORLD,"----The numerator %.16e + 1i*%.16e---- \n", numr, numi);
  PetscPrintf(PETSC_COMM_WORLD,"----The denominator %.16e + 1i*%.16e---- \n", denr, deni);

  PetscPrintf(PETSC_COMM_WORLD,"----The current eigenvalue is %.16e + 1i*%.16e---- at iteratiion %d \n", Rel, Iml, i);

   VecDestroy(&Ax);
   VecDestroy(&Bx);
   VecDestroy(&Rex);
   VecDestroy(&Imx);
   VecDestroy(&ReAx);
   VecDestroy(&ImAx);
   VecDestroy(&ReBx);
   VecDestroy(&ImBx);


   PetscFunctionReturn(0);
   
}
