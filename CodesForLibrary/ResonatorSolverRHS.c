#include <petsc.h>
#include <time.h>
#include "Resonator.h"

extern int count;
extern int its;
extern int maxit;
extern int ccount;
extern double cldos;

#undef __FUNCT__ 
#define __FUNCT__ "ResonatorSolverRHS"
double ResonatorSolverRHS(int Mxyz,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;
  
  myfundataRHStype *ptmyfundata = (myfundataRHStype *) data;
  
  int Nx = ptmyfundata->SNx;
  int Ny = ptmyfundata->SNy;
  int Nz = ptmyfundata->SNz;
  double hx = ptmyfundata->Shx;
  double hy = ptmyfundata->Shy;
  double hz = ptmyfundata->Shz;
  double omega = ptmyfundata->Somega;
  KSP ksp = ptmyfundata->Sksp;
  Vec epspmlQ = ptmyfundata->SepspmlQ;
  Vec epsmedium = ptmyfundata->Sepsmedium;
  Vec epsC = ptmyfundata->SepsC;
  Vec epsCi = ptmyfundata->SepsCi;
  Vec epsP = ptmyfundata->SepsP;
  Vec x = ptmyfundata->Sx;
  Vec J = ptmyfundata->SJ;
  Vec b = ptmyfundata->Sb;
  Vec vR = ptmyfundata->SvR;
  Vec epsSReal = ptmyfundata->SepsSReal;

  Mat A = ptmyfundata->SA;
  Mat D = ptmyfundata->SD;
  Mat M = ptmyfundata->SM;
  
  char *filenameComm = ptmyfundata->SfilenameComm;

  int ncx = ptmyfundata->Sncx;
  int ncy = ptmyfundata->Sncy;
  int ncz = ptmyfundata->Sncz;

  int clx = ptmyfundata->Sclx;
  int cly = ptmyfundata->Scly;
  int clz = ptmyfundata->Sclz;

  int Jdirection = ptmyfundata->SJdirection;



  int Nxyz = Nx*Ny*Nz;
  double hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;
 
  /*---variables for grad -----*/
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

  VecSet(epsgrad,0.0);
  VecAssemblyBegin(epsgrad);
  VecAssemblyEnd(epsgrad);

  Vec tepsgrad;
  VecDuplicate(epsgrad,&tepsgrad);
  /*-------------------------*/
 
  // copy epsopt to epsSReal;
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);
 
  // Update the diagonals of M Matrix;
  ModifyMatDiagonals(M, A,D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz,omega);



  if (its> 15 || count< 15 )
    {
      PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern, LU is redone! \n");
      ierr = KSPSetOperators(ksp,M,M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  else
    {ierr = KSPSetOperators(ksp,M,M,SAME_PRECONDITIONER);CHKERRQ(ierr);}


  /*-----------------KSP Solving------------------*/ 
  // now loop over all j's

  double tldos, ldos=0;//ldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  int ix,iy,iz,nj;

  nj = ncx*ncy*ncz;

  PetscPrintf(PETSC_COMM_WORLD,"the value of nj is %d \n", nj);
  
  for (ix=0; ix<ncx; ix++)
    for (iy=0; iy<ncy;iy++)
      for(iz=0; iz<ncz;iz++)
	{

	  // set the current
	  VecSet(J,0.0); // initialization
	  if (Jdirection == 1)
	    SourceSingleSetX(PETSC_COMM_WORLD, J, Nx, Ny, Nz, clx-1+ix, cly-1+iy,clz-1+iz,1.0/hxyz);
	  else if (Jdirection == 3)
	    SourceSingleSetZ(PETSC_COMM_WORLD, J, Nx, Ny, Nz,clx-1+ix, cly-1+iy,clz-1+iz,1.0/hxyz);
	  else if (Jdirection == 12)
	    {
	      SourceSingleSetX(PETSC_COMM_WORLD, J, Nx, Ny, Nz, clx-1+ix, cly-1+iy,clz-1+iz,1.0/hxyz);
	      SourceSingleSetY(PETSC_COMM_WORLD, J, Nx, Ny, Nz, clx-1+ix, cly-1+iy,clz-1+iz,1.0/hxyz);
	    }	  
	  else 
	    PetscPrintf(PETSC_COMM_WORLD," Please specify correct direction of current: x (1) or z (3)\n "); 
	  
	  // now b=i*omega*J;
	  ierr = MatMult(D,J,b);CHKERRQ(ierr);
	  VecScale(b,omega);

#if 0
	  //clock_t tstart, tend;  int tpast; tstart=clock();  
	  PetscLogDouble t1,t2,tpast;
	  ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif

	  //KSP Solving
	  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
	  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
	  //ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %D----\n ",its);CHKERRQ(ierr);


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
#if 0
	  ierr = PetscTime(&t2);CHKERRQ(ierr);
	  tpast = t2 - t1;

	  int rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	  if(rank==0)
	    PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);
#endif   

	  ierr = VecDot(x,J,&tldos); // replace weightedJ by J; since weigth is always 1.0;
	  tldos = -1.0*tldos*hxyz;
	  ldos += tldos;

	  PetscPrintf(PETSC_COMM_WORLD,"---The current ldos at step %d (current position %d and %d ) is %.16e \n", count, ix,iy, tldos);
	   
	  char id[100];
	  sprintf(id,"%.2d%.2d.m",ix,iy);
	  OutputVec(PETSC_COMM_WORLD,x,"linearx",id); 
  
	  /*------------------------------------------------*/
	  /*-----------Now take care of gradients-------------*/
	  /*------------------------------------------------*/

	  if (grad) {  
	    /* Adjoint-Method tells us Mtran*lambba =J -> x = i*omega/weight*conj(lambda);  therefore the derivative is Re(x^2*weight*i*omega*(1+i/Qabs)*epspml) = Re(x^2*epscoef) ; here, I omit two minus signs: one is M'*lam= -j; the other is -Re(***). minus minus is a plus.*/
	    int aconj=0;
	    CmpVecProd(x,epscoef,tmp,D,aconj,tmpa,tmpb);
	    CmpVecProd(x,tmp,tepsgrad,D,aconj,tmpa,tmpb);
	    VecAXPY(epsgrad,hxyz,tepsgrad); // epsgrad+=tepsgrad average;//the factor hxyz handle both 2D and 3D;
	  }

	}
  // take the average;
  VecScale(epsgrad,1.0/nj);
  ldos = ldos/nj; 


  // set imaginary part of epsgrad = 0; ( we're only interested in real part;
  ierr = VecPointwiseMult(epsgrad,epsgrad,vR); CHKERRQ(ierr);

  // vgrad =A'*epsgrad; A' is the restriction matrix; Mapped to the small grid;
  ierr = MatMultTranspose(A,epsgrad,vgrad);CHKERRQ(ierr);   
OutputVec(PETSC_COMM_WORLD,epsgrad,"epsgrad",".m");
 OutputVec(PETSC_COMM_WORLD,vgrad,"vgrad",".m");

  // copy vgrad (distributed vector) to a regular array grad;
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
  ierr=VecDestroy(tepsgrad); CHKERRQ(ierr);
  
  count++;
  return ldos;
}











