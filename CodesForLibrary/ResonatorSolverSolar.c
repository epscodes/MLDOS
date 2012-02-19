#include <petsc.h>
#include <time.h>
#include "Resonator.h"

extern int count;
extern int its;
extern int maxit;
extern int ccount;
extern double cldos;

#undef __FUNCT__ 
#define __FUNCT__ "ResonatorSolverSolar"
double ResonatorSolverSolar(int Mxyz,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;
  
  myfundataSolartype *ptmyfundata = (myfundataSolartype *) data;
  
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

  
  int NJ = ptmyfundata->SNJ;
  int *JRandPos = ptmyfundata->SJRandPos;
 

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
  
 
  // copy epsopt to epsSReal;
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);

  // new Vec, doubles for J*epsjloc*E;
  double epsjloc;
  
  Vec epsCurrent; //epsCurrent is the current real epsilon everywhere; while epsC is calculated in ModifyMatDiagonals but with epsPML; here I need purely real epsilon;
  VecDuplicate(epsC,&epsCurrent); 
  ierr =MatMult(A, epsSReal,epsCurrent); CHKERRQ(ierr); 
  ierr = VecAXPY(epsCurrent,1.0,epsmedium); CHKERRQ(ierr);

  Vec tepsgrad;
  VecDuplicate(epsgrad,&tepsgrad);
  /*-------------------------*/
 
  // Update the diagonals of M Matrix;
  ModifyMatDiagonals(M, A,D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz,omega);


  // I always use LU decompostion;
  PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern, LU is redone! \n");
  ierr = KSPSetOperators(ksp,M,M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  /*-----------------KSP Solving------------------*/ 
  // now loop over all j's

  double tldos, ldos=0;//ldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  int i;
#if 1
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
#endif

  for (i=0; i<NJ; i++)
    {
      VecSet(J,0.0); // initialization (important!)
      VecAssemblyBegin(epsgrad);
      VecAssemblyEnd(epsgrad);
      SourceSingleSetGlobal(PETSC_COMM_WORLD,J, JRandPos[i],1.0/hxyz);	  
      // now b=i*omega*J;
      ierr = MatMult(D,J,b);CHKERRQ(ierr);
      VecScale(b,omega);



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
      // Get the value epsjloc;
      VecSet(tmp,0.0);
      VecSetValue(tmp,JRandPos[i],1.0,INSERT_VALUES);
      VecAssemblyBegin(tmp);
      VecAssemblyEnd(tmp);
      VecDot(tmp,epsCurrent,&epsjloc); 

      // maximize real(E*eps*J) = x*(epsSReal+1)*J; use J, not weightedJ;
      VecDot(x,J,&tldos);
      tldos = -1.0*epsjloc*tldos*hxyz;
      ldos += tldos;

      PetscPrintf(PETSC_COMM_WORLD,"---The current ldos at step %d (current position %d) is %.16e \n", count, i, tldos);
	   
      //char id[100];
      //sprintf(id,"%.4d.m",i);
      //OutputVec(PETSC_COMM_WORLD,x,"linearx",id); 
  
      /*------------------------------------------------*/
      /*-----------Now take care of gradients-------------*/
      /*------------------------------------------------*/

      if (grad) {  
	/* Adjoint-Method tells us Mtran*lambba =J -> x = i*omega/weight*conj(lambda);  therefore the derivative is Re(x^2*weight*i*omega*(1+i/Qabs)*epspml) = Re(x^2*epscoef) ; here, I omit two minus signs: one is M'*lam= -j; the other is -Re(***). minus minus is a plus.*/
	int aconj=0;	
	CmpVecProd(x,epscoef,tmp,D,Nxyz,aconj,tmpa,tmpb);
	CmpVecProd(x,tmp,tepsgrad,D,Nxyz,aconj,tmpa,tmpb);
	// tepsgrad is the old derivate; new derivative = eps_center*tepsgrad + ldos_c;
	// where ldos_c is a zero vector except at one postion = ldos;
	VecScale(tepsgrad,epsjloc*hxyz);

	VecSet(tmp,0.0);
	VecSetValue(tmp,JRandPos[i],1.0,INSERT_VALUES);
	VecAssemblyBegin(tmp);
	VecAssemblyEnd(tmp);
	VecScale(tmp, tldos/epsjloc);
 
	VecAXPY(tepsgrad,1.0,tmp); // tepsgrad(i) = tepsgrad(i) + tldos/epsjloc;
	VecAXPY(epsgrad,1.0,tepsgrad); // epsgrad+=tepsgrad;
	
      }

    }

#if 1
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0)
    PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);
#endif   


  // take the average;
  VecScale(epsgrad,1.0/NJ);
  ldos = ldos/NJ; 


  // set imaginary part of epsgrad = 0; ( we're only interested in real part;
  ierr = VecPointwiseMult(epsgrad,epsgrad,vR); CHKERRQ(ierr);

  // vgrad =A'*epsgrad; A' is the restriction matrix; Mapped to the small grid;
  ierr = MatMultTranspose(A,epsgrad,vgrad);CHKERRQ(ierr);   

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
  ierr=VecDestroy(epsCurrent);CHKERRQ(ierr);
  count++;
  return ldos;
}











