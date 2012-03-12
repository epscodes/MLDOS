#include <petsc.h>
#include <time.h>
#include "Resonator.h"


extern int count;
extern int its;
extern int maxit;
extern int ccount;
extern double cldos;

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
extern double kxbase, kybase, kzbase;

#undef __FUNCT__ 
#define __FUNCT__ "ResonatorSolverSolar"
double ResonatorSolverSolar(int Mxyz,double *epsopt, double *grad, void *data)
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
	  double blochbc[3]={i*kxstep + kxbase*2*PI,j*kystep + kybase*2*PI,k*kzstep + kzbase*2*PI};
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


/*-------------Compute ldos and epsgrad for a fixed k----------- */

int SolarComputeKernel(Vec epsCurrent, Vec epsOmegasqr, Vec epsOmegasqri, double blochbc[3], double *ptkldos, Vec kepsgrad)
{
  /*Create KSP */
  KSP ksp;
  PC pc; 
  PetscErrorCode ierr;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MAT_SOLVER_MUMPS);CHKERRQ(ierr);
  int maxkspit = 20;
  ierr = KSPSetTolerances(ksp,1e-12,PETSC_DEFAULT,PETSC_DEFAULT,maxkspit);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /*Create Vectors */
  Vec J, b, x, tmp, tmpa, tmpb;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &J);CHKERRQ(ierr);
  ierr=VecDuplicate(J,&b); CHKERRQ(ierr);
  ierr=VecDuplicate(J,&x); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr); 
  ierr=VecDuplicate(J,&tmp);CHKERRQ(ierr);
  ierr=VecDuplicate(J,&tmpa);CHKERRQ(ierr);
  ierr=VecDuplicate(J,&tmpb);CHKERRQ(ierr);

  /*Create Doubles*/
  double epsjloc;  

  /*------Create M Operator ---------------*/
  Mat M;  
  // constrcut M = curl \muinv curl - eps*omega^2 operator based on k;
  MoperatorGeneralBloch(MPI_COMM_WORLD, &M, Nx, Ny, Nz, hx, hy, hz, bx,by, bz, muinv, BCPeriod, blochbc, epsOmegasqr, epsOmegasqri);

  SolarEigenvaluesSolver(M,epsCurrent, epspmlQ, D);
  
  // I always use LU decompostion;
  PetscPrintf(PETSC_COMM_WORLD,"Same nonzero pattern, LU is redone! \n");
  ierr = KSPSetOperators(ksp,M,M,SAME_NONZERO_PATTERN);CHKERRQ(ierr); 

  /*-----------------KSP Solving------------------*/ 
  
  // now loop over all j's

  double tldos;//ldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  *ptkldos = 0;
  int i;
#if 1
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
#endif

  for (i=0; i<NJ; i++)
    {
      VecSet(J,0.0); // initialization (important!)
      VecAssemblyBegin(J);
      VecAssemblyEnd(J);
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

      /*--------------Finish KSP Solving---------------*/
      // Get the value epsjloc;
      VecSet(tmp,0.0);
      VecSetValue(tmp,JRandPos[i],1.0,INSERT_VALUES);
      VecAssemblyBegin(tmp);
      VecAssemblyEnd(tmp);
      VecDot(tmp,epsCurrent,&epsjloc); 

      PetscPrintf(PETSC_COMM_WORLD,"---The epsjloc is %.16e \n", epsjloc);

      // maximize real(E*eps*J) = x*(epsSReal+1)*J; use J, not weightedJ;
      VecDot(x,J,&tldos);
      tldos = -1.0*epsjloc*tldos*hxyz;
      *ptkldos += tldos;

      PetscPrintf(PETSC_COMM_WORLD,"---The current ldos at step %d (current position %d) is %.16e \n", count, i, tldos);
	   
      /*------------------------------------------------*/
      /*-----------Now take care of gradients-------------*/
      /*------------------------------------------------*/
      if (kepsgrad != PETSC_NULL)
	{  
	  Vec tepsgrad;
	  ierr=VecDuplicate(J,&tepsgrad);CHKERRQ(ierr);

	  /* Adjoint-Method tells us Mtran*lambba =J -> x = i*omega/weight*conj(lambda);  therefore the derivative is Re(x^2*weight*i*omega*(1+i/Qabs)*epspml) = Re(x^2*epscoef) ; here, I omit two minus signs: one is M'*lam= -j; the other is -Re(***). minus minus is a plus.*/
	  int aconj=0;	
	  CmpVecProd(x,epscoef,tmp,D,aconj,tmpa,tmpb);
	  CmpVecProd(x,tmp,tepsgrad,D,aconj,tmpa,tmpb);
	  // tepsgrad is the old derivate; new derivative = eps_center*tepsgrad + ldos_c;
	  // where ldos_c is a zero vector except at one postion = ldos;
	  VecScale(tepsgrad,epsjloc*hxyz);

	  VecSet(tmp,0.0);
	  VecSetValue(tmp,JRandPos[i],1.0,INSERT_VALUES);
	  VecAssemblyBegin(tmp);
	  VecAssemblyEnd(tmp);
	  VecScale(tmp, tldos/epsjloc);
 
	  VecAXPY(tepsgrad,1.0,tmp); // tepsgrad(i) = tepsgrad(i) + tldos/epsjloc;
	  VecAXPY(kepsgrad,1.0,tepsgrad); // epsgrad+=tepsgrad;
	 
	  ierr=VecDestroy(tepsgrad);CHKERRQ(ierr);
		
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
  *ptkldos = *ptkldos/NJ;
  if (kepsgrad != NULL)
    VecScale(kepsgrad,1.0/NJ);
 
  PetscPrintf(PETSC_COMM_WORLD," tpkldos is %.16e\n ", *ptkldos);

  //Destroy Stuff;
  ierr=VecDestroy(J);CHKERRQ(ierr);
  ierr=VecDestroy(b);CHKERRQ(ierr);
  ierr=VecDestroy(x); CHKERRQ(ierr);
  ierr=VecDestroy(tmp); CHKERRQ(ierr);
  ierr=VecDestroy(tmpa); CHKERRQ(ierr);
  ierr=VecDestroy(tmpb); CHKERRQ(ierr);
  ierr=MatDestroy(M); CHKERRQ(ierr);
  ierr=KSPDestroy(ksp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "ldossolar"
double ldossolar(int numofvar,double *varopt, double *grad, void *data)
{
  PetscErrorCode ierr;
  Vec epsSReal, epsCurrent; // create compatiable vectors with A.
  ierr = MatGetVecs(A,&epsSReal, &epsCurrent); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr); 
 
  // copy epsopt to epsSReal;
  myfundataSolartype *ptmyfundata = (myfundataSolartype *)data;
  ierr=ArrayToVec(ptmyfundata->Sptepsinput,epsSReal); CHKERRQ(ierr);

  //epsCurrent is the current real epsilon everywhere; while epsC is calculated in ModifyMatDiagonals but with epsPML; here I need purely real epsilon;
  ierr =MatMult(A, epsSReal,epsCurrent); CHKERRQ(ierr); 
  ierr = VecAXPY(epsCurrent,1.0,epsmedium); CHKERRQ(ierr);

  // Compute epsOmegasqr and epsOmegasqri;
  omega = varopt[0]; // now omega is degree of freedom;
  Vec epsOmegasqr, epsOmegasqri;
  VecDuplicate(epsCurrent,&epsOmegasqr);
  VecDuplicate(epsCurrent,&epsOmegasqri);
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
	  double blochbc[3]={i*kxstep+kxbase*2*PI,j*kystep+kybase*2*PI,k*kzstep+kzbase*2*PI};
	  PetscPrintf(PETSC_COMM_WORLD,"Compute value at k-points (%f,%f,%f) \n", blochbc[0], blochbc[1], blochbc[2]);
	  double kldos;
	  SolarComputeKernel(epsCurrent, epsOmegasqr, epsOmegasqri, blochbc, &kldos, PETSC_NULL);
	  ldos += kldos;
	}

  // take the average;
  ldos = ldos * kxyzstep;
  PetscPrintf(PETSC_COMM_WORLD,"---The average ldos at omega  %.16e  is %.16e  (step %d) \n", omega,ldos, count);
  PetscPrintf(PETSC_COMM_WORLD,"-------------------------------------------------------------- \n");

  if (grad)
    {
      PetscPrintf(PETSC_COMM_WORLD,"---Significantly wrong!!! Derivative is not provided! \n");
    }

  /*---Destroy Vectors *----*/
  ierr=VecDestroy(epsSReal);CHKERRQ(ierr);
  ierr=VecDestroy(epsOmegasqr);CHKERRQ(ierr);
  ierr=VecDestroy(epsOmegasqri);CHKERRQ(ierr);
  ierr=VecDestroy(epsCurrent);CHKERRQ(ierr); 
 
  count++;
  return ldos;
}






