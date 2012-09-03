#include <petsc.h>
#include <time.h>
#include "Resonator.h"
#include <complex.h>

extern int count;
extern int its;
extern int maxit;
extern int ccount;
extern double cldos;

/*global varaible */
extern int Nx, Ny, Nz, Nxyz;
extern double hx, hy, hz, hxyz,omega;
extern KSP ksp;
extern Vec epspmlQ, epsmedium, epsC, epsCi, epsP, x, b, weightedJ, vR, epsSReal;
extern Mat A, D, M;
extern char filenameComm[PETSC_MAX_PATH_LEN];

/*global variables for grad*/
extern Vec epscoef, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb;
extern  IS from, to;
extern VecScatter scatter;

/*global variable for min or max approach */
extern int minapproach;
extern int withepsinldos;
extern Vec pickposvec;
extern int outputbase;
extern double epsair;
extern int refinedldos;
extern int posj;

/* vector needed for lorentzian sqr weight; */
extern int lrzsqr;
extern Vec epsFReal;
extern double sqrtomegaI;
extern Vec nb;
extern Vec y;
extern Vec xsqr;
extern Mat C;
extern double betar;
extern double betai;
extern Vec weight;
extern double Qabs;

#undef __FUNCT__ 
#define __FUNCT__ "ResonatorSolver"
double ResonatorSolver(int DegFree,double *epsopt, double *grad, void *data)
{
  
  PetscErrorCode ierr;
  
  // copy epsopt to epsSReal;
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);

  // Update the diagonals of M Matrix;
  ModifyMatDiagonals(M, A,D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz,omega);
  
  #if 1
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"---Norm of error %g, Kryolv Iterations %d----\n ",norm,its);CHKERRQ(ierr);    
  ierr=VecDestroy(&xdiff);CHKERRQ(ierr);
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


  double ldos, tmpldos; //tmpldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  double complex ctmpldos;
  if (refinedldos)
    {
      double tmpldosreal, tmpldosimag;
      VecDot(x,weightedJ, &tmpldosreal);
      ierr = MatMult(D,x,tmp); CHKERRQ(ierr);
      VecDot(tmp,weightedJ,&tmpldosimag);
      tmpldosimag=-tmpldosimag;
      ctmpldos = (tmpldosreal + tmpldosimag*I);
      tmpldos = 1.0/creal(1.0/ctmpldos);
      PetscPrintf(PETSC_COMM_WORLD,"real part is %.16e and imag part is %.16e;\n computed refined real is %.16e \n", creal(ctmpldos),cimag(ctmpldos),tmpldos);
    }
  else
    { 
      ierr = VecDot(x,weightedJ,&tmpldos);
      
      if(lrzsqr)
	{ 
	  // absorption power is beta*integral(epsFReal*weight*E^2)
	  double abspwr, abspwri; 
	  CmpVecProd(x,x,xsqr,D,0,tmpa,tmpb);	 
	  VecPointwiseMult(tmp,xsqr,epsFReal);
	  VecPointwiseMult(tmp,tmp,weight);
	  CmpVecScale(tmp,tmpa,betar,betai,D,tmpb);

	  // Get real part of absorption power;
	  VecPointwiseMult(tmp,tmpa,vR); 
	  VecSum(tmp,&abspwr);
	  
	  // Get imaginary part of absorption power; not necessary;
	  MatMult(D,tmpa,tmpb);
	  VecPointwiseMult(tmp,tmpb,vR);
	  VecSum(tmp,&abspwri);
	  abspwri=-abspwri;

	  // Output the regular ldos and absorption power;
	  PetscPrintf(PETSC_COMM_WORLD,"---The partial ldos at step %.5d is %.16e \n", count,-tmpldos*hxyz);
	  PetscPrintf(PETSC_COMM_WORLD,"---The absportion power at step %.5d is %.16e + i*%.16e \n", count,-abspwr*hxyz,-abspwri*hxyz);
	  tmpldos = tmpldos - abspwr;
	}
        
    }

  tmpldos = -1.0*tmpldos;

  if(withepsinldos)
    ldos = tmpldos*(epsopt[posj]+epsair)*hxyz;
  else
    ldos = tmpldos*hxyz;

  if(minapproach)
    {
      PetscPrintf(PETSC_COMM_WORLD,"---The current ldos (minapp) at step %.5d is %.16e \n", count,ldos);
      ldos = 1.0/ldos;
      PetscPrintf(PETSC_COMM_WORLD,"---The current invldos (minapp) at step %.5d is %.16e \n", count,ldos);
    }
  else
    PetscPrintf(PETSC_COMM_WORLD,"---The current ldos at step %.5d is %.16e \n", count,ldos);

  PetscPrintf(PETSC_COMM_WORLD,"-------------------------------------------------------------- \n");

 
  /* Now store the epsilon at each step*/
  char buffer [100];

  int STORE=1;    
  if(STORE==1 && (count%outputbase==0))
    {
      sprintf(buffer,"%.5depsSReal.m",count);
      OutputVec(PETSC_COMM_WORLD, epsSReal, filenameComm, buffer);          }

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

    if (lrzsqr==0)
      {
	/* Adjoint-Method tells us Mtran*lambba =J -> x = i*omega/weight*conj(lambda);  therefore the derivative is Re(x^2*weight*i*omega*(1+i/Qabs)*epspml) = Re(x^2*epscoef) ; here, I omit two minus signs: one is M'*lam= -j; the other is -Re(***). minus minus is a plus.*/
	int aconj=0;
	CmpVecProd(x,epscoef,tmp,D,aconj,tmpa,tmpb);
	CmpVecProd(x,tmp,epsgrad,D,aconj,tmpa,tmpb);
      }
    else
      {
	/* Adjoint-Method tells us the gradient is i*omega*epspmlQ*x^2 + beta*x^2 + 2*beta*omega^2*epscoef*[transpose(M)\(epsFreal*weight*x)]*x;  where beta=2*absp*(1+1i/Qabs); absp=imag(sqrt(1+1i/Q)); due to the real representation of complex M, the last term in gradient should be 2*beta*omega^2*epscoef*(C*[transpose(M)\(C*(epsFReal*weight*x))])*x; */
	
	// first term: i*omega*epspmlQ*x^2 = epscoef*x^2;	
	//CmpVecProd(x,x,xsqr,D,0,tmpa,tmpb); xsqr is computed in objective.
	CmpVecProd(xsqr,epscoef,epsgrad,D,0,tmpa,tmpb);	

	// nb = C*x*epsFReal*weight;
       	MatMult(C,x,nb);
	VecPointwiseMult(nb,nb,epsFReal);
	VecPointwiseMult(nb,nb,weight);

	// y = C*transpose(M) \ nb;
	ierr = PetscGetTime(&t1);CHKERRQ(ierr);
	ierr = KSPSolveTranspose(ksp,nb,tmp);CHKERRQ(ierr);
	ierr = PetscGetTime(&t2);CHKERRQ(ierr);
	tpast = t2 - t1;
	if(rank==0)
	  PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving transpose is %f s \n",tpast);
	MatMult(C,tmp,y);

	// tmp = (2*beta*omega^2*y*x*epspmlQ); no weight in expression, since it is included in adjoint method nb;
	CmpVecProd(y,x,tmp,D,0,tmpa,tmpb);
	CmpVecProd(tmp,epspmlQ,y,D,0,tmpa,tmpb);
	CmpVecScale(y,tmp,2*pow(omega,2)*betar,2*pow(omega,2)*betai,D,tmpb);
	//VecPointwiseMult(tmp,tmp,vR);

	// tmpa = beta*weight*x^2
	CmpVecScale(xsqr,tmpa,betar,betai,D,tmpb);
	VecPointwiseMult(tmpa,tmpa,weight);

	// add these three terms together;
	VecAXPBYPCZ(epsgrad,1.0,1.0,1.0,tmp,tmpa);

      }
    

   if (withepsinldos) //epsgrad = epscenter*olddev + ldos(only first component;
     {  
       VecScale(epsgrad,epsopt[posj]+epsair);
       VecAXPY(epsgrad,tmpldos,pickposvec);
     }
      
   if (minapproach && !refinedldos)
     VecScale(epsgrad,-ldos*ldos*hxyz);
   else if (minapproach && refinedldos)
     {
       CmpVecScale(epsgrad,tmp, creal(cpow(1/(ctmpldos*hxyz),2)),cimag(cpow(1/(ctmpldos*hxyz),2)),D,tmpa);
       VecCopy(tmp,epsgrad);
       VecScale(epsgrad,-hxyz);
     }
   else
     VecScale(epsgrad,hxyz);// the factor hxyz handle both 2D and 3D;


   // set imaginary part of epsgrad = 0; ( we're only interested in real part;
   ierr = VecPointwiseMult(epsgrad,epsgrad,vR); CHKERRQ(ierr);

   // vgrad =A'*epsgrad; A' is the restriction matrix; Mapped to the small grid;
   ierr = MatMultTranspose(A,epsgrad,vgrad);CHKERRQ(ierr);   

// copy vgrad (distributed vector) to a regular array grad;
   ierr = VecToArray(vgrad,grad,scatter,from,to,vgradlocal,DegFree);
  }

  count++;

  return ldos;
}











