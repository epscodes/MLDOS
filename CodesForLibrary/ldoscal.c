#include <petsc.h>
#include <math.h>
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

extern double epsair;
extern int posj;

extern double blochbc[3];
extern int Npmlx, Npmly, Npmlz, BCPeriod, LowerPML;
extern double sigmax, sigmay, sigmaz, Qabs;
extern Vec J, weight, epsOmegasqr, epsOmegasqri, epsCurrent, epspml;
extern int bx[2], by[2], bz[2];
extern double *muinv;

/*---------global variables for mpi---------------*/
extern int myid, myrank; 
extern MPI_Comm comm_group;

/* vector needed for lorentzian sqr weight; */
extern double omega0, gamma0, domega;

#undef _FUNCT_
#define _FUNCT_ "fcnlrzsqr"
double fcnlrzsqr(double l,double omegacur)
{
  double lnorm;
  lnorm = l*(pow(gamma0,3)/PI*2)*domega/pow(pow(omegacur-omega0,2)+pow(gamma0,2),2);
  return lnorm;
}


#undef __FUNCT__ 
#define __FUNCT__ "ldoscal"
double ldoscal(double omega)
{
  PetscErrorCode ierr;

  int Nout=5;
  
  // get right handside b; 
  ierr = MatMult(D,J,b);CHKERRQ(ierr);
  VecScale(b,omega);
  
  /*----------- Define PML muinv vectors  */
 
  Vec muinvpml;
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML); 

  //double *muinv;
  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  int add=0;
  AddMuAbsorption(muinv,muinvpml,Qabs,add);
  ierr = VecDestroy(&muinvpml); CHKERRQ(ierr);  

  EpsPMLFull(comm_group, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);
 
  // compute epspmlQ,epscoef;
  EpsCombine(D, weight, epspml, epspmlQ, epscoef, Qabs, omega);

  // compute epsCurrent;
 ierr =MatMult(A, epsSReal,epsCurrent); CHKERRQ(ierr); 
  ierr = VecAXPY(epsCurrent,1.0,epsmedium); CHKERRQ(ierr);

  // Compute epsOmegasqr and epsOmegasqri;
  ierr = VecPointwiseMult(epsOmegasqr, epsCurrent,epspmlQ); CHKERRQ(ierr);
  ierr = VecScale(epsOmegasqr, pow(omega,2)); CHKERRQ(ierr);
  ierr = MatMult(D,epsOmegasqr,epsOmegasqri); CHKERRQ(ierr);


  /*--------- Setup the finitie difference matrix-------------*/
  //Mat M;
  MoperatorGeneralBloch(comm_group, &M, Nx, Ny, Nz, hx, hy, hz, bx,by, bz, muinv, BCPeriod, blochbc, epsOmegasqr, epsOmegasqri);
  free(muinv);


#if 1
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = PetscGetTime(&t1);CHKERRQ(ierr);
#endif
  /*-----------------KSP Solving------------------*/ 

#if 1 
  if (its> 15 || count< 15 )
    {
      if ( count % Nout == 1)
	{
	  PetscPrintf(comm_group,"Same nonzero pattern, LU is redone! \n");
	}
      ierr = KSPSetOperators(ksp,M,M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);}
  else
    {ierr = KSPSetOperators(ksp,M,M,SAME_PRECONDITIONER);CHKERRQ(ierr);}

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  if ( count % Nout ==1)
    {
      ierr = PetscPrintf(comm_group,"--- the number of Kryolv Iterations in this step is %D----\n ",its);CHKERRQ(ierr);
    }

#endif

  // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
#if 1
  {
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    if(its>(maxit-2))
      {
	PetscPrintf(comm_group,"Too many iterations needed! Recomputing \n");
	ierr = KSPSetOperators(ksp,M,M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
	if (count % Nout == 1)
	  {
	    ierr = PetscPrintf(comm_group,"--- the number of Kryolv Iterations in this step is %D---\n ",its);CHKERRQ(ierr);
	  }
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
  if (count % Nout == 1)
    {
      ierr = PetscPrintf(comm_group,"---Norm of error %g, Kryolv Iterations %d----\n ",norm,its);CHKERRQ(ierr);
    }
  ierr=VecDestroy(&xdiff);CHKERRQ(ierr);
#endif

  /*--------------Finish KSP Solving---------------*/
#if 1
  ierr = PetscGetTime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  if(myid==0)
    if (count % Nout ==1)
      {PetscPrintf(PETSC_COMM_SELF,"---The runing time is %f s \n",tpast);}
#endif   


  double ldos, tmpldos; //tmpldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  ierr = VecDot(x,weightedJ,&tmpldos);
  tmpldos = -1.0*tmpldos;
  ldos = tmpldos*hxyz;
  PetscPrintf(comm_group,"---The current ldos at step %.5d with omega %.16e is %.16e, reported by myrank %d.\n", count, omega, ldos, myrank);
  PetscPrintf(comm_group,"---The normalized ldos with omega %.16e is %.16e \n",omega, fcnlrzsqr(ldos,omega));
  //PetscPrintf(comm_group,"-------------------------------------------------------------- \n");

 
  count++;

  return ldos;
}











