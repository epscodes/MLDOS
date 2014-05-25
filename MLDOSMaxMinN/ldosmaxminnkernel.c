#include <petsc.h>
#include <time.h>
#include "Resonator.h"
#include <complex.h>
#include "maxminnheader.h"

extern int count;
extern int its;
extern int ccount;
extern double cldos;

/*global varaible */
extern int Nx, Ny, Nz, Nxyz;
extern double hx, hy, hz, hxyz;
extern Vec epspmlQ, epsmedium, epsC, epsCi, epsP, x, vR, epsSReal;
extern Mat M, A, D;
extern char filenameComm[PETSC_MAX_PATH_LEN];

/*global variables for grad*/
extern Vec epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb;
extern  IS from, to;
extern VecScatter scatter;

/*global variable for min or max approach */
extern int outputbase;
extern double epsair;

extern KSP ksp;
extern PC pc;

extern int Nj;
extern int sameomega;

#undef __FUNCT__ 
#define __FUNCT__ "ldosmaxminnkernel"
double ldosmaxminnkernel(int DegFree,double *epsopt, double *grad, void *data)
{
  PetscErrorCode ierr;

  myfundatatypemaxminn *ptmyfundata= (myfundatatypemaxminn *) data;

  int idj = ptmyfundata->Sidj;
  double omega= ptmyfundata->Somega;
  Vec b = ptmyfundata->Sb;
  Vec weightedJ = ptmyfundata->SweightedJ;
  Vec epscoef = ptmyfundata->Sepscoef;
 

  // copy epsopt to epsSReal;
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);

  // Update the diagonals of M Matrix;
  Mat Mone;
  MatDuplicate(M,MAT_COPY_VALUES,&Mone); // creat a copy of standar M (curl*muinv*curl);
  VecSet(epsP, 0.0); // set previous espP=0;
  ModifyMatDiagonals(Mone, A,D, epsSReal, epspmlQ, epsmedium, epsC, epsCi, epsP, Nxyz,omega);

#if 1
  //clock_t tstart, tend;  int tpast; tstart=clock();  
  PetscLogDouble t1,t2,tpast;
  ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif
  /*-----------------KSP Solving------------------*/ 

  //always use LU decomposition;

  if(sameomega && idj>0)
    {ierr = KSPSetOperators(ksp,Mone,Mone,SAME_PRECONDITIONER);CHKERRQ(ierr);}
  else
    {ierr = KSPSetOperators(ksp,Mone,Mone,SAME_NONZERO_PATTERN);CHKERRQ(ierr);}

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations in this step is %D----\n ",its);CHKERRQ(ierr);

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

  /*--------------Finish KSP Solving---------------*/
#if 1
  ierr = PetscTime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;
  PetscPrintf(PETSC_COMM_WORLD,"---The runing time is %f s \n",tpast);
#endif   

  double ldos, tmpldos, tmpldosr, tmpldosi; //tmpldos = -Re((weight.*J)'*E) or -Re(E'*(weight*J));
  ierr = VecDot(x,weightedJ,&tmpldos);
  CmpVecDot(x,weightedJ,&tmpldosr,&tmpldosi,D,vR,tmp,tmpa,tmpb);
  tmpldos = -1.0*tmpldos;
  ldos = tmpldos*hxyz;
  PetscPrintf(PETSC_COMM_WORLD,"---The current ldos at step %.5d with %d-th (directional) frequency %.8e  is      %.8e \n", count, idj+1, omega, ldos);
  PetscPrintf(PETSC_COMM_WORLD,"-------------------------------------------------------------- \n");

 
  /* Now store the epsilon at each step*/
  char buffer [100];

  int STORE=1;    
  if(STORE==1 && (count%outputbase==0))
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
    VecScale(epsgrad,hxyz);// the factor hxyz handle both 2D and 3D;

    // set imaginary part of epsgrad = 0; ( we're only interested in real part;
    ierr = VecPointwiseMult(epsgrad,epsgrad,vR); CHKERRQ(ierr);

    // vgrad =A'*epsgrad; A' is the restriction matrix; Mapped to the small grid;
    ierr = MatMultTranspose(A,epsgrad,vgrad);CHKERRQ(ierr);   

    // copy vgrad (distributed vector) to a regular array grad;
    ierr = VecToArray(vgrad,grad,scatter,from,to,vgradlocal,DegFree);
  }

  if(idj==(Nj-1))
    count++;

  MatDestroy(&Mone);

  return ldos;
}
