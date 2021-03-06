#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "Resonator.h"
#include "shgheader.h"

int count=1;
int its=100;
int maxit=20;
double cldos=0;//set initial ldos;
int ccount=1;
extern int mma_verbose;

int Job;
/*-----------------CreateGlobalVariables ----------------*/
int Nx, Ny, Nz, Nxyz;
double hx, hy, hz, hxyz, Qabs;
double *muinv;
Vec epspmlQ, epsmedium, epsC, epsCi, epsP, b, x, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, tmpc, weight;
Mat A, D, M;
IS from, to;
char filenameComm[PETSC_MAX_PATH_LEN];
VecScatter scatter;
//for Job==3;
double ldoscenter, omegacur=0;
// for maximize ldos or mimimize 1/ldos apporach;
int minapproach;
// for include eps(0,0,0) in the definition of ldos;
int withepsinldos;
Vec pickposvec;
double epsatinterest;
// for output structure every outputbase
int outputbase;
double epsair;
// for slective output
int cavityverbose;
// for refined definition of LDOS to compare with Q/V
int refinedldos;
// add posj for current location;
int posj;
// add indicator for lorenztian square weight;
// real eps in large grid; only need when lrzsqr=1;
// imag(sqrt(omega*(1+1i/Qabs))
int lrzsqr;
Vec epsFReal;
double sqrtomegaI; 
Vec nb;
Vec y;
Vec xsqr;
double betar, betai;
Mat C;
int newQdef;
/*---------*/

/*------------------------------------------------------*/

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize and Get the parameters from command line ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;

  PetscBool flg;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0) 
    mma_verbose=1;
    
  /*-------------------------------------------------*/
  int Mx,My,Mz,Mzslab, Npmlx,Npmly,Npmlz,DegFree, anisotropic;

  PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg);  MyCheckAndOutputInt(flg,Nx,"Nx","Nx");
  PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg);  MyCheckAndOutputInt(flg,Ny,"Ny","Nx");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");
  PetscOptionsGetInt(PETSC_NULL,"-Mx",&Mx,&flg);  MyCheckAndOutputInt(flg,Mx,"Mx","Mx");
  PetscOptionsGetInt(PETSC_NULL,"-My",&My,&flg);  MyCheckAndOutputInt(flg,My,"My","My");
  PetscOptionsGetInt(PETSC_NULL,"-Mz",&Mz,&flg);  MyCheckAndOutputInt(flg,Mz,"Mz","Mz");
  PetscOptionsGetInt(PETSC_NULL,"-Mzslab",&Mzslab,&flg);  MyCheckAndOutputInt(flg,Mzslab,"Mzslab","Mzslab");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlx",&Npmlx,&flg);  MyCheckAndOutputInt(flg,Npmlx,"Npmlx","Npmlx");
  PetscOptionsGetInt(PETSC_NULL,"-Npmly",&Npmly,&flg);  MyCheckAndOutputInt(flg,Npmly,"Npmly","Npmly");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");

  Nxyz = Nx*Ny*Nz;

  // if anisotropic !=0, Degree of Freedom = 3*Mx*My*Mz; else DegFree = Mx*My*Mz;
  PetscOptionsGetInt(PETSC_NULL,"-anisotropic",&anisotropic,&flg);
  if(!flg) anisotropic = 0; // by default, it is isotropc.
  DegFree = (anisotropic ? 3 : 1 )*Mx*My*((Mzslab==0)?Mz:1); 
  PetscPrintf(PETSC_COMM_WORLD," the Degree of Freedoms is %d \n ", DegFree);
  
  int DegFreeAll=DegFree+1;
  PetscPrintf(PETSC_COMM_WORLD," the Degree of Freedoms ALL is %d \n ", DegFreeAll);

  int BCPeriod, Jdirection, Jdirectiontwo, LowerPML;
  int bx[2], by[2], bz[2];
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod given");
  PetscOptionsGetInt(PETSC_NULL,"-Jdirection",&Jdirection,&flg);  MyCheckAndOutputInt(flg,Jdirection,"Jdirection","Diapole current direction");
  PetscOptionsGetInt(PETSC_NULL,"-Jdirectiontwo",&Jdirectiontwo,&flg);  MyCheckAndOutputInt(flg,Jdirectiontwo,"Jdirectiontwo","Diapole current direction for source two");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","PML in the lower xyz boundary");
  PetscOptionsGetInt(PETSC_NULL,"-bxl",bx,&flg);  MyCheckAndOutputInt(flg,bx[0],"bxl","BC at x lower");
  PetscOptionsGetInt(PETSC_NULL,"-bxu",bx+1,&flg);  MyCheckAndOutputInt(flg,bx[1],"bxu","BC at x upper");
  PetscOptionsGetInt(PETSC_NULL,"-byl",by,&flg);  MyCheckAndOutputInt(flg,by[0],"byl","BC at y lower");
  PetscOptionsGetInt(PETSC_NULL,"-byu",by+1,&flg);  MyCheckAndOutputInt(flg,by[1],"byu","BC at y upper");
  PetscOptionsGetInt(PETSC_NULL,"-bzl",bz,&flg);  MyCheckAndOutputInt(flg,bz[0],"bzl","BC at z lower");
  PetscOptionsGetInt(PETSC_NULL,"-bzu",bz+1,&flg);  MyCheckAndOutputInt(flg,bz[1],"bzu","BC at z upper");


  double  epssub, RRT, sigmax, sigmay, sigmaz ;
   
  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  hy = hx;
  hz = hx;
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;  

  double omega, omegaone, omegatwo, wratio;
  PetscOptionsGetReal(PETSC_NULL,"-omega",&omega,&flg);  MyCheckAndOutputDouble(flg,omega,"omega","omega");
   PetscOptionsGetReal(PETSC_NULL,"-wratio",&wratio,&flg);  MyCheckAndOutputDouble(flg,wratio,"wratio","wratio");
  omegaone=omega;
  omegatwo=wratio*omega;
  PetscPrintf(PETSC_COMM_WORLD,"---omegaone is %.16e and omegatwo is %.16e ---\n",omegaone, omegatwo);

  PetscOptionsGetReal(PETSC_NULL,"-Qabs",&Qabs,&flg); 
  if (flg && Qabs>1e+15)
    Qabs=1.0/0.0;
  MyCheckAndOutputDouble(flg,Qabs,"Qabs","Qabs");
  PetscOptionsGetReal(PETSC_NULL,"-epsair",&epsair,&flg);  MyCheckAndOutputDouble(flg,epsair,"epsair","epsair");
  PetscOptionsGetReal(PETSC_NULL,"-epssub",&epssub,&flg);  MyCheckAndOutputDouble(flg,epssub,"epssub","epssub");
  PetscOptionsGetReal(PETSC_NULL,"-RRT",&RRT,&flg);  MyCheckAndOutputDouble(flg,RRT,"RRT","RRT given");
  sigmax = pmlsigma(RRT,Npmlx*hx);
  sigmay = pmlsigma(RRT,Npmly*hy);
  sigmaz = pmlsigma(RRT,Npmlz*hz);  
  PetscPrintf(PETSC_COMM_WORLD,"----sigmax is %.12e \n",sigmax);
  PetscPrintf(PETSC_COMM_WORLD,"----sigmay is %.12e \n",sigmay);
  PetscPrintf(PETSC_COMM_WORLD,"----sigmaz is %.12e \n",sigmaz);

  char initialdata[PETSC_MAX_PATH_LEN]; //filenameComm[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-initialdata",initialdata,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdata,"initialdata","Inputdata file");
  PetscOptionsGetString(PETSC_NULL,"-filenameComm",filenameComm,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,filenameComm,"filenameComm","Output filenameComm");


  // add cx, cy, cz to indicate where the diapole current is;

  int cx, cy, cz;
  PetscOptionsGetInt(PETSC_NULL,"-cx",&cx,&flg); 
  if (!flg)
    {cx=(LowerPML)*floor(Nx/2); PetscPrintf(PETSC_COMM_WORLD,"cx is %d by default \n",cx);}
  else
    {PetscPrintf(PETSC_COMM_WORLD,"the current poisiont cx is %d \n",cx);}
  

  PetscOptionsGetInt(PETSC_NULL,"-cy",&cy,&flg); 
  if (!flg)
    {cy=(LowerPML)*floor(Ny/2); PetscPrintf(PETSC_COMM_WORLD,"cy is %d by default \n",cy);}
 else
    {PetscPrintf(PETSC_COMM_WORLD,"the current poisiont cy is %d \n",cy);}
  

  PetscOptionsGetInt(PETSC_NULL,"-cz",&cz,&flg); 
  if (!flg)
    {cz=(LowerPML)*floor(Nz/2); PetscPrintf(PETSC_COMM_WORLD,"cz is %d by default \n",cz);}
  else
    {PetscPrintf(PETSC_COMM_WORLD,"the current poisiont cz is %d \n",cz);}
    
  posj = (cx*Ny+ cy)*Nz + cz;
  PetscPrintf(PETSC_COMM_WORLD,"the posj is %d \n. ", posj);

  int fixpteps;
  PetscOptionsGetInt(PETSC_NULL,"-fixpteps",&fixpteps,&flg);  MyCheckAndOutputInt(flg,fixpteps,"fixpteps","fixpteps");

  // Get minapproach;
  PetscOptionsGetInt(PETSC_NULL,"-minapproach",&minapproach,&flg);  MyCheckAndOutputInt(flg,minapproach,"minapproach","minapproach");
   
  // Get withepsinldos;
  PetscOptionsGetInt(PETSC_NULL,"-withepsinldos",&withepsinldos,&flg);  MyCheckAndOutputInt(flg,withepsinldos,"withepsinldos","withepsinldos");
  
  // Get outputbase;
  PetscOptionsGetInt(PETSC_NULL,"-outputbase",&outputbase,&flg);  MyCheckAndOutputInt(flg,outputbase,"outputbase","outputbase");
  // Get cavityverbose;
  PetscOptionsGetInt(PETSC_NULL,"-cavityverbose",&cavityverbose,&flg);
  if(!flg) cavityverbose=0;
  PetscPrintf(PETSC_COMM_WORLD,"the cavity verbose is set as %d \n", cavityverbose); 
  // Get refinedldos;
  PetscOptionsGetInt(PETSC_NULL,"-refinedldos",&refinedldos,&flg);
  if(!flg) refinedldos=0;
  PetscPrintf(PETSC_COMM_WORLD,"the refinedldos is set as %d \n", refinedldos);
  // Get cmpwrhs;
  int cmpwrhs;
   PetscOptionsGetInt(PETSC_NULL,"-cmpwrhs",&cmpwrhs,&flg);
  if(!flg) cmpwrhs=0;
  PetscPrintf(PETSC_COMM_WORLD,"the cmpwrhs is set as %d \n", cmpwrhs);
  // Get lrzsqr;
   PetscOptionsGetInt(PETSC_NULL,"-lrzsqr",&lrzsqr,&flg);
  if(!flg) lrzsqr=0;
  PetscPrintf(PETSC_COMM_WORLD,"the lrzsqr is set as %d \n", lrzsqr);
  // Get newQdef;
   PetscOptionsGetInt(PETSC_NULL,"-newQdef",&newQdef,&flg);
  if(!flg) newQdef=0;
  PetscPrintf(PETSC_COMM_WORLD,"the newQdef is set as %d \n", newQdef);
  /*--------------------------------------------------------*/

  /*--------------------------------------------------------*/


  /*---------- Set the current source---------*/
  //Mat D; //ImaginaryIMatrix;
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  Vec J;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &J);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) J, "Source");CHKERRQ(ierr);
  VecSet(J,0.0); //initialization;

  if (Jdirection == 1)
    SourceSingleSetX(PETSC_COMM_WORLD, J, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (Jdirection ==2)
    SourceSingleSetY(PETSC_COMM_WORLD, J, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (Jdirection == 3)
    SourceSingleSetZ(PETSC_COMM_WORLD, J, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else
    PetscPrintf(PETSC_COMM_WORLD," Please specify correct direction of current: x (1) , y (2) or z (3)\n "); 

  Vec Jtwo;
  ierr = VecDuplicate(J, &Jtwo);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Jtwo, "Sourcetwo");CHKERRQ(ierr);
  VecSet(Jtwo,0.0); //initialization;

  if (Jdirectiontwo == 1)
    SourceSingleSetX(PETSC_COMM_WORLD, Jtwo, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (Jdirectiontwo ==2)
    SourceSingleSetY(PETSC_COMM_WORLD, Jtwo, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else if (Jdirectiontwo == 3)
    SourceSingleSetZ(PETSC_COMM_WORLD, Jtwo, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);
  else
    PetscPrintf(PETSC_COMM_WORLD," Please specify correct direction of current two: x (1) , y (2) or z (3)\n "); 


  //Vec b; // b= i*omega*J;
  Vec bone, btwo;

  ierr = VecDuplicate(J,&b);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b, "rhsone");CHKERRQ(ierr);

  ierr = VecDuplicate(J,&bone);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) bone, "rhsone");CHKERRQ(ierr);

  ierr = VecDuplicate(Jtwo,&btwo);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) btwo, "rhstwo");CHKERRQ(ierr);

  if (cmpwrhs==0)
    {
      ierr = MatMult(D,J,b);CHKERRQ(ierr);
      ierr = MatMult(D,Jtwo,btwo);CHKERRQ(ierr);
      
      VecCopy(b,bone);
      VecScale(bone,omegaone);

      VecScale(btwo,omegatwo);

      VecScale(b,omega);      
    }
  else
    {
      double complex cmpiomega;
      cmpiomega = cpow(1+I/Qabs,newQdef+1);
      double sqrtiomegaR = -omega*cimag(csqrt(cmpiomega));
      double sqrtiomegaI = omega*creal(csqrt(cmpiomega));
      PetscPrintf(PETSC_COMM_WORLD,"the real part of sqrt cmpomega is %g and imag sqrt is % g ", sqrtiomegaR, sqrtiomegaI);
      Vec tmpi;
      ierr = VecDuplicate(J,&tmpi);
      VecSet(b,0.0);
      VecSet(tmpi,0.0);
      CmpVecScale(J,b,sqrtiomegaR,sqrtiomegaI,D,tmpi);
      VecDestroy(&tmpi);
    }

  /*-------Get the weight vector ------------------*/
  //Vec weight;
  ierr = VecDuplicate(J,&weight); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) weight, "weight");CHKERRQ(ierr);

  if(LowerPML==0)
    GetWeightVec(weight, Nx, Ny,Nz); // new code handles both 3D and 2D;
  else
    VecSet(weight,1.0);

  Vec weightedJ;
  ierr = VecDuplicate(J,&weightedJ); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJ,J,weight);
  ierr = PetscObjectSetName((PetscObject) weightedJ, "weightedJ");CHKERRQ(ierr);

  Vec weightedJtwo;
  ierr = VecDuplicate(Jtwo,&weightedJtwo); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJtwo,Jtwo,weight);
  ierr = PetscObjectSetName((PetscObject) weightedJtwo, "weightedJtwo");CHKERRQ(ierr);

  //Vec vR;
  ierr = VecDuplicate(J,&vR); CHKERRQ(ierr);
  GetRealPartVec(vR, 6*Nxyz);

  // VecFReal;
  if (lrzsqr)
    { ierr = VecDuplicate(J,&epsFReal); CHKERRQ(ierr); 
      ierr = PetscObjectSetName((PetscObject) epsFReal, "epsFReal");CHKERRQ(ierr);

      if (newQdef==0)
	{
	  sqrtomegaI = omega*cimag(csqrt(1+I/Qabs));
	  PetscPrintf(PETSC_COMM_WORLD,"the real part of sqrt cmpomega is %g and imag sqrt is % g ", omega*creal(csqrt(1+I/Qabs)), sqrtomegaI);
	  betar = 2*sqrtomegaI;
	  betai = betar/Qabs;
	}
      else
	{
	  double gamma;
	  gamma = omega/Qabs;
	  betar = 2*gamma*(1-1.0/pow(Qabs,2));
	  betai = 2*gamma*(2.0/Qabs);
	}

      ierr = VecDuplicate(J,&nb); CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) nb, "nb"); CHKERRQ(ierr);
      
      ierr = VecDuplicate(J,&y); CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) y, "y"); CHKERRQ(ierr);
      
      ierr = VecDuplicate(J,&xsqr); CHKERRQ(ierr); // xsqr = x*x;
      ierr = PetscObjectSetName((PetscObject) xsqr, "xsqr"); CHKERRQ(ierr);
      CongMat(PETSC_COMM_WORLD, &C, 6*Nxyz);
}
  /*----------- Define PML muinv vectors  */
 
  Vec muinvpml;
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML); 

  //double *muinv;
  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  int add=0;
  AddMuAbsorption(muinv,muinvpml,Qabs,add);
  ierr = VecDestroy(&muinvpml); CHKERRQ(ierr);  

  /*---------- Define PML eps vectors: epspml---------- */  
  Vec epspml; //epspmlQ, epscoef;
  ierr = VecDuplicate(J,&epspml);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epspml,"EpsPMLFull"); CHKERRQ(ierr);
  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

  ierr = VecDuplicate(J,&epspmlQ);CHKERRQ(ierr);


  Vec epscoefone, epscoeftwo;
  ierr = VecDuplicate(J,&epscoefone);CHKERRQ(ierr);
  ierr = VecDuplicate(J,&epscoeftwo);CHKERRQ(ierr);
 
  // compute epspmlQ,epscoef;
  EpsCombine(D, weight, epspml, epspmlQ, epscoefone, Qabs, omegaone);
  EpsCombine(D, weight, epspml, epspmlQ, epscoeftwo, Qabs, omegatwo);
  /*--------- Setup the interp matrix ----------------------- */
  /* for a samll eps block, interp it into yee-lattice. The interp matrix A and PML epspml only need to generated once;*/
  

  //Mat A; 
  //new routine for myinterp;
  myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, LowerPML*floor((Nx-Mx)/2),LowerPML*floor((Ny-My)/2),LowerPML*floor((Nz-Mz)/2), Mx,My,Mz,Mzslab, anisotropic); // LoweerPML*Npmlx,..,.., specify where the interp starts;  

  //Vec epsSReal, epsgrad, vgrad; // create compatiable vectors with A.
  ierr = MatGetVecs(A,&epsSReal, &epsgrad); CHKERRQ(ierr);  
  ierr = PetscObjectSetName((PetscObject) epsgrad, "epsgrad");CHKERRQ(ierr);
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vgrad, "vgrad");CHKERRQ(ierr);
  
  /*---------Setup the epsmedium vector----------------*/
  //Vec epsmedium;
  ierr = VecDuplicate(J,&epsmedium); CHKERRQ(ierr);
  GetMediumVec(epsmedium,Nz,Mz,epsair,epssub);
 
  /*--------- Setup the finitie difference matrix-------------*/
  //Mat M;
  MoperatorGeneral(PETSC_COMM_WORLD, &M, Nx,Ny,Nz,hx,hy,hz, bx, by, bz,muinv,BCPeriod);
  free(muinv);

  /*--------Setup the KSP variables ---------------*/
  
  KSP kspone;
  PC pcone; 
  ierr = KSPCreate(PETSC_COMM_WORLD,&kspone);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetType(kspone, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(kspone,&pcone);CHKERRQ(ierr);
  ierr = PCSetType(pcone,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pcone,MATSOLVERPASTIX);CHKERRQ(ierr);
  ierr = PCSetFromOptions(pcone);
  int maxkspit = 20;
  ierr = KSPSetTolerances(kspone,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxkspit);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspone);CHKERRQ(ierr);

  KSP ksptwo;
  PC pctwo;
   ierr = KSPCreate(PETSC_COMM_WORLD,&ksptwo);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetType(ksptwo, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksptwo,&pctwo);CHKERRQ(ierr);
  ierr = PCSetType(pctwo,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pctwo,MATSOLVERPASTIX);CHKERRQ(ierr);
  ierr = PCSetFromOptions(pctwo);
  ierr = KSPSetTolerances(ksptwo,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxkspit);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksptwo);CHKERRQ(ierr);

  /*--------- Create the space for solution vector -------------*/
  //Vec x;
  ierr = VecDuplicate(J,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr); 
  
  /*----------- Create the space for final eps -------------*/

  //Vec epsC, epsCi, epsP;
  ierr = VecDuplicate(J,&epsC);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsC, "EpsC");CHKERRQ(ierr);
  ierr = VecDuplicate(J,&epsCi);CHKERRQ(ierr);
  ierr = VecDuplicate(J,&epsP);CHKERRQ(ierr);

  ierr = VecSet(epsP,0.0); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(epsP); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsP); CHKERRQ(ierr); 

  /*------------ Create space used in the solver ------------*/
  //Vec vgradlocal,tmp, tmpa,tmpb;
  ierr = VecCreateSeq(PETSC_COMM_SELF, DegFree, &vgradlocal); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmp); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmpa); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmpb); CHKERRQ(ierr);
 
  // Vec pickposvec; this vector is zero except that first entry is one;
  if (withepsinldos)
    { ierr = VecDuplicate(J,&pickposvec); CHKERRQ(ierr);
      ierr = VecSet(pickposvec,0.0); CHKERRQ(ierr);
      ierr = VecSetValue(pickposvec,posj+Jdirection*Nxyz,1.0,INSERT_VALUES);
      VecAssemblyBegin(pickposvec);
      VecAssemblyEnd(pickposvec);
    }
  /*------------ Create scatter used in the solver -----------*/
  //VecScatter scatter;
  //IS from, to;
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&to); CHKERRQ(ierr);

  /*-------------Read the input file -------------------------*/

  double *epsoptAll;
  epsoptAll = (double *) malloc(DegFreeAll*sizeof(double));

  FILE *ptf;
  ptf = fopen(initialdata,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");

  int i;
  // set the dielectric at the center is fixed, and alwyas high
  //epsopt[0]=myub; is defined below near lb and ub;
  for (i=0;i<DegFree;i++)
    { //PetscPrintf(PETSC_COMM_WORLD,"current eps reading is %lf \n",epsopt[i]);
      fscanf(ptf,"%lf",&epsoptAll[i]);
    }
  epsoptAll[DegFreeAll-1]=0; //initialize auxiliary variable;
  fclose(ptf);



  /*----declare these data types, althought they may not be used for job 2 -----------------*/
 
  double mylb,myub, *lb=NULL, *ub=NULL;
  int maxeval, maxtime, mynloptalg;
  double maxf;
  nlopt_opt  opt;
  nlopt_result result;
  /*--------------------------------------------------------------*/
  /*----Now based on Command Line, Do the corresponding job----*/
  /*----------------------------------------------------------------*/


  //int Job; set Job to be gloabl variables;
  PetscOptionsGetInt(PETSC_NULL,"-Job",&Job,&flg);  MyCheckAndOutputInt(flg,Job,"Job","The Job indicator you set");
  
  int numofvar=(Job==1)*DegFreeAll + (Job==3);

  /*--------   convert the epsopt array to epsSReal (if job!=optmization) --------*/
  if (Job==2 || Job ==3)
    {
      // copy epsilon from file to epsSReal; (different from FindOpt.c, because epsilon is not degree-of-freedoms in computeQ.
      // i) create a array to read file (done above in epsopt); ii) convert the array to epsSReal;
      int ns, ne;
      ierr = VecGetOwnershipRange(epsSReal,&ns,&ne);
      for(i=ns;i<ne;i++)
	{ ierr=VecSetValue(epsSReal,i,epsoptAll[i],INSERT_VALUES); 
	  CHKERRQ(ierr); }      
      if(withepsinldos)
	{ epsatinterest = epsoptAll[cx*Ny*Nz + cy*Nz + cz]  + epsair;
	  PetscPrintf(PETSC_COMM_WORLD, " the relative permitivity at the point of current is %.16e \n ",epsatinterest);}
      ierr = VecAssemblyBegin(epsSReal); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(epsSReal);  CHKERRQ(ierr);
    }

  if (Job==1 || Job==3)  // optimization bounds setup;
    {      
      PetscOptionsGetInt(PETSC_NULL,"-maxeval",&maxeval,&flg);  MyCheckAndOutputInt(flg,maxeval,"maxeval","max number of evaluation");
      PetscOptionsGetInt(PETSC_NULL,"-maxtime",&maxtime,&flg);  MyCheckAndOutputInt(flg,maxtime,"maxtime","max time of evaluation");
      PetscOptionsGetInt(PETSC_NULL,"-mynloptalg",&mynloptalg,&flg);  MyCheckAndOutputInt(flg,mynloptalg,"mynloptalg","The algorithm used ");

      PetscOptionsGetReal(PETSC_NULL,"-mylb",&mylb,&flg);  MyCheckAndOutputDouble(flg,mylb,"mylb","optimization lb");
      PetscOptionsGetReal(PETSC_NULL,"-myub",&myub,&flg);  MyCheckAndOutputDouble(flg,myub,"myub","optimization ub");

      
 
      lb = (double *) malloc(numofvar*sizeof(double));
      ub = (double *) malloc(numofvar*sizeof(double));

      // the dielectric constant at center is fixed!
      for(i=0;i<numofvar;i++)
	{
	  lb[i] = mylb;
	  ub[i] = myub;
	}  //initial guess, lower bounds, upper bounds;

      // set lower and upper bounds for auxiliary variable;
      lb[numofvar-1]=0;
      ub[numofvar-1]=1.0/0.0;

      //fix the dielectric at the center to be high for topology optimization;
      if (Job==1 && fixpteps==1)
	{
	  epsoptAll[0]=myub;
	  lb[0]=myub;
	  ub[0]=myub;
	}



      opt = nlopt_create(mynloptalg, numofvar);
      
      myfundatatypeshg data[2] = {{omegaone, bone, weightedJ, epscoefone,kspone},{omegatwo, btwo, weightedJtwo, epscoeftwo,ksptwo}};

      nlopt_add_inequality_constraint(opt,ldosconstraint, &data[0], 1e-8);
      nlopt_add_inequality_constraint(opt,ldosconstraint, &data[1], 1e-8);

      nlopt_set_lower_bounds(opt,lb);
      nlopt_set_upper_bounds(opt,ub);
      nlopt_set_maxeval(opt,maxeval);
      nlopt_set_maxtime(opt,maxtime);


      /*add functionality to choose local optimizer; */
      int mynloptlocalalg;
      nlopt_opt local_opt;
      PetscOptionsGetInt(PETSC_NULL,"-mynloptlocalalg",&mynloptlocalalg,&flg);  MyCheckAndOutputInt(flg,mynloptlocalalg,"mynloptlocalalg","The local optimization algorithm used ");
      if (mynloptlocalalg)
	{ 
	  local_opt=nlopt_create(mynloptlocalalg,numofvar);
	  nlopt_set_ftol_rel(local_opt, 1e-14);
	  nlopt_set_maxeval(local_opt,100000);
	  nlopt_set_local_optimizer(opt,local_opt);
	}
    }

  switch (Job)
    {
    case 1:
      {
	if (minapproach)
	  nlopt_set_min_objective(opt,maxminobjfun,NULL);// NULL: no data to be passed because of global variables;
	else
	  nlopt_set_max_objective(opt,maxminobjfun,NULL);

	result = nlopt_optimize(opt,epsoptAll,&maxf);
      }      
      break;
    case 2 :  //AnalyzeStructure
      { 
	int Linear, Eig, maxeigit;
	PetscOptionsGetInt(PETSC_NULL,"-Linear",&Linear,&flg);  MyCheckAndOutputInt(flg,Linear,"Linear","Linear solver indicator");
	PetscOptionsGetInt(PETSC_NULL,"-Eig",&Eig,&flg);  MyCheckAndOutputInt(flg,Eig,"Eig","Eig solver indicator");
	PetscOptionsGetInt(PETSC_NULL,"-maxeigit",&maxeigit,&flg);  MyCheckAndOutputInt(flg,maxeigit,"maxeigit","maximum number of Eig solver iterations is");

	/*----------------------------------*/
	//EigenSolver(Linear, Eig, maxeigit);
	/*----------------------------------*/

	OutputVec(PETSC_COMM_WORLD, weight,filenameComm, "weight.m");
      }
      break;   
    default:
      PetscPrintf(PETSC_COMM_WORLD,"--------Interesting! You're doing nothing!--------\n ");
 }


  if(Job==1 || Job==3)
    {
      /* print the optimization parameters */
#if 0
      double xrel, frel, fabs;
      // double *xabs;
      frel=nlopt_get_ftol_rel(opt);
      fabs=nlopt_get_ftol_abs(opt);
      xrel=nlopt_get_xtol_rel(opt);
      PetscPrintf(PETSC_COMM_WORLD,"nlopt frel is %g \n",frel);
      PetscPrintf(PETSC_COMM_WORLD,"nlopt fabs is %g \n",fabs);
      PetscPrintf(PETSC_COMM_WORLD,"nlopt xrel is %g \n",xrel);
      //nlopt_result nlopt_get_xtol_abs(const nlopt_opt opt, double *tol);
#endif
      /*--------------*/

      if (result < 0) {
	PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);
      }
      else {
	PetscPrintf(PETSC_COMM_WORLD,"found extremum  %0.16e\n", minapproach?1.0/maxf:maxf); 
      }

      PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);


      if(Job==1)
	{ //OutputVec(PETSC_COMM_WORLD, epsopt,filenameComm, "epsopt.m");
	  //OutputVec(PETSC_COMM_WORLD, epsgrad,filenameComm, "epsgrad.m");
	  //OutputVec(PETSC_COMM_WORLD, vgrad,filenameComm, "vgrad.m");
	  //OutputVec(PETSC_COMM_WORLD, x,filenameComm, "x.m");
	  int rankA;
	  MPI_Comm_rank(PETSC_COMM_WORLD, &rankA);

	  if(rankA==0)
	    {
	      ptf = fopen(strcat(filenameComm,"epsopt.txt"),"w");
	      for (i=0;i<DegFree;i++)
		fprintf(ptf,"%0.16e \n",epsoptAll[i]);
	      fclose(ptf);
	      PetscPrintf(PETSC_COMM_WORLD,"the t parameter is %.8e \n",epsoptAll[DegFreeAll-1]);
	    }  
	}

      nlopt_destroy(opt);
    }
     


  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

  /*------------------------------------*/
 

  /* ----------------------Destroy Vecs and Mats----------------------------*/ 

  free(epsoptAll);
  free(lb);
  free(ub);
  ierr = VecDestroy(&J); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);
  ierr = VecDestroy(&weight); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ); CHKERRQ(ierr);
  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ); CHKERRQ(ierr);
  ierr = VecDestroy(&epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsgrad); CHKERRQ(ierr);
  ierr = VecDestroy(&vgrad); CHKERRQ(ierr);  
  ierr = VecDestroy(&epsmedium); CHKERRQ(ierr);
  ierr = VecDestroy(&epsC); CHKERRQ(ierr);
  ierr = VecDestroy(&epsCi); CHKERRQ(ierr);
  ierr = VecDestroy(&epsP); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&vgradlocal);CHKERRQ(ierr);
  ierr = VecDestroy(&tmp); CHKERRQ(ierr);
  ierr = VecDestroy(&tmpa); CHKERRQ(ierr);
  ierr = VecDestroy(&tmpb); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);  
  ierr = MatDestroy(&D); CHKERRQ(ierr);
  ierr = MatDestroy(&M); CHKERRQ(ierr);  
 

  ierr = VecDestroy(&epscoefone); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoeftwo); CHKERRQ(ierr);
  ierr = KSPDestroy(&kspone);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksptwo);CHKERRQ(ierr);

  ISDestroy(&from);
  ISDestroy(&to);

  if (withepsinldos)
    {ierr=VecDestroy(&pickposvec); CHKERRQ(ierr);}

  if (lrzsqr)
    {
      ierr=VecDestroy(&epsFReal); CHKERRQ(ierr);
      ierr=VecDestroy(&xsqr); CHKERRQ(ierr);
      ierr=VecDestroy(&y); CHKERRQ(ierr);
      ierr=VecDestroy(&nb); CHKERRQ(ierr);
      ierr=MatDestroy(&C); CHKERRQ(ierr);
    }

  ierr = VecDestroy(&bone); CHKERRQ(ierr);
  ierr = VecDestroy(&btwo); CHKERRQ(ierr);
  ierr = VecDestroy(&Jtwo); CHKERRQ(ierr);
  

  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    //if (rank == 0) fgetc(stdin);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}




