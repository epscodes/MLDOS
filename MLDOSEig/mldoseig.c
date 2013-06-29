#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <complex.h>
#include "Resonator.h"
#include <slepc.h>
#include "eigheader.h"

/*-----------------CreateGlobalVariables ----------------*/
int Nx, Ny, Nz, Nxyz;
double hx, hy, hz, hxyz, Qabs;
double *muinv;
Vec epspmlQ, epscoef, epsmedium, epsC, epsCi, epsP, vR, epsSReal, weight;
Mat A, D, M;
IS from, to;
char filenameComm[PETSC_MAX_PATH_LEN];
VecScatter scatter;
double epsair;

/*--redundant variable; just for global variables in eps.c; useless--*/
int lrzsqr=0;
int newQdef=0;
Vec epsFReal;
int Job=1;

/*------------------------------------------------------*/

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize and Get the parameters from command line ------*/
  SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;
  PetscBool flg;
    
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
  
  int BCPeriod, LowerPML;
  int bx[2], by[2], bz[2];
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod given");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","PML in the lower xyz boundary");
  PetscOptionsGetInt(PETSC_NULL,"-bxl",bx,&flg);  MyCheckAndOutputInt(flg,bx[0],"bxl","BC at x lower");
  PetscOptionsGetInt(PETSC_NULL,"-bxu",bx+1,&flg);  MyCheckAndOutputInt(flg,bx[1],"bxu","BC at x upper");
  PetscOptionsGetInt(PETSC_NULL,"-byl",by,&flg);  MyCheckAndOutputInt(flg,by[0],"byl","BC at y lower");
  PetscOptionsGetInt(PETSC_NULL,"-byu",by+1,&flg);  MyCheckAndOutputInt(flg,by[1],"byu","BC at y upper");
  PetscOptionsGetInt(PETSC_NULL,"-bzl",bz,&flg);  MyCheckAndOutputInt(flg,bz[0],"bzl","BC at z lower");
  PetscOptionsGetInt(PETSC_NULL,"-bzu",bz+1,&flg);  MyCheckAndOutputInt(flg,bz[1],"bzu","BC at z upper");


  double  epssub, RRT, sigmax, sigmay, sigmaz;
   
  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  hy = hx;
  hz = hx;
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;  

  double omega;
  PetscOptionsGetReal(PETSC_NULL,"-omega",&omega,&flg);  MyCheckAndOutputDouble(flg,omega,"omega","omega");


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

  /*--------------------------------------------------------*/


  /*---------- Set the current source---------*/
  //Mat D; //ImaginaryIMatrix;
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  Vec J;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &J);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) J, "Source");CHKERRQ(ierr);
  VecSet(J,0.0); //initialization;

  /*-------Get the weight vector ------------------*/
  //Vec weight;
  ierr = VecDuplicate(J,&weight); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) weight, "weight");CHKERRQ(ierr);

  if(LowerPML==0)
    GetWeightVec(weight, Nx, Ny,Nz); // new code handles both 3D and 2D;
  else
    VecSet(weight,1.0);

  //Vec vR;
  ierr = VecDuplicate(J,&vR); CHKERRQ(ierr);
  GetRealPartVec(vR, 6*Nxyz);


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
  ierr = VecDuplicate(J,&epscoef);CHKERRQ(ierr);

  // compute epspmlQ,epscoef;
  EpsCombine(D, weight, epspml, epspmlQ, epscoef, Qabs, omega);
 
  /*--------- Setup the interp matrix ----------------------- */
  /* for a samll eps block, interp it into yee-lattice. The interp matrix A and PML epspml only need to generated once;*/
  
  //Mat A; 
  //new routine for myinterp;
  myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, LowerPML*floor((Nx-Mx)/2),LowerPML*floor((Ny-My)/2),LowerPML*floor((Nz-Mz)/2), Mx,My,Mz,Mzslab, anisotropic); // LoweerPML*Npmlx,..,.., specify where the interp starts;  

  //Vec epsSReal; // create compatiable vectors with A.
  Vec epsCurrent;
  ierr = MatGetVecs(A,&epsSReal, &epsCurrent); CHKERRQ(ierr);  
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsCurrent, "epsCurrent");CHKERRQ(ierr);
  
  /*---------Setup the epsmedium vector----------------*/
  //Vec epsmedium;
  ierr = VecDuplicate(J,&epsmedium); CHKERRQ(ierr);
  GetMediumVec(epsmedium,Nz,Mz,epsair,epssub);

  
 
  /*--------- Setup the finitie difference matrix-------------*/
  //Mat M;
  MoperatorGeneral(PETSC_COMM_WORLD, &M, Nx,Ny,Nz,hx,hy,hz, bx, by, bz,muinv,BCPeriod);
  free(muinv);

  
  
  /*----------- Create the space for final eps -------------*/

  //Vec epsC, epsCi, epsP;
  ierr = VecDuplicate(J,&epsC);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsC, "EpsC");CHKERRQ(ierr);
  ierr = VecDuplicate(J,&epsCi);CHKERRQ(ierr);
  ierr = VecDuplicate(J,&epsP);CHKERRQ(ierr);

  ierr = VecSet(epsP,0.0); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(epsP); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsP); CHKERRQ(ierr); 

  /*------------ Create scatter used in the solver -----------*/
  //VecScatter scatter;
  //IS from, to;
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&to); CHKERRQ(ierr);

  /*-------------Read the input file -------------------------*/

  double *epsopt;
  epsopt = (double *) malloc(DegFree*sizeof(double));

  FILE *ptf;
  ptf = fopen(initialdata,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");

  int i;
  for (i=0;i<DegFree;i++)
    { 
      fscanf(ptf,"%lf",&epsopt[i]);
      //PetscPrintf(PETSC_COMM_WORLD,"current eps reading at %d is %.12e \n",i,epsopt[i]);
    }
  fclose(ptf);


  // copy epsopt to epsSReal;
  ierr=ArrayToVec(epsopt, epsSReal); CHKERRQ(ierr);

  // compute epsCurrent;
  ierr =MatMult(A, epsSReal,epsCurrent); CHKERRQ(ierr);
  ierr = VecAXPY(epsCurrent,1.0,epsmedium); CHKERRQ(ierr);

  /* perform the eigenvalue job */
  ComputeEigFun(epsCurrent,omega);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

  /* ----------------------Destroy Vecs and Mats----------------------------*/ 
  free(epsopt);
  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ); CHKERRQ(ierr);
  ierr = VecDestroy(&epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium); CHKERRQ(ierr);
  ierr = VecDestroy(&epsC); CHKERRQ(ierr);
  ierr = VecDestroy(&epsCi); CHKERRQ(ierr);
  ierr = VecDestroy(&epsP); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef); CHKERRQ(ierr);
  ierr = VecDestroy(&weight); CHKERRQ(ierr);
  ierr = VecDestroy(&J); CHKERRQ(ierr);
  ierr = VecDestroy(&epsCurrent); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);  
  ierr = MatDestroy(&D); CHKERRQ(ierr);
  ierr = MatDestroy(&M); CHKERRQ(ierr);


  ISDestroy(&from);
  ISDestroy(&to);

  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    //if (rank == 0) fgetc(stdin);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = SlepcFinalize(); CHKERRQ(ierr);

  return 0;
}




