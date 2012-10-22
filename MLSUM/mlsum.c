#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "Resonator.h"

int count=1;
int its=100;
int maxit=20;
double cldos=0;//set initial ldos;
int ccount=1;

/*-----------------CreateGlobalVariables ----------------*/
int Nx, Ny, Nz, Nxyz;
double hx, hy, hz, hxyz, omegal, omegau, Qabs;
double *muinv;
Vec epspmlQ, epscoef, epsmedium, epsC, epsCi, epsP, x, b, weightedJ, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, tmpc, weight;
Mat A, D, M;
IS from, to;
char filenameComm[PETSC_MAX_PATH_LEN];
KSP ksp;
VecScatter scatter;
//for Job==3;

double epsair;

// add posj for current location;
int posj;
// add indicator for lorenztian square weight;
// real eps in large grid; only need when lrzsqr=1;
// imag(sqrt(omega*(1+1i/Qabs))

Vec epsFReal;
int Job=4;
int lrzsqr=0;
int newQdef=0;

double blochbc[3]={0,0,0};

int Npmlx, Npmly, Npmlz, BCPeriod, LowerPML;
double sigmax, sigmay, sigmaz, Qabs;
Vec J,epsOmegasqr, epsOmegasqri, epsCurrent, epspml;
int bx[2], by[2], bz[2];

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

   
  /*-------------------------------------------------*/
  int Mx,My,Mz,Mzslab,DegFree, anisotropic;

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

  int Jdirection;
 
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod given");
  PetscOptionsGetInt(PETSC_NULL,"-Jdirection",&Jdirection,&flg);  MyCheckAndOutputInt(flg,Jdirection,"Jdirection","Diapole current direction");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","PML in the lower xyz boundary");
  PetscOptionsGetInt(PETSC_NULL,"-bxl",bx,&flg);  MyCheckAndOutputInt(flg,bx[0],"bxl","BC at x lower");
  PetscOptionsGetInt(PETSC_NULL,"-bxu",bx+1,&flg);  MyCheckAndOutputInt(flg,bx[1],"bxu","BC at x upper");
  PetscOptionsGetInt(PETSC_NULL,"-byl",by,&flg);  MyCheckAndOutputInt(flg,by[0],"byl","BC at y lower");
  PetscOptionsGetInt(PETSC_NULL,"-byu",by+1,&flg);  MyCheckAndOutputInt(flg,by[1],"byu","BC at y upper");
  PetscOptionsGetInt(PETSC_NULL,"-bzl",bz,&flg);  MyCheckAndOutputInt(flg,bz[0],"bzl","BC at z lower");
  PetscOptionsGetInt(PETSC_NULL,"-bzu",bz+1,&flg);  MyCheckAndOutputInt(flg,bz[1],"bzu","BC at z upper");


  double  epssub, RRT;
   
  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  hy = hx;
  hz = hx;
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;  

  PetscOptionsGetReal(PETSC_NULL,"-omegal",&omegal,&flg);  MyCheckAndOutputDouble(flg,omegal,"omegal","omegal");
  PetscOptionsGetReal(PETSC_NULL,"-omegau",&omegau,&flg);  MyCheckAndOutputDouble(flg,omegau,"omegau","omegau");

  int Nomega;
  PetscOptionsGetInt(PETSC_NULL,"-Nomega",&Nomega,&flg);  MyCheckAndOutputDouble(flg,Nomega,"Nomega","Nomega");

  double domega;
  domega = (omegau-omegal)/(Nomega-1);


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

  /*--------------------------------------------------------*/

  /*--------------------------------------------------------*/


  /*---------- Set the current source---------*/
  //Mat D; //ImaginaryIMatrix;
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  
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

  //Vec b; // b= i*omega*J;
  ierr = VecDuplicate(J,&b);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b, "rhs");CHKERRQ(ierr); //only created spaces, but value is  specified in ldoscal routine;


 
   

  /*-------Get the weight vector ------------------*/
  //Vec weight;
  ierr = VecDuplicate(J,&weight); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) weight, "weight");CHKERRQ(ierr);

  if(LowerPML==0)
    GetWeightVec(weight, Nx, Ny,Nz); // new code handles both 3D and 2D;
  else
    VecSet(weight,1.0);

  //Vec weightedJ;
  ierr = VecDuplicate(J,&weightedJ); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJ,J,weight);
  ierr = PetscObjectSetName((PetscObject) weightedJ, "weightedJ");CHKERRQ(ierr);

  //Vec vR;
  ierr = VecDuplicate(J,&vR); CHKERRQ(ierr);
  GetRealPartVec(vR, 6*Nxyz);

   /*---------- Define PML eps vectors: epspml---------- */  
  //Vec epspml; //epspmlQ, epscoef;
  ierr = VecDuplicate(J,&epspml);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epspml,"EpsPMLFull"); CHKERRQ(ierr);

  ierr = VecDuplicate(J,&epspmlQ);CHKERRQ(ierr);
  ierr = VecDuplicate(J,&epscoef);CHKERRQ(ierr); //only created the spaces;
  


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
 
  
  // create the vector for epsOmegasqr and epsOmegasqri;
  VecDuplicate(J,&epsOmegasqr);
  VecDuplicate(J,&epsOmegasqri);

  //Vec epsCurrent; //epsCurrent is the current real epsilon everywhere; while epsC is calculated in ModifyMatDiagonals but with epsPML; here I need purely real epsilon;
  VecDuplicate(J,&epsCurrent); 

  /*--------Setup the KSP variables ---------------*/
  
  //KSP ksp;
  PC pc; 
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERPASTIX);CHKERRQ(ierr);
  ierr = PCSetFromOptions(pc);
  int maxkspit = 20;
  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxkspit);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

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
  // set the dielectric at the center is fixed, and alwyas high
  //epsopt[0]=myub; is defined below near lb and ub;
  for (i=0;i<DegFree;i++)
    { //PetscPrintf(PETSC_COMM_WORLD,"current eps reading is %lf \n",epsopt[i]);
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);


  /*--------------------------------------------------------------*/
  /*----Now based on Command Line, Do the corresponding job----*/
  /*----------------------------------------------------------------*/



  /*--------   convert the epsopt array to epsSReal (if job!=optmization) --------*/
  
  // copy epsilon from file to epsSReal; (different from FindOpt.c, because epsilon is not degree-of-freedoms in computeQ.
  // i) create a array to read file (done above in epsopt); ii) convert the array to epsSReal;
  int ns, ne;
  ierr = VecGetOwnershipRange(epsSReal,&ns,&ne);
  for(i=ns;i<ne;i++)
    { ierr=VecSetValue(epsSReal,i,epsopt[i],INSERT_VALUES); 
      CHKERRQ(ierr); }      
    
  ierr = VecAssemblyBegin(epsSReal); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsSReal);  CHKERRQ(ierr);

  /*----call the routine to calculate the ldos---*/
  
  
  double *data;
  data = (double*) malloc(sizeof(double)*Nomega);
 
  for (i=0; i<Nomega; i++)
    {
      data[i]=ldoscal(omegal+i*domega);
    }

   int rank;
   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
   if (rank==0)
     {
       
       double mysum=0;
       double omegacur=0, omega0=2*PI;
       double gamma = omega0/20;
       for(i=0; i<Nomega; i++)
	 {
	   omegacur=omegal+i*domega;
	   mysum = mysum + data[i]/pow(pow(omegacur-omega0,2)+pow(gamma,2),2); 
	 }
       
       mysum=(pow(gamma,3)/PI*2)*mysum*domega;
       printf("the sum you calculated is %.16e \n",mysum);

       char filenameoutput[PETSC_MAX_PATH_LEN];
       strcpy(filenameoutput,filenameComm);
       strcat(filenameoutput,"data.txt");
       
       ptf = fopen(filenameoutput,"w");
       PetscPrintf(PETSC_COMM_WORLD,"writing to files \n");

       for (i=0;i<Nomega;i++)
	 {
	   fprintf(ptf,"%.16e \n",data[i]);
	 }
       fclose(ptf);
     }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

  //OutputVec(PETSC_COMM_WORLD, x,filenameComm, "x.m");

  /*------------------------------------*/
 

  /* ----------------------Destroy Vecs and Mats----------------------------*/ 

  free(epsopt);
  ierr = VecDestroy(&J); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);
  ierr = VecDestroy(&weight); CHKERRQ(ierr);
  ierr = VecDestroy(&weightedJ); CHKERRQ(ierr);
  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef); CHKERRQ(ierr);
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
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  ISDestroy(&from);
  ISDestroy(&to);

  ierr = VecDestroy(&epsOmegasqr);CHKERRQ(ierr);
  ierr = VecDestroy(&epsOmegasqri);CHKERRQ(ierr);
  ierr = VecDestroy(&epsCurrent);CHKERRQ(ierr);
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




