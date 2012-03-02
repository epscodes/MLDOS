#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include "Resonator.h"

int count=1;
int its=100;
int maxit=20;
double cldos=0;//set initial ldos;
int ccount=1;
extern int mma_verbose;

/*-----------------CreateGlobalVariables ----------------*/
int Nx, Ny, Nz, Nxyz, BCPeriod, NJ, nkx, nky, nkz, nkxyz;
double hx, hy, hz, hxyz, omega, kxstep, kystep, kzstep, kxyzstep;
int bx[2], by[2], bz[2];
double *muinv;
int *JRandPos;
Vec epspmlQ, epsmedium, vR, epscoef;
Mat A, D;
IS from, to;
char filenameComm[PETSC_MAX_PATH_LEN];

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize and Get the parameters from command line ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;
  PetscTruth flg;
  
  mma_verbose=1;

  /*-------------------------------------------------*/
  int Mx,My,Mz,Mzslab, Npmlx,Npmly,Npmlz, Mxyz;

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
  PetscOptionsGetInt(PETSC_NULL,"-NJ",&NJ,&flg);  MyCheckAndOutputInt(flg,NJ,"NJ","NJ");

  Nxyz = Nx*Ny*Nz;
  Mxyz = Mx*My*((Mzslab==0)?Mz:1);

  int LowerPML;
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod given");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","PML in the lower xyz boundary");
  PetscOptionsGetInt(PETSC_NULL,"-bxl",bx,&flg);  MyCheckAndOutputInt(flg,bx[0],"bxl","BC at x lower");
  PetscOptionsGetInt(PETSC_NULL,"-bxu",bx+1,&flg);  MyCheckAndOutputInt(flg,bx[1],"bxu","BC at x upper");
  PetscOptionsGetInt(PETSC_NULL,"-byl",by,&flg);  MyCheckAndOutputInt(flg,by[0],"byl","BC at y lower");
  PetscOptionsGetInt(PETSC_NULL,"-byu",by+1,&flg);  MyCheckAndOutputInt(flg,by[1],"byu","BC at y upper");
  PetscOptionsGetInt(PETSC_NULL,"-bzl",bz,&flg);  MyCheckAndOutputInt(flg,bz[0],"bzl","BC at z lower");
  PetscOptionsGetInt(PETSC_NULL,"-bzu",bz+1,&flg);  MyCheckAndOutputInt(flg,bz[1],"bzu","BC at z upper");


  double Qabs,epsair, epssub, RRT, sigmax, sigmay, sigmaz , prop;
   
  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  hy = hx;
  hz = hx;
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;  

  PetscOptionsGetReal(PETSC_NULL,"-omega",&omega,&flg);  MyCheckAndOutputDouble(flg,omega,"omega","omega");
  PetscOptionsGetReal(PETSC_NULL,"-prop",&prop,&flg);  MyCheckAndOutputDouble(flg,prop,"prop","prop");
  omega=omega*prop;
  PetscPrintf(PETSC_COMM_WORLD,"the effective frequency in computation is %.16e \n",omega);

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

  char initialdata[PETSC_MAX_PATH_LEN], randomcurrentpos[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-initialdata",initialdata,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdata,"initialdata","Inputdata file");
  PetscOptionsGetString(PETSC_NULL,"-randomcurrentpos",randomcurrentpos,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,randomcurrentpos,"randomcurrentpos","Random Current Postions file");
  PetscOptionsGetString(PETSC_NULL,"-filenameComm",filenameComm,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,filenameComm,"filenameComm","Output filenameComm");

  PetscOptionsGetInt(PETSC_NULL,"-nkx",&nkx,&flg);  MyCheckAndOutputInt(flg,nkx,"nkx","nkx");
  PetscOptionsGetInt(PETSC_NULL,"-nky",&nky,&flg);  MyCheckAndOutputInt(flg,nky,"nky","nky");
  PetscOptionsGetInt(PETSC_NULL,"-nkz",&nkz,&flg);  MyCheckAndOutputInt(flg,nkz,"nkz","nkz");
  nkxyz = nkx * nky * nkz;
  
  PetscOptionsGetReal(PETSC_NULL,"-kxstep",&kxstep,&flg);  MyCheckAndOutputDouble(flg,kxstep,"kxstep","kxstep");
  kystep = kxstep;
  kzstep = kxstep;
  kxyzstep = (Nz==1)*kxstep*kystep + (Nz>1)*kxstep*kystep*kzstep;  


  /*--------------------------------------------------------*/


  /*---------- Set the current source---------*/
  //ImaginaryIMatrix;
  ImagIMat(PETSC_COMM_WORLD, &D,Nxyz);

  /*-------Get the weight vector ------------------*/
  Vec weight;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &weight);CHKERRQ(ierr);
  if(LowerPML==0)
    GetWeightVec(weight, Nx, Ny,Nz); // new code handles both 3D and 2D;
  else
    VecSet(weight,1.0);


  ierr = VecDuplicate(weight,&vR); CHKERRQ(ierr);
  GetRealPartVec(vR, Nxyz);

  /*----------- Define PML muinv vectors  */
 
  Vec muinvpml;
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML); 

  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  int add=0;
  AddMuAbsorption(muinv,muinvpml,Qabs,Nxyz,add);
  ierr = VecDestroy(muinvpml); CHKERRQ(ierr);  

  /*---------- Define PML eps vectors: epspml---------- */  
  Vec epspml;
  ierr = VecDuplicate(weight,&epspml);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epspml,"EpsPMLFull"); CHKERRQ(ierr);
  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

  ierr = VecDuplicate(weight,&epspmlQ);CHKERRQ(ierr);
  ierr = VecDuplicate(weight,&epscoef);CHKERRQ(ierr);
 
  // compute epspmlQ,epscoef;
  EpsCombine(D, weight, epspml, epspmlQ, epscoef, Qabs, omega);

  /*--------- Setup the interp matrix -------------------*/  
  //new routine for myinterp;
  myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, LowerPML*Npmlx,LowerPML*Npmly,LowerPML*Npmlz, Mx,My,Mz,Mzslab); // LoweerPML*Npmlx,..,.., specify where the interp starts;    

  /*---------Setup the epsmedium vector----------------*/
  ierr = VecDuplicate(weight,&epsmedium); CHKERRQ(ierr);
  GetMediumVec(epsmedium,Nz,Mz,epsair,epssub);
 
  /*---------Create scatter used in the solver-------------*/
 
  ierr =ISCreateStride(PETSC_COMM_SELF,Mxyz,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,Mxyz,0,1,&to); CHKERRQ(ierr);

  /*-------------Read the input file -------------------------*/

  double *epsopt;
  epsopt = (double *) malloc(Mxyz*sizeof(double));

  FILE *ptf;
  ptf = fopen(initialdata,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");

  int i;
  for (i=0;i<Mxyz;i++)
    { //PetscPrintf(PETSC_COMM_WORLD,"current eps reading is %lf \n",epsopt[i]);
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);

  //int *JRandPos;
  JRandPos = (int *) malloc(NJ*sizeof(int));

  ptf = fopen(randomcurrentpos,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from randomcurrentpos file \n");
  for (i=0;i<NJ;i++)
    {
      fscanf(ptf,"%d",&JRandPos[i]);
    }
  fclose(ptf);



  /*----declare these data types, althought they may not be used for job 2 -----------------*/
 
  double mylb,myub, *varopt, *lb=NULL, *ub=NULL;
  int maxeval, maxtime, mynloptalg;
  double maxf;
  nlopt_opt  opt;
  nlopt_result result;
  /*--------------------------------------------------------------*/
  /*----Now based on Command Line, Do the corresponding job----*/
  /*----------------------------------------------------------------*/

  
  int numofvar=Mxyz;

  /*--------   convert the epsopt array to epsSReal (if job!=optmization) --------*/      
  PetscOptionsGetInt(PETSC_NULL,"-maxeval",&maxeval,&flg);  MyCheckAndOutputInt(flg,maxeval,"maxeval","max number of evaluation");
  PetscOptionsGetInt(PETSC_NULL,"-maxtime",&maxtime,&flg);  MyCheckAndOutputInt(flg,maxtime,"maxtime","max time of evaluation");
  PetscOptionsGetInt(PETSC_NULL,"-mynloptalg",&mynloptalg,&flg);  MyCheckAndOutputInt(flg,mynloptalg,"mynloptalg","The algorithm used ");

  PetscOptionsGetReal(PETSC_NULL,"-mylb",&mylb,&flg);  MyCheckAndOutputDouble(flg,mylb,"mylb","optimization lb");
  PetscOptionsGetReal(PETSC_NULL,"-myub",&myub,&flg);  MyCheckAndOutputDouble(flg,myub,"myub","optimization ub");

      
 
  lb = (double *) malloc(numofvar*sizeof(double));
  ub = (double *) malloc(numofvar*sizeof(double));

  for(i=0;i<numofvar;i++)
    {
      lb[i] = mylb;
      ub[i] = myub;
    }  //initial guess, lower bounds, upper bounds;

  opt = nlopt_create(mynloptalg, numofvar);

  nlopt_set_lower_bounds(opt,lb);
  nlopt_set_upper_bounds(opt,ub);
  nlopt_set_maxeval(opt,maxeval);
  nlopt_set_maxtime(opt,maxtime);
    
  
  myfundataSolartype myfundata = {omega};
  nlopt_set_max_objective(opt, ResonatorSolverSolar, &myfundata);
  result = nlopt_optimize(opt,epsopt,&maxf);
  

 
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
    PetscPrintf(PETSC_COMM_WORLD,"found extremum  %0.16e\n", maxf); 
  }

  PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);

  int rankA;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rankA);

  if(rankA==0)
    {
      ptf = fopen(strcat(filenameComm,"epsopt.txt"),"w");
      for (i=0;i<Mxyz;i++)
	fprintf(ptf,"%0.16e \n",epsopt[i]);
      fclose(ptf);
    }  
	

  nlopt_destroy(opt);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

  /*------------------------------------*/

  /* ----------------------Destroy Vecs and Mats----------------------------*/ 

  free(epsopt);
  free(JRandPos);
  free(lb);
  free(ub);
  free(muinv);
  ierr = VecDestroy(weight); CHKERRQ(ierr);
  ierr = VecDestroy(vR); CHKERRQ(ierr);
  ierr = VecDestroy(epspml); CHKERRQ(ierr);
  ierr = VecDestroy(epspmlQ); CHKERRQ(ierr);
  ierr = VecDestroy(epscoef); CHKERRQ(ierr);
  ierr = VecDestroy(epsmedium); CHKERRQ(ierr);
  ierr = MatDestroy(A); CHKERRQ(ierr);  
  ierr = MatDestroy(D); CHKERRQ(ierr);

  ISDestroy(from);
  ISDestroy(to);
 

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



