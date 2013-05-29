#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include "Resonator.h"
#include "solarheader.h"
//#include <slepc.h>
//#include <slepceps.h>

int count=1;
int its=100;
int maxit=20;
double cldos=0;//set initial ldos;
int ccount=1;
extern int mma_verbose;

int Job;

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
double kxbase, kybase, kzbase;

Mat TMSixToTwo, D2D;
Vec epspmlQ2D, epsmedium2D, vR2D, epscoef2D;
int TMID;
int NeedEig;

/*---------global variables for mpi---------------*/
int commsz, myrank, mygroup, myid, numgroups; 
MPI_Comm comm_group, comm_sum;

/*---------global variables for BoWeight----------*/
int weightappid; // 0 (0), eps (1), stepfun preferring low dielectric (2);
double hw; // half width;

/*some global variables declaration in Eps.c, not really needed. */
int newQdef=0;
int lrzsqr=0;
Vec epsFReal;

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
  
  //PetscOptionsGetReal(PETSC_NULL,"-kxstep",&kxstep,&flg);  MyCheckAndOutputDouble(flg,kxstep,"kxstep","kxstep");
  kxstep = 1.0/(Nx*hx)/nkx;
  kystep = 1.0/(Ny*hy)/nky;
  kzstep = 1.0/(Nz*hz)/nkz;
  kxyzstep = (Nz==1)*kxstep*kystep + (Nz>1)*kxstep*kystep*kzstep;  

  PetscOptionsGetReal(PETSC_NULL,"-kxbase",&kxbase,&flg);  MyCheckAndOutputDouble(flg,kxbase,"kxbase","kxbase");
  PetscOptionsGetReal(PETSC_NULL,"-kybase",&kybase,&flg);  MyCheckAndOutputDouble(flg,kybase,"kybase","kybase");
  PetscOptionsGetReal(PETSC_NULL,"-kzbase",&kzbase,&flg);  MyCheckAndOutputDouble(flg,kzbase,"kzbase","kzbase");

 PetscOptionsGetInt(PETSC_NULL,"-TMID",&TMID,&flg);  MyCheckAndOutputInt(flg,TMID,"TMID","TMID");
 PetscOptionsGetInt(PETSC_NULL,"-NeedEig",&NeedEig,&flg);  MyCheckAndOutputInt(flg,NeedEig,"NeedEig","NeedEig");

 // get information for MPI;
 PetscOptionsGetInt(PETSC_NULL,"-numgroups",&numgroups,&flg);  MyCheckAndOutputInt(flg,numgroups,"numgroups","numgroups");
 // set up MPI slpit;
 mympisetup();
 PetscPrintf(PETSC_COMM_WORLD,"MPI split is set up !\n");

 // get variables for BoWeight;
 PetscOptionsGetInt(PETSC_NULL,"-weightappid",&weightappid,&flg);  MyCheckAndOutputInt(flg,weightappid,"weightappid","weightappid");
 PetscOptionsGetReal(PETSC_NULL,"-hw",&hw,&flg);  MyCheckAndOutputDouble(flg,hw,"hw","hw");

  /*--------------------------------------------------------*/

 if (myrank==0)
   mma_verbose=1;

  /*---------- Set the current source---------*/
  //ImaginaryIMatrix;
  ImagIMat(comm_group, &D,6*Nxyz);

  /*-------Get the weight vector ------------------*/
  Vec weight;

  if (TMID==1)
    {       
      TMprojmat(comm_group, &TMSixToTwo, Nxyz);
      ierr=MatGetVecs(TMSixToTwo,&weight,&epspmlQ2D); CHKERRQ(ierr);
      ierr=VecDuplicate(epspmlQ2D, &epsmedium2D); CHKERRQ(ierr);
      ierr=VecDuplicate(epspmlQ2D, &vR2D); CHKERRQ(ierr);
      ierr=VecDuplicate(epspmlQ2D, &epscoef2D); CHKERRQ(ierr);
      ImagIMat(comm_group, &D2D, 2*Nxyz);
    }
  else
    { ierr = VecCreateMPI(comm_group, PETSC_DECIDE, 6*Nxyz, &weight);CHKERRQ(ierr); }

 

  if(LowerPML==0)
    GetWeightVec(weight, Nx, Ny,Nz); // new code handles both 3D and 2D;
  else
    VecSet(weight,1.0);


  ierr = VecDuplicate(weight,&vR); CHKERRQ(ierr);
  GetRealPartVec(vR, 6*Nxyz);

  /*----------- Define PML muinv vectors  */
 
  Vec muinvpml;
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML); 

  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  int add=0;
  AddMuAbsorption(muinv,muinvpml,Qabs,add);
  ierr = VecDestroy(&muinvpml); CHKERRQ(ierr);  

  /*---------- Define PML eps vectors: epspml---------- */  
  Vec epspml;
  ierr = VecDuplicate(weight,&epspml);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epspml,"EpsPMLFull"); CHKERRQ(ierr);
  EpsPMLFull(comm_group, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

  ierr = VecDuplicate(weight,&epspmlQ);CHKERRQ(ierr);
  ierr = VecDuplicate(weight,&epscoef);CHKERRQ(ierr);
 
  // compute epspmlQ,epscoef;
  EpsCombine(D, weight, epspml, epspmlQ, epscoef, Qabs, omega);

  /*---------Setup the epsmedium vector----------------*/
  ierr = VecDuplicate(weight,&epsmedium); CHKERRQ(ierr);
  GetMediumVec(epsmedium,Nz,Mz,epsair,epssub);


  if (TMID==1)
    {
      ierr=MatMult(TMSixToTwo,epsmedium,epsmedium2D); CHKERRQ(ierr);
      ierr=MatMult(TMSixToTwo,epspmlQ,epspmlQ2D); CHKERRQ(ierr);
      ierr=MatMult(TMSixToTwo,epscoef,epscoef2D); CHKERRQ(ierr);
      ierr=MatMult(TMSixToTwo,vR,vR2D);CHKERRQ(ierr);
     }


  /*--------- Setup the interp matrix -------------------*/  
  //new routine for myinterp;
  if (TMID==1)
    myinterpTM2D(comm_group, &A, Nx,Ny,LowerPML*Npmlx,LowerPML*Npmly, Mx,My);
  else
    myinterp(comm_group, &A, Nx,Ny,Nz, LowerPML*Npmlx,LowerPML*Npmly,LowerPML*Npmlz, Mx,My,Mz,Mzslab, 0); // LoweerPML*Npmlx,..,.., specify where the interp starts; 0 for isotropic;   
 
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
 
  //int Job; set Job to be gloabl variables;
  PetscOptionsGetInt(PETSC_NULL,"-Job",&Job,&flg);  MyCheckAndOutputInt(flg,Job,"Job","The Job indicator you set");

  int numofvar=(Job==1)*Mxyz + (Job==3);

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
    
  
  myfundataSolartype myfundata = {omega, epsopt};

  if (Job==1)
    {
      if (TMID==1)
	nlopt_set_max_objective(opt, ResonatorSolverSolar2D, &myfundata);
      else
	nlopt_set_max_objective(opt, ResonatorSolverSolar, &myfundata);
      result = nlopt_optimize(opt,epsopt,&maxf);
    }
  else if (Job==3)
    {
      varopt = (double *) malloc(numofvar*sizeof(double));
      varopt[0] = omega;
      if (TMID==1)
	nlopt_set_max_objective(opt, ldossolar2D, &myfundata);
      else
	nlopt_set_max_objective(opt, ldossolar, &myfundata);
      result = nlopt_optimize(opt,varopt,&maxf);
      free(varopt);
    }

 
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
  ierr = VecDestroy(&weight); CHKERRQ(ierr);
  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&epspml); CHKERRQ(ierr);
  ierr = VecDestroy(&epspmlQ); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);  
  ierr = MatDestroy(&D); CHKERRQ(ierr);

  ISDestroy(&from);
  ISDestroy(&to);
 
  if (TMID==1)
    {
      ierr = VecDestroy(&vR2D); CHKERRQ(ierr);
      ierr = VecDestroy(&epspmlQ2D); CHKERRQ(ierr);
      ierr = VecDestroy(&epscoef2D); CHKERRQ(ierr);
      ierr = VecDestroy(&epsmedium2D); CHKERRQ(ierr);
      ierr = MatDestroy(&D2D); CHKERRQ(ierr);
      ierr = MatDestroy(&TMSixToTwo); CHKERRQ(ierr);
    }



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




