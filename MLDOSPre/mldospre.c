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
int CCSA=41;
extern int mma_verbose, ccsa_verbose;

int Job;

/*-----------------CreateGlobalVariables ----------------*/
int Nx, Ny, Nz, Nxyz;
double hx, hy, hz, hxyz, omega, Qabs;
double *muinv;
Vec epspmlQ, epscoef, epsmedium, epsC, epsCi, epsP, x, b, weightedJ, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, tmpc, weight;
Mat A, D, M;
IS from, to;
char filenameComm[PETSC_MAX_PATH_LEN];
KSP ksp;
VecScatter scatter;
/*--------------------------------------------------------*/
// variables for Pre;
int withpre, PreInitialized, countPre=1, PreVerbose=1;
Vec epsPre, epsLast, xPre, xLast, xWPreSqr, xWLastSqr, vscratch, vecdeps;
double ldosLast, ldosPre;
/*--------------------------------------------------*/

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize and Get the parameters from command line ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;

  PetscTruth flg;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0)   
    { ccsa_verbose=1; 
      mma_verbose=1;
    }
  /*-------------------------------------------------*/
  int Mx,My,Mz,Mzslab, Npmlx,Npmly,Npmlz,Mxyz;

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
  Mxyz = Mx*My*((Mzslab==0)?Mz:1);

  int BCPeriod, Jdirection, LowerPML;
  int bx[2], by[2], bz[2];
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod given");
  PetscOptionsGetInt(PETSC_NULL,"-Jdirection",&Jdirection,&flg);  MyCheckAndOutputInt(flg,Jdirection,"Jdirection","Diapole current direction");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","PML in the lower xyz boundary");
  PetscOptionsGetInt(PETSC_NULL,"-bxl",bx,&flg);  MyCheckAndOutputInt(flg,bx[0],"bxl","BC at x lower");
  PetscOptionsGetInt(PETSC_NULL,"-bxu",bx+1,&flg);  MyCheckAndOutputInt(flg,bx[1],"bxu","BC at x upper");
  PetscOptionsGetInt(PETSC_NULL,"-byl",by,&flg);  MyCheckAndOutputInt(flg,by[0],"byl","BC at y lower");
  PetscOptionsGetInt(PETSC_NULL,"-byu",by+1,&flg);  MyCheckAndOutputInt(flg,by[1],"byu","BC at y upper");
  PetscOptionsGetInt(PETSC_NULL,"-bzl",bz,&flg);  MyCheckAndOutputInt(flg,bz[0],"bzl","BC at z lower");
  PetscOptionsGetInt(PETSC_NULL,"-bzu",bz+1,&flg);  MyCheckAndOutputInt(flg,bz[1],"bzu","BC at z upper");


  double epsair, epssub, RRT, sigmax, sigmay, sigmaz ;
   
  PetscOptionsGetReal(PETSC_NULL,"-hx",&hx,&flg);  MyCheckAndOutputDouble(flg,hx,"hx","hx");
  hy = hx;
  hz = hx;
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;  

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

  char initialdata[PETSC_MAX_PATH_LEN];
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

  int fixpteps;
  PetscOptionsGetInt(PETSC_NULL,"-fixpteps",&fixpteps,&flg);  MyCheckAndOutputInt(flg,fixpteps,"fixpteps","fixpteps");
  
  //get withpre
  PetscOptionsGetInt(PETSC_NULL,"-withpre",&withpre,&flg);  MyCheckAndOutputInt(flg,withpre,"withpre","withpre");

  int mynloptlocalalg;
  if (withpre)
    {
      PetscOptionsGetInt(PETSC_NULL,"-mynloptlocalalg",&mynloptlocalalg,&flg);  MyCheckAndOutputInt(flg,mynloptlocalalg,"mynloptlocalalg","The local optimization algorithm used ");
    }
    
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

  //Vec b; // b= i*omega*J;
  ierr = VecDuplicate(J,&b);CHKERRQ(ierr);
  ierr = MatMult(D,J,b);CHKERRQ(ierr);
  VecScale(b,omega);

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

  //Vec vR;
  ierr = VecDuplicate(J,&vR); CHKERRQ(ierr);
  GetRealPartVec(vR, 6*Nxyz);

  /*----------- Define PML muinv vectors  */
 
  Vec muinvpml;
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML); 

  double *muinv;
  muinv = (double *) malloc(sizeof(double)*6*Nxyz);
  int add=0;
  AddMuAbsorption(muinv,muinvpml,Qabs,add);
  ierr = VecDestroy(muinvpml); CHKERRQ(ierr);  

  /*---------- Define PML eps vectors: epspml---------- */  
  Vec epspml; // epspmlQ;
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
  myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, LowerPML*Npmlx,LowerPML*Npmly,LowerPML*Npmlz, Mx,My,Mz,Mzslab); // LoweerPML*Npmlx,..,.., specify where the interp starts;  

  //Vec epsSReal, epsgrad, vgrad; // create compatiable vectors with A.
  ierr = MatGetVecs(A,&epsSReal, &epsgrad); CHKERRQ(ierr);  
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
  
  //KSP ksp;
  PC pc; 
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MAT_SOLVER_PASTIX);CHKERRQ(ierr);
  int maxkspit = 20;
  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxkspit);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /*--------- Create the space for solution vector -------------*/
  //Vec x;
  ierr = VecDuplicate(J,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr); 
  
  Vec cglambda;
  ierr = VecDuplicate(J,&cglambda);CHKERRQ(ierr);
  
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
  ierr = VecCreateSeq(PETSC_COMM_SELF, Mxyz, &vgradlocal); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmp); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmpa); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmpb); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmpc); CHKERRQ(ierr);
 
  /*------------ Create scatter used in the solver -----------*/
  //VecScatter scatter;
  //IS from, to;
  ierr =ISCreateStride(PETSC_COMM_SELF,Mxyz,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,Mxyz,0,1,&to); CHKERRQ(ierr);

  /*-------------Read the input file -------------------------*/

  double *epsopt;
  epsopt = (double *) malloc(Mxyz*sizeof(double));

  FILE *ptf;
  ptf = fopen(initialdata,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");

  int i;
  // set the dielectric at the center is fixed, and alwyas high
  //epsopt[0]=myub; is defined below near lb and ub;
  for (i=0;i<Mxyz;i++)
    { //PetscPrintf(PETSC_COMM_WORLD,"current eps reading is %lf \n",epsopt[i]);
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);


  /*----declare these data types, althought they may not be used for job 2 -----------------*/
 
  double mylb,myub, *varopt, *lb=NULL, *ub=NULL;
  int maxeval, maxtime, mynloptalg;
  double maxf;
  nlopt_opt  opt, local_opt;
  nlopt_result result;
  /*--------------------------------------------------------------*/
  /*----Now based on Command Line, Do the corresponding job----*/
  /*----------------------------------------------------------------*/


  //int Job; set Job to be gloabl variables;
  PetscOptionsGetInt(PETSC_NULL,"-Job",&Job,&flg);  MyCheckAndOutputInt(flg,Job,"Job","The Job indicator you set");
  
  int numofvar=(Job==1)*Mxyz + (Job==3);

  /*--------   convert the epsopt array to epsSReal (if job!=optmization) --------*/
  if (Job==2 || Job ==3)
    {
      // copy epsilon from file to epsSReal; (different from FindOpt.c, because epsilon is not degree-of-freedoms in computeQ.
      // i) create a array to read file (done above in epsopt); ii) convert the array to epsSReal;
      int ns, ne;
      ierr = VecGetOwnershipRange(epsSReal,&ns,&ne);
      for(i=ns;i<ne;i++)
	{ ierr=VecSetValue(epsSReal,i,epsopt[i],INSERT_VALUES); 
	  CHKERRQ(ierr); }      
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

      //fix the dielectric at the center to be high for topology optimization;
      if (Job==1 && fixpteps==1)
	{
	  epsopt[0]=myub;
	  lb[0]=myub;
	  ub[0]=myub;
	}


 /*-------create spaces for pre--------------*/
      if (Job==1 && withpre==1)
	{
	  if(mynloptalg==CCSA)
	    {
	      ierr=VecDuplicate(epsSReal,&epsPre); CHKERRQ(ierr);
	      ierr=VecDuplicate(epsSReal,&epsLast); CHKERRQ(ierr);
	      ierr=VecDuplicate(epsSReal,&xPre); CHKERRQ(ierr); 
	      ierr=VecDuplicate(epsSReal,&xLast); CHKERRQ(ierr);
	      ierr=VecDuplicate(epsSReal,&xWPreSqr);CHKERRQ(ierr);
	      ierr=VecDuplicate(epsSReal,&xWLastSqr);CHKERRQ(ierr);
	      ierr=VecDuplicate(epsSReal,&vscratch);CHKERRQ(ierr);
	      ierr=VecDuplicate(epsSReal,&vecdeps);CHKERRQ(ierr);
	    }
	  else
	    PetscPrintf(PETSC_COMM_WORLD,"the nlopt algorithm you provided do not support preconditioning! ignoring withpre option \n");
	}


      opt = nlopt_create(mynloptalg, numofvar);
      nlopt_set_lower_bounds(opt,lb);
      nlopt_set_upper_bounds(opt,ub);
      nlopt_set_maxeval(opt,maxeval);
      nlopt_set_maxtime(opt,maxtime);

      if(Job==1 && withpre==1 && mynloptalg==CCSA)
	{
	  PetscPrintf(PETSC_COMM_WORLD,"using the precondition version \n");
	  nlopt_set_precond_max_objective(opt, ResonatorSolverPre, prefun, NULL);
	  if (mynloptlocalalg)
	    { 
	      local_opt=nlopt_create(mynloptlocalalg,Mxyz);
	      nlopt_set_ftol_rel(local_opt, 1e-14);
	      nlopt_set_maxeval(local_opt,100000);
	      nlopt_set_local_optimizer(opt,local_opt);
	    }
	}
      else if (Job==1 && !withpre)
	nlopt_set_max_objective(opt,ResonatorSolverPre,NULL);
      
    }




  switch (Job)
    {
    case 1:
      {
	//myfundatatype myfundata = {Nx, Ny, Nz, hx, hy, hz, omega, ksp, epspmlQ, epscoef, epsmedium, epsC, epsCi, epsP, x, cglambda, b, weightedJ, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, A, D, M, from, to, scatter, filenameComm};
	//nlopt_set_max_objective(opt, ResonatorSolverPre, NULL);
      result = nlopt_optimize(opt,epsopt,&maxf);
      }      
      break;
    case 2 :  //AnalyzeStructure
      { 
	int Linear, Eig, maxeigit;
	PetscOptionsGetInt(PETSC_NULL,"-Linear",&Linear,&flg);  MyCheckAndOutputInt(flg,Linear,"Linear","Linear solver indicator");
	PetscOptionsGetInt(PETSC_NULL,"-Eig",&Eig,&flg);  MyCheckAndOutputInt(flg,Eig,"Eig","Eig solver indicator");
	PetscOptionsGetInt(PETSC_NULL,"-maxeigit",&maxeigit,&flg);  MyCheckAndOutputInt(flg,maxeigit,"maxeigit","maximum number of Eig solver iterations is");


	//myfundatatype myfundata = {Nx, Ny, Nz, hx, hy, hz, omega, ksp, epspmlQ, epscoef, epsmedium, epsC, epsCi, epsP, x, cglambda, b, weightedJ, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, A, D, M, from, to, scatter, filenameComm};

	/*----------------------------------*/
	EigenSolver(Linear, Eig, maxeigit);
	/*----------------------------------*/

	OutputVec(PETSC_COMM_WORLD, weight,filenameComm, "weight.m");
      }
      break;   

    case 3: //computeQ
      {
	/* Job 3 do nothing;
	int optmax;
	PetscOptionsGetInt(PETSC_NULL,"-optmax",&optmax,&flg);  MyCheckAndOutputInt(flg,optmax,"optmax","Indicator for max ComputeQ is ");

	double ldoscenter;
	PetscOptionsGetReal(PETSC_NULL,"-ldoscenter",&ldoscenter,&flg);  MyCheckAndOutputDouble(flg,ldoscenter,"ldoscenter","ldoscenter");

	// use spartptdouble to store current omega;
	double spareptdouble[] = {0}; //initialization;
	myfundatatypeq myfundataq = {Nx, Ny, Nz, hx, hy, hz, omega, ldoscenter, spareptdouble, ksp, epspmlQ, epscoef, epsC, epsCi, epsP, x, cglambda, b, weightedJ, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, A, D, M, from, to, scatter, filenameComm};
       
	varopt = (double *) malloc(numofvar*sizeof(double));
	varopt[0] = omega;
      
	if(optmax==1)
	  nlopt_set_max_objective(opt, ldos, &myfundataq);
	else
	  nlopt_set_min_objective(opt, ldosdiff, &myfundataq);
      
	result = nlopt_optimize(opt,varopt,&maxf);
	PetscPrintf(PETSC_COMM_WORLD,"--the frequency intersted is %0.16e------ \n",*varopt);
	free(varopt);
	*/
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
	PetscPrintf(PETSC_COMM_WORLD,"found extremum  %0.16e\n", maxf); 
      }

      PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);


      if(Job==1)
	{ //OutputVec(PETSC_COMM_WORLD, epsopt,filenameComm, "epsopt.m");
	   OutputVec(PETSC_COMM_WORLD, epsSReal,filenameComm, "epsSReal.m");
 
	   //OutputVec(PETSC_COMM_WORLD,x,filenameComm,"x.m");
	   //OutputVec(PETSC_COMM_WORLD,epsgrad,filenameComm,"grad.m");
	  int rankA;
	  MPI_Comm_rank(PETSC_COMM_WORLD, &rankA);

	  if(rankA==0)
	    {
	      ptf = fopen(strcat(filenameComm,"epsopt.txt"),"w");
	      for (i=0;i<Mxyz;i++)
		fprintf(ptf,"%0.16e \n",epsopt[i]);
	      fclose(ptf);
	    }  
	}

      nlopt_destroy(opt);
    }
     


  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

  /*------------------------------------*/
 

  /* ----------------------Destroy Vecs and Mats----------------------------*/ 

  free(epsopt);
  free(lb);
  free(ub);
  ierr = VecDestroy(J); CHKERRQ(ierr);
  ierr = VecDestroy(b); CHKERRQ(ierr);
  ierr = VecDestroy(weight); CHKERRQ(ierr);
  ierr = VecDestroy(weightedJ); CHKERRQ(ierr);
  ierr = VecDestroy(vR); CHKERRQ(ierr);
  ierr = VecDestroy(epspml); CHKERRQ(ierr);
  ierr = VecDestroy(epspmlQ); CHKERRQ(ierr);
  ierr = VecDestroy(epscoef); CHKERRQ(ierr);
  ierr = VecDestroy(epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(epsgrad); CHKERRQ(ierr);
  ierr = VecDestroy(vgrad); CHKERRQ(ierr);  
  ierr = VecDestroy(epsmedium); CHKERRQ(ierr);
  ierr = VecDestroy(epsC); CHKERRQ(ierr);
  ierr = VecDestroy(epsCi); CHKERRQ(ierr);
  ierr = VecDestroy(epsP); CHKERRQ(ierr);
  ierr = VecDestroy(x); CHKERRQ(ierr);
  ierr = VecDestroy(cglambda);CHKERRQ(ierr);
  ierr = VecDestroy(vgradlocal);CHKERRQ(ierr);
  ierr = VecDestroy(tmp); CHKERRQ(ierr);
  ierr = VecDestroy(tmpa); CHKERRQ(ierr);
  ierr = VecDestroy(tmpb); CHKERRQ(ierr);
  ierr = MatDestroy(A); CHKERRQ(ierr);  
  ierr = MatDestroy(D); CHKERRQ(ierr);
  ierr = MatDestroy(M); CHKERRQ(ierr);  
  ierr = KSPDestroy(ksp);CHKERRQ(ierr);

  ISDestroy(from);
  ISDestroy(to);

  if(withpre==1)
    {
       ierr=VecDestroy(epsLast);CHKERRQ(ierr);
       ierr=VecDestroy(epsPre);CHKERRQ(ierr);
       ierr=VecDestroy(xLast);CHKERRQ(ierr);
       ierr=VecDestroy(xPre);CHKERRQ(ierr);
       ierr=VecDestroy(xWLastSqr);CHKERRQ(ierr);
       ierr=VecDestroy(xWPreSqr);CHKERRQ(ierr);
       ierr=VecDestroy(vscratch);CHKERRQ(ierr);
       ierr=VecDestroy(vecdeps);CHKERRQ(ierr);
       if (mynloptlocalalg) 
	 nlopt_destroy(local_opt);
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




