#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include "Resonator.h"

int count=1;
int its=100;
int itsx=100;
int itsy=100;
int maxit=20;
double cldos=0;//set initial ldos;
int ccount=1;
extern int mma_verbose;


#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize and Get the parameters from command line ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;

  PetscTruth flg;

  /*-------------------------------------------------*/
  int Nx,Ny,Nz,Mx,My,Mz,Mzslab, Npmlx,Npmly,Npmlz, Nxyz,Mxyz;

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
  int bcx[2], bcy[2], bcz[2];
  PetscOptionsGetInt(PETSC_NULL,"-BCPeriod",&BCPeriod,&flg);  MyCheckAndOutputInt(flg,BCPeriod,"BCPeriod","BCPeriod given");
  PetscOptionsGetInt(PETSC_NULL,"-LowerPML",&LowerPML,&flg);  MyCheckAndOutputInt(flg,LowerPML,"LowerPML","PML in the lower xyz boundary");
  

  double hx, hy,hz, hxyz, omega, Qabs,epsair, epssub, RRT, sigmax, sigmay, sigmaz ;
   
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

  char initialdata[PETSC_MAX_PATH_LEN], filenameComm[PETSC_MAX_PATH_LEN];
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
    
  /*--------------------------------------------------------*/

  /*--------------------------------------------------------*/


  /*---------- Set the current source---------*/
  Mat D; //ImaginaryIMatrix;
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nxyz);

  Vec Jx, Jy;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nxyz, &Jx);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Jx, "SourceX");CHKERRQ(ierr);
  VecSet(Jx,0.0); //initialization;
  SourceSingleSetX(PETSC_COMM_WORLD, Jx, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);

  ierr = VecDuplicate(Jx,&Jy); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Jy, "SourceY");CHKERRQ(ierr);
  VecSet(Jy,0.0); //initialization;
  SourceSingleSetY(PETSC_COMM_WORLD, Jy, Nx, Ny, Nz, cx, cy, cz,1.0/hxyz);

  Vec bx, by; // b= i*omega*J;
  ierr = VecDuplicate(Jx,&bx);CHKERRQ(ierr);
  ierr = MatMult(D,Jx,bx);CHKERRQ(ierr);
  VecScale(bx,omega);

  ierr = VecDuplicate(Jy,&by);CHKERRQ(ierr);
  ierr = MatMult(D,Jy,by);CHKERRQ(ierr);
  VecScale(by,omega);

  /*-------Get the weight vector ------------------*/
  Vec weight;
  ierr = VecDuplicate(Jx,&weight); CHKERRQ(ierr);
  
  if(LowerPML==0)
    GetWeightVec(weight, Nx, Ny,Nz); // new code handles both 3D and 2D;
  else
    VecSet(weight,1.0);

  Vec weightedJx, weightedJy;
  ierr = VecDuplicate(Jx,&weightedJx); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJx,Jx,weight);

  ierr = VecDuplicate(Jy,&weightedJy); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJy,Jy,weight);

  Vec vR;
  ierr = VecDuplicate(Jx,&vR); CHKERRQ(ierr);
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
  Vec epspml, epspmlQ, epscoef;
  ierr = VecDuplicate(Jx,&epspml);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epspml,"EpsPMLFull"); CHKERRQ(ierr);
  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

  ierr = VecDuplicate(Jx,&epspmlQ);CHKERRQ(ierr);
  ierr = VecDuplicate(Jx,&epscoef);CHKERRQ(ierr);
 
  // compute epspmlQ,epscoef;
  EpsCombine(D, weight, epspml, epspmlQ, epscoef, Qabs, omega);

  /*--------- Setup the interp matrix ----------------------- */
  /* for a samll eps block, interp it into yee-lattice. The interp matrix A and PML epspml only need to generated once;*/
  

  Mat A; 
  //new routine for myinterp;
  myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, LowerPML*Npmlx,LowerPML*Npmly,LowerPML*Npmlz, Mx,My,Mz,Mzslab); // LoweerPML*Npmlx,..,.., specify where the interp starts;  

  Vec epsSReal, epsgrad, vgrad; // create compatiable vectors with A.
  ierr = MatGetVecs(A,&epsSReal, &epsgrad); CHKERRQ(ierr);  
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vgrad, "vgrad");CHKERRQ(ierr);

  /*---------Setup the epsmedium vector----------------*/
  Vec epsmedium;
  ierr = VecDuplicate(Jx,&epsmedium); CHKERRQ(ierr);
  GetMediumVec(epsmedium,Nz,Mz,epsair,epssub);
 
  /*--------- Setup the finitie difference matrix-------------*/
  Mat MatX, MatY;
  bcz[0]=1, bcz[1]=1;

  bcx[0]=-1; bcx[1]=-1; bcy[0]=1;bcy[1]=1; // for Jdirection in x;
  MoperatorGeneral(PETSC_COMM_WORLD, &MatX, Nx,Ny,Nz,hx,hy,hz, bcx, bcy, bcz,muinv,BCPeriod);

  bcx[0]=1, bcx[1]=1; bcy[0]=-1; bcy[1]=-1;
  MoperatorGeneral(PETSC_COMM_WORLD, &MatY, Nx,Ny,Nz,hx,hy,hz, bcx, bcy, bcz,muinv,BCPeriod);
  free(muinv);

  /*--------Setup the KSP variables ---------------*/
  
  KSP kspx, kspy;
  PC pcx, pcy; 
  ierr = KSPCreate(PETSC_COMM_WORLD,&kspx);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetType(kspx, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(kspx,&pcx);CHKERRQ(ierr);
  ierr = PCSetType(pcx,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pcx,MAT_SOLVER_PASTIX);CHKERRQ(ierr);
  int maxkspit = 20;
  ierr = KSPSetTolerances(kspx,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxkspit);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspx);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&kspy);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetType(kspy, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(kspy,&pcy);CHKERRQ(ierr);
  ierr = PCSetType(pcy,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pcy,MAT_SOLVER_PASTIX);CHKERRQ(ierr);
 
  ierr = KSPSetTolerances(kspy,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxkspit);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspy);CHKERRQ(ierr);


  /*--------- Create the space for solution vector -------------*/
  Vec solx, soly;
  ierr = VecDuplicate(Jx,&solx);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) solx, "SolutionX");CHKERRQ(ierr); 

  ierr = VecDuplicate(Jy,&soly);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) soly, "SolutionY");CHKERRQ(ierr); 


  Vec cglambda;
  ierr = VecDuplicate(Jx,&cglambda);CHKERRQ(ierr);
  
  /*----------- Create the space for final eps -------------*/

  Vec epsC, epsCi, epsPx, epsPy;
  ierr = VecDuplicate(Jx,&epsC);CHKERRQ(ierr);
  ierr = VecDuplicate(Jx,&epsCi);CHKERRQ(ierr);
  ierr = VecDuplicate(Jx,&epsPx);CHKERRQ(ierr);
  ierr = VecDuplicate(Jy,&epsPy);CHKERRQ(ierr);
 
  ierr = VecSet(epsPx,0.0); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(epsPx); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsPx); CHKERRQ(ierr); 
  ierr = VecSet(epsPy,0.0); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(epsPy); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsPy); CHKERRQ(ierr); 

  /*------------ Create space used in the solver ------------*/
  Vec vgradlocal,tmp, tmpa,tmpb;
  ierr = VecCreateSeq(PETSC_COMM_SELF, Mxyz, &vgradlocal); CHKERRQ(ierr);
  ierr = VecDuplicate(Jx,&tmp); CHKERRQ(ierr);
  ierr = VecDuplicate(Jx,&tmpa); CHKERRQ(ierr);
  ierr = VecDuplicate(Jx,&tmpb); CHKERRQ(ierr);
 
  /*------------ Create scatter used in the solver -----------*/
  VecScatter scatter;
  IS from, to;
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



  /*----declare these data types, althought they may not be used for job 2 -----------------*/
 
  double mylb,myub, *varopt, *lb=NULL, *ub=NULL;
  int maxeval, maxtime, mynloptalg;
  double maxf;
  nlopt_opt  opt;
  nlopt_result result;
  /*--------------------------------------------------------------*/
  /*----Now based on Command Line, Do the corresponding job----*/
  /*----------------------------------------------------------------*/


  int Job;
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
    }




  switch (Job)
    {
    case 1:
      {
	myfundataPOLXYtype myfundata = {Nx, Ny, Nz, hx, hy, hz, omega, kspx, kspy, epspmlQ, epscoef, epsmedium, epsC, epsCi, epsPx, epsPy, solx, soly, cglambda, bx, by, weightedJx, weightedJy, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, A, D, MatX, MatY, from, to, scatter, filenameComm};
      nlopt_set_max_objective(opt, ResonatorSolverPOLXY, &myfundata);
      result = nlopt_optimize(opt,epsopt,&maxf);
      }      
      break;
    case 2 :  //AnalyzeStructure
      { 
	int Linear, Eig, maxeigit;
	PetscOptionsGetInt(PETSC_NULL,"-Linear",&Linear,&flg);  MyCheckAndOutputInt(flg,Linear,"Linear","Linear solver indicator");
	PetscOptionsGetInt(PETSC_NULL,"-Eig",&Eig,&flg);  MyCheckAndOutputInt(flg,Eig,"Eig","Eig solver indicator");
	PetscOptionsGetInt(PETSC_NULL,"-maxeigit",&maxeigit,&flg);  MyCheckAndOutputInt(flg,maxeigit,"maxeigit","maximum number of Eig solver iterations is");


	myfundataPOLXYtype myfundata = {Nx, Ny, Nz, hx, hy, hz, omega, kspx, kspy, epspmlQ, epscoef, epsmedium, epsC, epsCi, epsPx, epsPy, solx, soly, cglambda, bx, by, weightedJx, weightedJy, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, A, D, MatX, MatY, from, to, scatter, filenameComm};

	/*----------------------------------*/
	EigenSolver(&myfundata,Linear, Eig, maxeigit);
	/*----------------------------------*/
      }
      break;   

    case 3: //computeQ
      {
	int optmax;
	PetscOptionsGetInt(PETSC_NULL,"-optmax",&optmax,&flg);  MyCheckAndOutputInt(flg,optmax,"optmax","Indicator for max ComputeQ is ");

	double ldoscenter;
	PetscOptionsGetReal(PETSC_NULL,"-ldoscenter",&ldoscenter,&flg);  MyCheckAndOutputDouble(flg,ldoscenter,"ldoscenter","ldoscenter");

	// use spartptdouble to store current omega;
	double spareptdouble[] = {0}; //initialization;
	myfundataPOLXYtypeq myfundataq = {Nx, Ny, Nz, hx, hy, hz, omega, ldoscenter, spareptdouble, kspx, kspy, epspmlQ, epscoef, epsC, epsCi, epsPx, epsPy, solx, soly, cglambda, bx, by, weightedJx, weightedJy, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, A, D, MatX, MatY, from, to, scatter, filenameComm};
       
	varopt = (double *) malloc(numofvar*sizeof(double));
	varopt[0] = omega;
      
	if(optmax==1)
	  nlopt_set_max_objective(opt, ldos, &myfundataq);
	else
	  nlopt_set_min_objective(opt, ldosdiff, &myfundataq);
      
	result = nlopt_optimize(opt,varopt,&maxf);
	PetscPrintf(PETSC_COMM_WORLD,"--the frequency intersted is %0.16e------ \n",*varopt);
	free(varopt);
      }
      break;

    default:
      PetscPrintf(PETSC_COMM_WORLD,"--------Inteeresting! You're doing nothing!--------\n ");
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
  ierr = VecDestroy(Jx); CHKERRQ(ierr);
  ierr = VecDestroy(Jy); CHKERRQ(ierr);

  ierr = VecDestroy(bx); CHKERRQ(ierr);
  ierr = VecDestroy(by); CHKERRQ(ierr);
 
  ierr = VecDestroy(weight); CHKERRQ(ierr);
  ierr = VecDestroy(weightedJx); CHKERRQ(ierr);
  ierr = VecDestroy(weightedJy); CHKERRQ(ierr);

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
  ierr = VecDestroy(epsPx); CHKERRQ(ierr);
  ierr = VecDestroy(epsPy); CHKERRQ(ierr);

  ierr = VecDestroy(solx); CHKERRQ(ierr);
  ierr = VecDestroy(soly); CHKERRQ(ierr);

  ierr = VecDestroy(cglambda);CHKERRQ(ierr);
  ierr = VecDestroy(vgradlocal);CHKERRQ(ierr);
  ierr = VecDestroy(tmp); CHKERRQ(ierr);
  ierr = VecDestroy(tmpa); CHKERRQ(ierr);
  ierr = VecDestroy(tmpb); CHKERRQ(ierr);
  ierr = MatDestroy(A); CHKERRQ(ierr);  
  ierr = MatDestroy(D); CHKERRQ(ierr);
  ierr = MatDestroy(MatX); CHKERRQ(ierr);  
  ierr = MatDestroy(MatY); CHKERRQ(ierr);  

  ierr = KSPDestroy(kspx);CHKERRQ(ierr);
  ierr = KSPDestroy(kspy);CHKERRQ(ierr);

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




