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

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize and Get the parameters from command line ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;

  /*
    Nx = atoi(argv[1]);
    Ny = atoi(argv[2]);
    Nz = atoi(argv[3]);
    Mx = atoi(argv[4]);
    My = atoi(argv[5]);
    Mz = atoi(argv[6]); // note I add Mz on Nov 22, 2011;
    Mzslab = atoi(argv[6+1]); // note Mzslab is different for Mz;
    Npmlx = atoi(argv[7+1]);
    Npmly = atoi(argv[8+1]);
    Npmlz = atoi(argv[9+1]);
    epsair = atof(argv[10+1]);
    epssub = atof(argv[11+1]);    
    hx = atof(argv[12+1]);
    omega = atof(argv[13+1]);
    Qabs = atof(argv[14+1]);
    initialdata = argv[15+1];
    filenameComm = argv[16+1];
    maxeval = atoi(argv[17+1]);
    maxtime = atof(argv[18+1]);
    mynloptalg = atoi(argv[19+1]);
    mylb = atof(argv[20+1]);
    myub = atof(argv[21+1]);
    BCPeriod = atoi(argv[22+1]);
    bx[0] = atoi(argv[23+1]);
    bx[1] = atoi(argv[24+1]);
    by[0] = atoi(argv[25+1]);
    by[1] = atoi(argv[26+1]);
    bz[0] = atoi(argv[27+1]);
    bz[1] = atoi(argv[28+1]);
    Jdirection = atoi(argv[29+1]);    
    LowerPML = atoi(argv[30+1]);
    RRT = atof(argv[31+1]);
   */

  mma_verbose=1;
 
  /* set nlopt variables */
  int maxeval=atoi(argv[18]); // for nlopt purpose;
  double maxtime = atof(argv[19]);
  int mynloptalg = atoi(argv[20]); // 11 for LBFGS and 24 for MMA;
  
  double epsair = atof(argv[11]);
  double epssub = atof(argv[12]);

  double mylb = atof(argv[21]);
  double myub = atof(argv[22]); 
  

  int Nx, Ny, Nz, Nxyz;
  int Npmlx, Npmly, Npmlz;
  double omega, Qabs, RRT;
  
  Nx = atoi(argv[1]);
  Ny = atoi(argv[2]);
  Nz = atoi(argv[3]);
  Nxyz = Nx*Ny*Nz;
  
  
  Npmlx = atoi(argv[8]);
  Npmly = atoi(argv[9]);
  Npmlz = atoi(argv[10]); 
  
  omega = atof(argv[14]);
  Qabs = atof(argv[15]);
  if (Qabs>1e+15)
    Qabs=1.0/0.0;
  RRT = atof(argv[32]);
  
  int BCPeriod = atoi(argv[23]);
  
  char *filenameComm, *initialdata;
  initialdata = argv[16];
  filenameComm = argv[17];
 
   /*------- Set the Boundary Conditions----------*/
  double hx, hy, hz, hxyz;

  hx = atof(argv[13]);
  hy = hx;
  hz = hx;
  hxyz = (Nz==1)*hx*hy + (Nz>1)*hx*hy*hz;

  int bx[2],by[2],bz[2];
  bx[0] = atoi(argv[24]); // odd boundary condition for 3D;
  bx[1] = atoi(argv[25]);
  by[0] = atoi(argv[26]);
  by[1] = atoi(argv[27]);
  bz[0] = atoi(argv[28]);
  bz[1] = atoi(argv[29]);

 
  /*-------------------- Setup PML Parameters ----------------------- */

  double sigmax, sigmay, sigmaz;
  sigmax = pmlsigma(RRT,Npmlx*hx);
  sigmay = pmlsigma(RRT,Npmly*hy);
  sigmaz = pmlsigma(RRT,Npmlz*hz);  

  /*---------- Set the current source---------*/
  Mat D; //ImaginaryIMatrix;
  ImagIMat(PETSC_COMM_WORLD, &D,Nxyz);

  Vec J;
  int Jdirection = atoi(argv[30]);

  if (Jdirection == 1)
    SourceSingleSetX(PETSC_COMM_WORLD, &J, Nx, Ny, Nz, 0, 0, 0, 1.0/hxyz);
  else if (Jdirection == 3)
    SourceSingleSetZ(PETSC_COMM_WORLD, &J, Nx, Ny, Nz, 0, 0, 0, 1.0/hxyz);
  else
    PetscPrintf(PETSC_COMM_WORLD," Please specify corret direction of current "); 

  Vec tmp; //space for calculation;
  ierr = VecDuplicate(J,&tmp); CHKERRQ(ierr);

  Vec b; // b= i*omega*J;
  ierr = VecDuplicate(J,&b);CHKERRQ(ierr);
  ierr = MatMult(D,J,b);CHKERRQ(ierr);
  VecScale(b,omega);

  /*-------Get the weight vector ------------------*/
  Vec weight;
  ierr = VecDuplicate(J,&weight); CHKERRQ(ierr);
  GetWeightVec(weight, Nx, Ny,Nz); // new code handles both 3D and 2D;

  Vec weightedJ;
  ierr = VecDuplicate(J,&weightedJ); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJ,J,weight);

  Vec vR;
  ierr = VecDuplicate(J,&vR); CHKERRQ(ierr);
  GetRealPartVec(vR, Nxyz);

  /*----------- Define PML muinv vectors  */
  int LowerPML = atoi(argv[31]); // 1 for PML on lower b.c., 0 otherwise;
  Vec muinvpml; 
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

  AddMuAbsorption(muinvpml, Qabs, D, tmp);

  /*---------- Define PML eps vectors: epspml---------- */  
  Vec epspml, epspmlQ, epscoef;
  ierr = VecDuplicate(J,&epspml);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epspml,"EpsPMLFull"); CHKERRQ(ierr);
  EpsPMLFull(PETSC_COMM_WORLD, epspml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

  ierr = VecDuplicate(J,&epspmlQ);CHKERRQ(ierr);
  ierr = VecDuplicate(J,&epscoef);CHKERRQ(ierr);
 
  // compute epspmlQ,epscoef;
  EpsCombine(D, weight, epspml, epspmlQ, epscoef, Qabs, omega);

  /*--------- Setup the interp matrix ----------------------- */
  /* for a samll eps block, interp it into yee-lattice. The interp matrix A and PML epspml only need to generated once;*/
  
  int Mx = atoi(argv[4]), My = atoi(argv[5]), Mz = atoi(argv[6]), Mzslab = atoi(argv[7]);
  int Mxyz = Mx*My*Mz; 
  Mat A; 

  //new routine for myinterp;
  myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, 0,0,0, Mx,My,Mz,Mzslab); // 0,0,0 specify where the interp starts; 
 
  Vec epsSReal, epsgrad, vgrad; // create compatiable vectors with A.
  ierr = MatGetVecs(A,&epsSReal, &epsgrad); CHKERRQ(ierr);  
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vgrad, "vgrad");CHKERRQ(ierr);

  /*---------Setup the epsmedium vector----------------*/
  Vec epsmedium;
  ierr = VecDuplicate(J,&epsmedium); CHKERRQ(ierr);
  GetMediumVec(epsmedium,Nz,Mz,epsair,epssub);


  /*---------- Output the commandline parameters--------*/
  PetscPrintf(PETSC_COMM_WORLD,"The algorithm used is %d \n",mynloptalg);
  PetscPrintf(PETSC_COMM_WORLD,"Nx is %d \n",Nx);
  PetscPrintf(PETSC_COMM_WORLD,"Ny is %d \n",Ny);
  PetscPrintf(PETSC_COMM_WORLD,"Nz is %d \n",Nz);
  PetscPrintf(PETSC_COMM_WORLD,"Mx is %d \n",Mx);
  PetscPrintf(PETSC_COMM_WORLD,"My is %d \n",My);
  PetscPrintf(PETSC_COMM_WORLD,"Mz is %d \n",Mz);
  PetscPrintf(PETSC_COMM_WORLD,"Mzslab is %d \n",Mzslab);
  PetscPrintf(PETSC_COMM_WORLD,"Npmlx is %d \n",Npmlx);
  PetscPrintf(PETSC_COMM_WORLD,"Npmly is %d \n",Npmly);
  PetscPrintf(PETSC_COMM_WORLD,"Npmlz is %d \n",Npmlz);
  PetscPrintf(PETSC_COMM_WORLD,"epsair is %lf \n",epsair);
  PetscPrintf(PETSC_COMM_WORLD,"epsslab is %lf \n",epssub);
  PetscPrintf(PETSC_COMM_WORLD,"hx is %lf \n",hx);
  PetscPrintf(PETSC_COMM_WORLD,"omega is %lf \n",omega);
  PetscPrintf(PETSC_COMM_WORLD,"Qabs is %lf \n",Qabs);
  PetscPrintf(PETSC_COMM_WORLD,"Inputdata file is %s \n",initialdata);
  PetscPrintf(PETSC_COMM_WORLD,"Outputfilename is %s \n",filenameComm);
  PetscPrintf(PETSC_COMM_WORLD,"max number of evaluation is %d \n",maxeval);
  PetscPrintf(PETSC_COMM_WORLD,"max time of evaluation is %f \n",maxtime);
  PetscPrintf(PETSC_COMM_WORLD,"optimization lb is %lf \n",mylb);
  PetscPrintf(PETSC_COMM_WORLD,"optimization ub is %lf \n",myub);
  PetscPrintf(PETSC_COMM_WORLD,"BCPeriod given is %d \n",BCPeriod);
  PetscPrintf(PETSC_COMM_WORLD,"BC in lower-x end is %d \n",bx[0]);
  PetscPrintf(PETSC_COMM_WORLD,"BC in upper-x end is %d  \n",bx[1]);
  PetscPrintf(PETSC_COMM_WORLD,"BC in lower-y end is %d \n",by[0]);
  PetscPrintf(PETSC_COMM_WORLD,"BC in upper-y end is %d  \n",by[1]);
  PetscPrintf(PETSC_COMM_WORLD,"BC in lower-z end is %d \n",bz[0]);
  PetscPrintf(PETSC_COMM_WORLD,"BC in upper-z end is %d  \n",bz[1]);
  PetscPrintf(PETSC_COMM_WORLD,"Diapole current direction is %d \n",Jdirection);
  PetscPrintf(PETSC_COMM_WORLD,"PML in the lower xyz boundary is %d \n",LowerPML);
 PetscPrintf(PETSC_COMM_WORLD,"RRT given is %.12e \n",RRT);
 PetscPrintf(PETSC_COMM_WORLD,"sigmax is %.12e \n",sigmax);
 PetscPrintf(PETSC_COMM_WORLD,"sigmay is %.12e \n",sigmay);
 PetscPrintf(PETSC_COMM_WORLD,"sigmaz is %.12e \n",sigmaz);
 
  /*--------- Setup the finitie difference matrix-------------*/
  Mat M;
  MoperatorGeneral(PETSC_COMM_WORLD, &M, Nx,Ny,Nz,hx,hy,hz, bx, by, bz, muinvpml,tmp,BCPeriod);
  
 /*--------Setup the KSP variables ---------------*/
  
  KSP ksp;
  PC pc; 
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MAT_SOLVER_PASTIX);
  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxit);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /*--------- Create the space for solution vector -------------*/
  Vec x;
  ierr = VecDuplicate(J,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr); 

  Vec cglambda;
  ierr = VecDuplicate(J,&cglambda);CHKERRQ(ierr);
  
  /*----------- Create the space for final eps -------------*/

  Vec epsC, epsCi, epsP;
  ierr = VecDuplicate(J,&epsC);CHKERRQ(ierr);
  ierr = VecDuplicate(J,&epsCi);CHKERRQ(ierr);
  ierr = VecDuplicate(J,&epsP);CHKERRQ(ierr);

  ierr = VecSet(epsP,0.0); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(epsP); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsP); CHKERRQ(ierr); 

  /*------------ Create space used in the solver ------------*/
  Vec vgradlocal, tmpa,tmpb;
  ierr = VecCreateSeq(PETSC_COMM_SELF, Mxyz, &vgradlocal); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmpa); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmpb); CHKERRQ(ierr);
 
  /*------------ Create scatter used in the solver -----------*/
  VecScatter scatter;
  IS from, to;
  ierr =ISCreateStride(PETSC_COMM_SELF,Mxyz,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,Mxyz,0,1,&to); CHKERRQ(ierr);


  /*---------Variables passed to Solver--------*/
  /*  epsopt=epsSReal (before interpolation), this is the varibale we optimized;*/      

  myfundatatype myfundata = {Nx, Ny, Nz, hx, hy, hz, omega, ksp, epspmlQ, epscoef, epsmedium, epsC, epsCi, epsP, x, cglambda, b, weightedJ, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, A, D, M, from, to, scatter, filenameComm};
  
  double *epsopt, *lb, *ub;
  epsopt = (double *) malloc(Mxyz*sizeof(double));
  lb = (double *) malloc(Mxyz*sizeof(double));
  ub = (double *) malloc(Mxyz*sizeof(double));


  // read initial data from a file;
  FILE *ptf;
  ptf = fopen(initialdata,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");


  int i;
  for(i=0;i<Mxyz;i++)
    {
      fscanf(ptf,"%lf",&epsopt[i]);
      //PetscPrintf(PETSC_COMM_WORLD,"current eps reading is %lf \n",epsopt[i]);
      lb[i] = mylb;
      ub[i] = myub;
    }  //initial guess, lower bounds, upper bounds;
 
  fclose(ptf);

#if 1
  double maxf;
  nlopt_opt  opt;
  nlopt_result result;
  opt = nlopt_create(mynloptalg, Mxyz);

  nlopt_set_lower_bounds(opt,lb);
  nlopt_set_upper_bounds(opt,ub);
  nlopt_set_maxeval(opt,maxeval);
  nlopt_set_maxtime(opt,maxtime);

  nlopt_set_max_objective(opt, ResonatorSolver, &myfundata);
  result = nlopt_optimize(opt,epsopt,&maxf);

  /* print the optimization parameters */
 #if 1

  double xrel, frel, fabs;
  double *xabs;
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
    PetscPrintf(PETSC_COMM_WORLD,"found maximum  %0.16e\n", maxf); 
  }

  PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);


  PetscPrintf(PETSC_COMM_WORLD,"--Computed fields is %0.10e------ \n",maxf);
#endif 
 

  /*---------------- Output the results we need------------------------ */
  PetscPrintf(PETSC_COMM_WORLD,"--------Output the reuslts------\n");
  OutputVec(PETSC_COMM_WORLD, x,filenameComm, "x.m");
  OutputVec(PETSC_COMM_WORLD, epsSReal,filenameComm, "epsSReal.m");
  
  
  //OutputVec(PETSC_COMM_WORLD, epsopt,filenameComm, "epsopt.m");

      int rankA;
       MPI_Comm_rank(PETSC_COMM_WORLD, &rankA);

       if(rankA==0)
	 {
	   ptf = fopen(strcat(filenameComm,"epsopt.txt"),"w");
	   for (i=0;i<Mxyz;i++)
	     fprintf(ptf,"%0.16e \n",epsopt[i]);
	 }
  

  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

  /* ----------------------Destroy Vecs and Mats----------------------------*/ 
  nlopt_destroy(opt);

  free(epsopt);
  free(lb);
  free(ub);

  ierr = VecDestroy(J); CHKERRQ(ierr);
  ierr = VecDestroy(b); CHKERRQ(ierr);
  ierr = VecDestroy(weight); CHKERRQ(ierr);
  ierr = VecDestroy(weightedJ); CHKERRQ(ierr);
  ierr = VecDestroy(vR); CHKERRQ(ierr);
  ierr = VecDestroy(muinvpml); CHKERRQ(ierr);
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




