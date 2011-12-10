#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include "Resonator.h"

int count=1;
int its=100;
int maxit=15;
int ccount=1;
double cldos=0;
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
    Mzslab = atoi(argv[6]); // note Mzslab is different for Mz;
    Npmlx = atoi(argv[7]);
    Npmly = atoi(argv[8]);
    Npmlz = atoi(argv[9]);
    epsair = atof(argv[10]);
    epssub = atof(argv[11]);    
    hx = atof(argv[12]);
    omega = atof(argv[13]);
    Qabs = atof(argv[14]);
    initialdata = argv[15];
    filenameComm = argv[16];
    maxeval = atoi(argv[17]);
    maxtime = atof(argv[18]);
    mynloptalg = atoi(argv[19]);
    mylb = atof(argv[20]);
    myub = atof(argv[21]);
    BCPeriod = atoi(argv[22]);
    bx[0] = atoi(argv[23]);
    bx[1] = atoi(argv[24]);
    Jdirection = atoi(argv[25]);    
    LowerPML = atoi(argv[26]);
    // below is new for this frequency optimization;
    optmax = atoi(argv[27]);
    sparedouble = atof(argv[28]);
   */

   /* set nlopt variables */
  int maxeval = atoi(argv[12]); // for nlopt purpose;
  double maxtime = atof(argv[13]);
  int mynloptalg = atoi(argv[14]);
  double mylb = atof(argv[15]);
  double myub = atof(argv[16]);
  int optmax = atoi(argv[17]);
  double sparedouble = atof(argv[18]); //use it to store lcenter;


 PetscPrintf(PETSC_COMM_WORLD,"nlopt method use is %d \n",mynloptalg);
  
  int Nx, Ny, Nz, Nxyz;
  int Npmlx, Npmly, Npmlz;
  double omega, Qabs, RRT;
  
  Nx = atoi(argv[1]);
  Ny = atoi(argv[2]);
  Nz = 1;
  Nxyz = Nx*Ny*Nz;
  
  Npmlx = atoi(argv[5]);
  Npmly = atoi(argv[6]);
  Npmlz = 0; 
  
  omega = atof(argv[8]);
  Qabs = atof(argv[9]);
  RRT = 1e-25;
  
  int BCPeriod;
  BCPeriod = 3;

  
  char *filenameComm, *initialdata;
  initialdata = argv[10];
  filenameComm = argv[11];

   /*------- Set the Boundary Conditions----------*/
  double hx, hy, hz;

  hx = atof(argv[7]);
  //hx = 1.0/(Nx-0.5);// for Dirichlet B.C.'s.
  //hx = 1.0/Nx; // for Periodic B.C.'s
  hy = hx;
  hz = hx;// avoid hz = inf for Nz=1; 

  int bx[2],by[2],bz[2];
  bx[0] = -1;
  bx[1] = -1;
  by[0] = 1;
  by[1] = 1;
  bz[0] = 1;
  bz[1] = 1;

 
  /*-------------------- Setup PML Parameters ----------------------- */

  double sigmax, sigmay, sigmaz;
  sigmax = pmlsigma(RRT,Npmlx*hx);
  sigmay = pmlsigma(RRT,Npmly*hx);
  sigmaz = pmlsigma(RRT,Npmlz*hz);  

  /*---------- Set the current source---------*/
  Mat D; //ImaginaryIMatrix;
  ImagIMat(PETSC_COMM_WORLD, &D,Nxyz);

  Vec J;
  SourceSingleSetX(PETSC_COMM_WORLD, &J, Nx, Ny, Nz, 0, 0, 0, 1.0/(hx*hy));

  Vec b; // b= i*omega*J;
  ierr = VecDuplicate(J,&b);CHKERRQ(ierr);
  ierr = MatMult(D,J,b);CHKERRQ(ierr);
  VecScale(b,omega);

  /*-------Get the weight vector ------------------*/
  Vec weight;
  ierr = VecDuplicate(J,&weight); CHKERRQ(ierr);
  GetWeightVec(weight, Nx, Ny,Nz);

  Vec weightedJ;
  ierr = VecDuplicate(J,&weightedJ); CHKERRQ(ierr);
  ierr = VecPointwiseMult(weightedJ,J,weight);

  Vec vR;
  ierr = VecDuplicate(J,&vR); CHKERRQ(ierr);
  GetRealPartVec(vR, Nxyz);

  /*----------- Define PML muinv vectors  */
  int LowerPML = 0; // 1 for PML on lower b.c., 0 otherwise;
  Vec muinvpml; 
  MuinvPMLFull(PETSC_COMM_SELF, &muinvpml,Nx,Ny,Nz,Npmlx,Npmly,Npmlz,sigmax,sigmay,sigmaz,omega, LowerPML);

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
  
  int Mx = atoi(argv[3]), My = atoi(argv[4]), Mz = 1;
  int Mxyz = Mx*My*Mz; 
  Mat A; 

  myinterp(PETSC_COMM_WORLD, &A, Nx,Ny,Nz, 0,0,0, Mx,My,Mz); // 0,0,0 specify where the interp starts;  

  Vec epsSReal, epsgrad, vgrad; // create compatiable vectors with A.
  ierr = MatGetVecs(A,&epsSReal, &epsgrad); CHKERRQ(ierr);  
  ierr = VecDuplicate(epsSReal, &vgrad); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vgrad, "vgrad");CHKERRQ(ierr);



  /*---------- Output the commandline parameters--------*/
  
  PetscPrintf(PETSC_COMM_WORLD,"Nx is %d \n",Nx);
  PetscPrintf(PETSC_COMM_WORLD,"Ny is %d \n",Ny);
  PetscPrintf(PETSC_COMM_WORLD,"Mx is %d \n",Mx);
  PetscPrintf(PETSC_COMM_WORLD,"My is %d \n",My);
  PetscPrintf(PETSC_COMM_WORLD,"Npmlx is %d \n",Npmlx);
  PetscPrintf(PETSC_COMM_WORLD,"Npmly is %d \n",Npmly);
  PetscPrintf(PETSC_COMM_WORLD,"hx is %lf \n",hx);
  PetscPrintf(PETSC_COMM_WORLD,"omega is %lf \n",omega);
  PetscPrintf(PETSC_COMM_WORLD,"the Qabs is %lf \n", Qabs);
  PetscPrintf(PETSC_COMM_WORLD,"Inputdata file is %s \n",initialdata);
  PetscPrintf(PETSC_COMM_WORLD,"Outputfilename is %s \n",filenameComm);
  PetscPrintf(PETSC_COMM_WORLD,"Max number of evaluation is %d \n",maxeval);
  PetscPrintf(PETSC_COMM_WORLD,"Max time of evaluation is %f seconds \n",maxtime);
  PetscPrintf(PETSC_COMM_WORLD,"optimization lb is %lf \n",mylb);
  PetscPrintf(PETSC_COMM_WORLD,"optimization ub is %lf \n",myub);
  PetscPrintf(PETSC_COMM_WORLD,"doing maximizing is %d \n",optmax);
  PetscPrintf(PETSC_COMM_WORLD,"target ldos is %.16e \n",sparedouble);
  /*--------- Setup the finitie difference matrix-------------*/
  Mat M;
  MoperatorGeneral(PETSC_COMM_WORLD, &M, Nx,Ny,Nz,hx,hy,hz, bx, by, bz, muinvpml,BCPeriod);
  
 /*--------Setup the KSP variables ---------------*/
  
  KSP ksp;
  PC pc; 
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MAT_SOLVER_SUPERLU_DIST);
  //ierr = PCFactorSetMatSolverPackage(pc,MAT_SOLVER_PASTIX);
  //ierr = PCFactorSetMatSolverPackage(pc,MAT_SOLVER_MUMPS);
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
  Vec vgradlocal, tmp, tmpa,tmpb;
  ierr = VecCreateSeq(PETSC_COMM_SELF, Mxyz, &vgradlocal); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmp); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmpa); CHKERRQ(ierr);
  ierr = VecDuplicate(J,&tmpb); CHKERRQ(ierr);
 
  /*------------ Create scatter used in the solver -----------*/
  VecScatter scatter;
  IS from, to;
  ierr =ISCreateStride(PETSC_COMM_SELF,Mxyz,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,Mxyz,0,1,&to); CHKERRQ(ierr);


  /*---------Variables passed to Solver--------*/
  /*  epsopt=epsSReal (before interpolation), this is the varibale we optimized;*/      
// set initial data;
  
  int numofvar=1;
  double *varopt, *lb, *ub;
  varopt = (double *) malloc(numofvar*sizeof(double));
  lb = (double *) malloc(numofvar*sizeof(double));
  ub = (double *) malloc(numofvar*sizeof(double));
  
  lb[0] = mylb;
  ub[0] = myub;
  varopt[0] = omega;
  
  // use spartptdouble to store current omega;
  double spareptdouble[] = {0}; //initialization;

  myfundatatype myfundata = {Nx, Ny, Nz, hx, hy, hz, omega, sparedouble, spareptdouble, ksp, epspmlQ, epscoef, epsC, epsCi, epsP, x, cglambda, b, weightedJ, vR, epsSReal, epsgrad, vgrad, vgradlocal, tmp, tmpa, tmpb, A, D, M, from, to, scatter, filenameComm};
 
  // copy epsilon from file to epsSReal; (different from FindOpt.c, because epsilon is not degree-of-freedoms in computeQ.

  // i) create a array to read file; ii) convert the array to epsSReal;
  double *epsinput;
  epsinput = (double *) malloc(Mxyz*sizeof(double));

  FILE *ptf;
  ptf = fopen(initialdata,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");

  int i;
  for (i=0;i<Mxyz;i++)
     fscanf(ptf,"%lf",&epsinput[i]);
  fclose(ptf);


  int j, ns, ne;
  ierr = VecGetOwnershipRange(epsSReal,&ns,&ne);
  for(j=ns;j<ne;j++)
    { ierr=VecSetValue(epsSReal,j,epsinput[j],INSERT_VALUES); 
      CHKERRQ(ierr); }

  ierr = VecAssemblyBegin(epsSReal); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(epsSReal);  CHKERRQ(ierr);

  free(epsinput);


#if 0
  double myldos;
  double *agrad;
  myldos=ldos(numofvar, varopt, agrad, &myfundata);

#endif



#if 1
  double maxf;
  nlopt_opt  opt;
  nlopt_result result;
  opt = nlopt_create(mynloptalg, numofvar);
  nlopt_set_lower_bounds(opt,lb);
  nlopt_set_upper_bounds(opt,ub);
  nlopt_set_maxeval(opt,maxeval);
  nlopt_set_maxtime(opt,maxtime);

  if(optmax==1)
    nlopt_set_max_objective(opt, ldos, &myfundata);
  else
    nlopt_set_min_objective(opt, ldosdiff, &myfundata);

  result = nlopt_optimize(opt,varopt,&maxf);

 #if 0
  double xrel, frel, fabs;
  double *xabs;
   frel=nlopt_get_ftol_rel(opt);
   fabs=nlopt_get_ftol_abs(opt);
   xrel=nlopt_get_xtol_rel(opt);
   PetscPrintf(PETSC_COMM_WORLD,"nlopt frel is %.20e \n",frel);
   PetscPrintf(PETSC_COMM_WORLD,"nlopt fabs is %.20e \n",fabs);
   PetscPrintf(PETSC_COMM_WORLD,"nlopt xrel is %.20e \n",xrel);
//nlopt_result nlopt_get_xtol_abs(const nlopt_opt opt, double *tol);
#endif


  if (result < 0) {
    PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"found extermum  %0.16e\n", maxf); 
  }

 PetscPrintf(PETSC_COMM_WORLD,"--the frequency intersted is %0.16e------ \n",*varopt);


  PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);


 
#endif 
 

  
   ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

  /* ----------------------Destroy Vecs and Mats----------------------------*/ 
   // nlopt_destroy(opt);

  free(varopt);
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




