/*------------Specify the dipole (N is odd) or four dipoles (N is even) -----*/

#if 0
  int scx=Nx/2, scy=Ny/2, scz=0;
  
  if (Nx%2 == 0)
    {
      SourceSingleSet(PETSC_COMM_WORLD, b, Nx, Ny, Nz, scx, scy, scz, omega/pow(hx,2));
      SourceSingleSet(PETSC_COMM_WORLD, b, Nx, Ny, Nz, scx-1, scy, scz, omega/pow(hx,2));
      SourceSingleSet(PETSC_COMM_WORLD, b, Nx, Ny, Nz, scx, scy-1, scz, omega/pow(hx,2));
      SourceSingleSet(PETSC_COMM_WORLD, b, Nx, Ny, Nz, scx-1, scy-1, scz, omega/pow(hx,2));
    }
  else
    SourceSingleSet(PETSC_COMM_WORLD, b, Nx, Ny, Nz, scx, scy, scz, omega/pow(hx,2));
#endif    


/*--------------------------------------------------*/


/* From epsprimitive in a small grid, output    epspmlQ = epspml*(1+i/Qabs)
and epsfinal = (A*epsprimitive).*epspmlQ */
#undef __FUNCT__ 
#define __FUNCT__ "EpsCombineOld"
void EpsCombineOld(MPI_Comm comm, Mat A, Mat D, Vec epsSReal, Vec epspml, Vec epspmlQ, Vec epsfinal, double Qabs)
{
  PetscErrorCode ierr;
    
  // compute epspmlQ = epspml*(1+i/Qabs);
  ierr =MatMult(D,epspml,epspmlQ); CHKERRQ(ierr);
  ierr =VecScale(epspmlQ, 1.0/Qabs); CHKERRQ(ierr);
  ierr =VecAXPY(epspmlQ, 1.0, epspml);CHKERRQ(ierr);
  
  // compute epsfinal:
  ierr =MatMult(A, epsSReal,epsfinal); CHKERRQ(ierr); 
  VecShift(epsfinal,1.0); // remember to add 1 everyone where.

  
  ierr = VecPointwiseMult(epsfinal,epsfinal,epspmlQ); CHKERRQ(ierr);
}


/* ---------------demonstrate how to use scatter-----------------------*/

#undef __FUNCT__ 
#define __FUNCT__ "EpsModifyMat"
void EpsModifyMat(MPI_Comm comm, Mat M, Vec epsDiff, int Nxyz, double omega)
{
  PetscErrorCode ierr;
  int ns, ne, nrow;
  ierr = MatGetOwnershipRange(M, &ns, &ne); CHKERRQ(ierr);
  nrow = ne-ns;
  
  Vec epsDiffRealLocal, epsDiffImagLocal;
  VecCreateSeq(PETSC_COMM_SELF,nrow, &epsDiffRealLocal);
  VecCreateSeq(PETSC_COMM_SELF,nrow, &epsDiffImagLocal);
  
  /* -- scatter epsDiff */
  int *idfrom;
  idfrom = (int *) malloc(nrow*sizeof(int));
  

  int i;
  for(i=ns; i<ne; i++)
    idfrom[i-ns] = i%(3*Nxyz);
   
  IS from, to;
  int zero=0, one=1;
  ISCreateGeneral(comm,nrow,idfrom,&from);
  ISCreateStride(comm,nrow,zero,one,&to);

  VecScatter scatter;
 
  /* Scatter the real part */
  VecScatterCreate(epsDiff,from, epsDiffRealLocal, to, &scatter);
  VecScatterBegin(scatter,epsDiff,epsDiffRealLocal,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(scatter,epsDiff,epsDiffRealLocal,INSERT_VALUES,SCATTER_FORWARD);

  ierr=ISDestroy(from);CHKERRQ(ierr);
  ierr=VecScatterDestroy(scatter);CHKERRQ(ierr);

  /* Scatter the imaginary part */
  //update IS from;
  for(i=ns; i<ne; i++)
    idfrom[i-ns] = i%(3*Nxyz)+3*Nxyz;
  ISCreateGeneral(comm,nrow,idfrom,&from);

  VecScatterCreate(epsDiff,from, epsDiffImagLocal, to, &scatter);
  VecScatterBegin(scatter,epsDiff,epsDiffImagLocal,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(scatter,epsDiff,epsDiffImagLocal,INSERT_VALUES,SCATTER_FORWARD);
  
  ierr=VecScatterDestroy(scatter);CHKERRQ(ierr);

  /*Destroy Stuff */
  ierr=ISDestroy(from);CHKERRQ(ierr);
  ierr=ISDestroy(to);CHKERRQ(ierr);
  free(idfrom);

  double *ptReal, *ptImag;
  VecGetArray(epsDiffRealLocal,&ptReal);
  VecGetArray(epsDiffImagLocal,&ptImag);


  double omegasqr = pow(omega,2);
 
  /*-----------Change the matrix element of M = M - eps*omega^2;----------------*/
  for (i = ns; i < ne; ++i) 
    {
      ierr = MatSetValue(M,i,i,-omegasqr*ptReal[i-ns],ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(M,i,(i+3*Nxyz)%(6*Nxyz), pow(-1,i/(3*Nxyz))*omegasqr*ptImag[i-ns],ADD_VALUES);CHKERRQ(ierr);
    }

  ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  VecRestoreArray(epsDiffRealLocal,&ptReal);
  VecRestoreArray(epsDiffImagLocal,&ptImag);
 
  /*Destroy stuff*/
  VecDestroy(epsDiffRealLocal);
  VecDestroy(epsDiffImagLocal);
}


/*------------------------------------------------------------*/
/* another way to implement ModifyMatDiagonals */
#if 0
  /*----Compute the effective epsilon */
     
  EpsCombine(PETSC_COMM_WORLD,A, D, epsSReal,epspml,epspmlQ,epsCfinal,Qabs);
  
  //store the current - previous into current epsilon;
  VecAXPY(epsCfinal,-1.0,epsPfinal);
  
  // epsCfinal is the epsDiff now;

  /*Assembly the Matrix: ONLY need to modify the diagonals */
  EpsModifyMat(PETSC_COMM_WORLD, M,epsCfinal,Nxyz,omega);

  // update previous epsilon epsPfinal to current values;
  VecAXPY(epsPfinal,1.0,epsCfinal);

#endif
/*------------------------------------------------------------*/


/*------------------------------------------------------------*/

 /* an implementation to check the error of solving systems */

#if 0
  PetscReal norm;
  PetscInt its;
  ierr = MatMult(M,x, xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"----Norm of error %A, Iterations %D----\n ",norm,its);CHKERRQ(ierr);

  double rnorm;
  KSPGetResidualNorm(ksp,&rnorm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"----Norm of error %A,given by petsc in Iterations %D----\n ",rnorm,its);


  ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"----Maximum E fields %A----\n ",norm);CHKERRQ(ierr);

#endif

/*------------------------------------------------------------*/


 /*--------------Output the fields at a particular point -------*/
#if 0
  //int outid = 2*Nxyz + round(0.25/hx)*Ny + round(0.25/hy)*Nz;
  int outid = 2*Nxyz;
  PetscPrintf(PETSC_COMM_WORLD,"---The point (0.25, 0.25) is %d --- \n", outid);
  int Pos[2]={outid,outid+3*Nxyz};
  double ptValues[2];
  RetrieveVecPoints(x, 2,Pos, ptValues);
  PetscPrintf(PETSC_COMM_WORLD,"---The E fields at (0.25, 0.25) is ( %2.15f, %2.15f ) --- \n", ptValues[0],ptValues[1]);
#endif

/*------------------------------------------------------------*/


/* this yee_interp was coded by stevenj*/
#undef __FUNCT__ 
#define __FUNCT__ "yee_interp"
PetscErrorCode yee_interp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double x0, double y0, double z0,double x1, double y1, double z1, int Mx, int My, int Mz)
{
  Mat A;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;
  double xshift = -0.5 + Mx * 0.5 / (x1 - x0) / Nx;
  double yshift = -0.5 + My * 0.5 / (y1 - y0) / Ny;
  double zshift = -0.5 + Mz * 0.5 / (z1 - z0) / Nz;
  int i;
  int Nc = 3; //modified;

     
  ierr = MatCreateMPIAIJ(comm, PETSC_DECIDE, PETSC_DECIDE,
			 Nx*Ny*Nz*6, Mx*My*Mz,
			 nz, NULL, nz, NULL, &A); CHKERRQ(ierr);
     
  ierr = MatGetOwnershipRange(A, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    int ix, iy, iz, ic;
    double xd,yd,zd; /* (ix,iy,iz) location in d coordinates */
    int ixd,iyd,izd; /* rounded (xd,yd,zd) */
    int j, id;

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;
    ic = (j /= Nx) % Nc; // modifed, Nc = 3;

    xd = (ix*(1.0/Nx) - x0) * (Mx / (x1-x0)) + (ic == 0 ? xshift : -0.5);
    ixd = floor(xd + 0.5);
    if (ixd < 0 || ixd >= Mx) continue;
    yd = (iy*(1.0/Ny) - y0) * (My / (y1-y0)) + (ic == 1 ? yshift : -0.5);
    iyd = floor(yd + 0.5);
    if (iyd < 0 || iyd >= My) continue;
    zd = (iz*(1.0/Nz) - z0) * (Mz / (z1-z0)) + (ic == 2 ? zshift : -0.5);
    izd = floor(zd + 0.5);
    if (izd < 0 || izd >= Mz) continue;
	  
    id = (ixd*My + iyd)*Mz + izd;
    ierr = MatSetValue(A, i, id, 1.0, INSERT_VALUES);
    CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) A,
			    "Yee_interp_matrix"); CHKERRQ(ierr);
  *Aout = A;
  PetscFunctionReturn(0);
}



/*-- Same as Yee_interp, except now (Lx,Ly,Lz) indicating the index (Nx,Ny,Nz), instead of (1,1,1) --*/
#undef __FUNCT__ 
#define __FUNCT__ "General_interp"
PetscErrorCode General_interp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double hx, double hy, double hz, double x0, double y0, double z0, double x1, double y1, double z1,int Mx, int My, int Mz)
{
  // double hxeff = hx*(Nx-0.5)/Nx;
  //double hyeff = hy*(Ny-0.5)/Ny;
  //double hzeff = hz*(Nz-0.5)/Nz;

  double hxeff = hx;
  double hyeff = hy;
  double hzeff = hz;

  Mat A;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;
  double xshift = -0.5 + Mx * 0.5 / (x1 - x0) * hxeff;
  double yshift = -0.5 + My * 0.5 / (y1 - y0) * hyeff;
  double zshift = -0.5 + Mz * 0.5 / (z1 - z0) * hzeff;
  int i;
  int Nc = 3; //new modified;
     
  ierr = MatCreateMPIAIJ(comm, PETSC_DECIDE, PETSC_DECIDE,
			 Nx*Ny*Nz*6, Mx*My*Mz,
			 nz, NULL, nz, NULL, &A); CHKERRQ(ierr);
     
  ierr = MatGetOwnershipRange(A, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    int ix, iy, iz, ic;
    double xd,yd,zd; /* (ix,iy,iz) location in d coordinates */
    int ixd,iyd,izd; /* rounded (xd,yd,zd) */
    int j, id;

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;
    ic = (j /= Nx) % Nc;// new modified;

    xd = (ix*(hxeff) - x0) * (Mx / (x1-x0)) + (ic == 0 ? xshift : -0.5);
    ixd = floor(xd + 0.5);
    if (ixd < 0 || ixd >= Mx) continue;
    yd = (iy*(hyeff) - y0) * (My / (y1-y0)) + (ic == 1 ? yshift : -0.5);
    iyd = floor(yd + 0.5);
    if (iyd < 0 || iyd >= My) continue;
    zd = (iz*(hzeff) - z0) * (Mz / (z1-z0)) + (ic == 2 ? zshift : -0.5);
    izd = floor(zd + 0.5);
    if (izd < 0 || izd >= Mz) continue;
	  
    id = (ixd*My + iyd)*Mz + izd;
    ierr = MatSetValue(A, i, id, 1.0, INSERT_VALUES);
    CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) A,
			    "General_interp_matrix"); CHKERRQ(ierr);
  *Aout = A;
  PetscFunctionReturn(0);
}



/* On Dec 3rd 2011, I combine myinterpSlab into the case myinterp so that the new my interp handle both slab and nonslab case */
#undef __FUNCT__ 
#define __FUNCT__ "myinterpSlab"
PetscErrorCode myinterpSlab(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int Nxo, int Nyo, int Nzo, int Mx, int My, int Mz, int Mzslab)
{
  Mat A;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;
  double shift =  0.5;
  int i;
  int Nc = 3; //modified;

     
  ierr = MatCreateMPIAIJ(comm, PETSC_DECIDE, PETSC_DECIDE,
			 Nx*Ny*Nz*6, Mx*My*Mz,
			 nz, NULL, nz, NULL, &A); CHKERRQ(ierr);
     
  ierr = MatGetOwnershipRange(A, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    int ix, iy, iz, ic;
    double xd,yd,zd; /* (ix,iy,iz) location in d coordinates */
    int ixd,iyd,izd; /* rounded (xd,yd,zd) */
    int j, id;

    iz = (j = i) % Nz;
    iy = (j /= Nz) % Ny;
    ix = (j /= Ny) % Nx;
    ic = (j /= Nx) % Nc; // modifed, Nc = 3;

    xd = (ix-Nxo) + (ic!= 0)*shift;
    ixd = ceil(xd-0.5);
    if (ixd < 0 || ixd >= Mx) continue;
   
    yd = (iy-Nyo) + (ic!= 1)*shift;
    iyd = ceil(yd - 0.5);
    if (iyd < 0 || iyd >= My) continue;

    zd = (iz-Nzo) + (ic!= 2)*shift;
    izd = ceil(zd - 0.5);
    if (izd < 0 || izd >= Mzslab) continue; // modified for slab;
	  
    id = (ixd*My + iyd)*Mz ; // modified for slab;
    ierr = MatSetValue(A, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) A,
			    "Yee_interp_matrix"); CHKERRQ(ierr);

  *Aout = A;
  PetscFunctionReturn(0);
}


/*--------- another method to calculate derivatives--------------*/
#if 0
 /* Adjoint-Method tells us Mtran*lambda = -weight.*J= -weightedJ; therefore we have to solve M*conj(lambda) = -weightedJ; then the derivative is -conj(lambda).*[-omega^2*epspml*(1+i/Qabs)]*x); it is equivalent to solve M*z = weightedJ, and derivative is z*epspml*(1+i/Qabs)*omega^2*x = z*epspmlQ*x*omega^2; */
    Vec cglambda = ptmyfundata->Scglambda;
   int its2;
   ierr = KSPSolve(ksp,weightedJ,cglambda);CHKERRQ(ierr);
   ierr = KSPGetIterationNumber(ksp,&its2);CHKERRQ(ierr);
   ierr = PetscPrintf(PETSC_COMM_WORLD,"--- the number of Kryolv Iterations for Adjoint equation is %D----\n ",its2);CHKERRQ(ierr);
   int aconj=0;
   CmpVecProd(cglambda,epspmlQ,tmp,D,aconj,tmpa,tmpb);
   CmpVecProd(x,tmp,epsgrad,D,aconj,tmpa,tmpb); 
   VecScale(epsgrad,-pow(omega,2)*hx*hy); // the minus sign is because the quation we are solving is M z = -weightedJ; // the factor hx*hy is from 2D intergartion;
#endif
