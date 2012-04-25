#include <stdio.h>
#include <math.h>
#include <petsc.h>


#undef __FUNCT__ 
#define __FUNCT__ "CmpVecProd"
PetscErrorCode CmpVecProd(Vec va, Vec vb, Vec vout, Mat D, int aconj, Vec vai, Vec vbi)
{
  PetscErrorCode ierr;

  int N; // total length;
  ierr=VecGetSize(va, &N); CHKERRQ(ierr);

  ierr=MatMult(D,va,vai);CHKERRQ(ierr);
  ierr=MatMult(D,vb,vbi);CHKERRQ(ierr);

  double *a, *b, *ai, *bi, *out;
  ierr=VecGetArray(va,&a);CHKERRQ(ierr);
  ierr=VecGetArray(vb,&b);CHKERRQ(ierr);
  ierr=VecGetArray(vai,&ai);CHKERRQ(ierr);
  ierr=VecGetArray(vbi,&bi);CHKERRQ(ierr);
  ierr=VecGetArray(vout,&out);CHKERRQ(ierr);

  int i, ns, ne, nlocal;
  ierr = VecGetOwnershipRange(vout, &ns, &ne);
  nlocal = ne-ns;

  int sign = pow(-1,aconj);

  
  for (i=0; i<nlocal; i++)
    {  
      if(i<(N/2-ns)) // N is the total length of Vec;
	out[i] = a[i]*b[i] - sign*ai[i]*bi[i];
      else
	out[i] = ai[i]*b[i] + sign*a[i]*bi[i];
    }
  
  ierr=VecRestoreArray(va,&a);CHKERRQ(ierr);
  ierr=VecRestoreArray(vb,&b);CHKERRQ(ierr);
  ierr=VecRestoreArray(vai,&ai);CHKERRQ(ierr);
  ierr=VecRestoreArray(vbi,&bi);CHKERRQ(ierr);
  ierr=VecRestoreArray(vout,&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "ImagIMat"
PetscErrorCode ImagIMat(MPI_Comm comm, Mat *Dout, int N)
{
  Mat D;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;  
  int i;
     
  ierr = MatCreateMPIAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, N,N,nz, NULL, nz, NULL, &D); CHKERRQ(ierr); // here N is total length;
      
  ierr = MatGetOwnershipRange(D, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    int id = (i+N/2)%(N);
    double sign = pow(-1.0, (i<N/2));
    ierr = MatSetValue(D, i, id, sign, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) D, "ImaginaryIMatrix"); CHKERRQ(ierr);
  
  *Dout = D;
  PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "CongMat"
PetscErrorCode CongMat(MPI_Comm comm, Mat *Cout, int N)
{
  Mat C;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;  
  int i;
     
  ierr = MatCreateMPIAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, N,N,nz, NULL, nz, NULL, &C); CHKERRQ(ierr);
     
  ierr = MatGetOwnershipRange(C, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    double sign = pow(-1.0, (i>(N/2-1)));
    ierr = MatSetValue(C, i, i, sign, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) C,"CongMatrix"); CHKERRQ(ierr);
  
  *Cout = C;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetWeightVec"
PetscErrorCode GetWeightVec(Vec weight,int Nx, int Ny, int Nz)
{
   PetscErrorCode ierr;
   int i, j, ns, ne, ix, iy, iz, ic;
   double value, tmp;

   int Nc = 3;
   ierr = VecGetOwnershipRange(weight,&ns,&ne); CHKERRQ(ierr);
   
   if (Nx == 1 || Ny == 1 || Nz == 1)
     {
       tmp = 4.0;
       PetscPrintf(PETSC_COMM_WORLD,"---Caution! Treat as a 2D problem and Weight is divieded by 2 \n");
     }
   else
     tmp = 8.0;

   for(i=ns; i<ne; i++)
     {
       iz = (j = i) % Nz;
       iy = (j /= Nz) % Ny;
       ix = (j /= Ny) % Nx;
       ic = (j /= Nx) % Nc;       

       value = tmp; // tmp = 8.0 for 3D and 4.0 for 2D

       if(ic==0)
	 value = value/(((iy==0)+1.0)*((iz==0)+1.0));
       if(ic==1)
	 value = value/(((ix==0)+1.0)*((iz==0)+1.0));
       if(ic==2)
	 value = value/(((ix==0)+1.0)*((iy==0)+1.0));

       VecSetValue(weight, i, value, INSERT_VALUES);
     }
   ierr = VecAssemblyBegin(weight); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(weight); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "GetMediumVec"
PetscErrorCode GetMediumVec(Vec epsmedium,int Nz, int Mz, double epsair, double epssub)
{
   PetscErrorCode ierr;
   int i, iz, ns, ne;
   double value;
   ierr = VecGetOwnershipRange(epsmedium,&ns,&ne); CHKERRQ(ierr);
   for(i=ns;i<ne; i++)
     {
       iz = i%Nz;
       if (iz<Mz)
	 value = epsair;
       else
	 value = epssub;
        
       VecSetValue(epsmedium, i, value, INSERT_VALUES);

       }

   ierr = VecAssemblyBegin(epsmedium); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(epsmedium); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "GetRealPartVec"
PetscErrorCode GetRealPartVec(Vec vR, int N)
{
   PetscErrorCode ierr;
   int i, ns, ne;

   ierr = VecGetOwnershipRange(vR,&ns,&ne); CHKERRQ(ierr);

   for(i=ns; i<ne; i++)
     {
       if (i<N/2)
	 VecSetValue(vR,i,1.0,INSERT_VALUES);
       else
	 VecSetValue(vR,i,0.0,INSERT_VALUES);
     }

   ierr = VecAssemblyBegin(vR); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(vR); CHKERRQ(ierr);
   
   PetscFunctionReturn(0);
  
}

PetscErrorCode  ArrayToVec(double *pt, Vec V)
{
  PetscErrorCode ierr;
  int j, ns, ne;

  ierr = VecGetOwnershipRange(V,&ns,&ne);
   for(j=ns;j<ne;j++)
    { ierr=VecSetValue(V,j,pt[j],INSERT_VALUES); 
      CHKERRQ(ierr);
    }

  ierr = VecAssemblyBegin(V); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(V);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode VecToArray(Vec V, double *pt, VecScatter scatter, IS from, IS to, Vec Vlocal, int Mxyz)
{
  PetscErrorCode ierr;

 // scatter V to Vlocal;
    ierr =VecScatterCreate(V,from,Vlocal,to,&scatter); CHKERRQ(ierr);
    VecScatterBegin(scatter,V,Vlocal,INSERT_VALUES,SCATTER_FORWARD);
   VecScatterEnd(scatter,V,Vlocal,INSERT_VALUES,SCATTER_FORWARD);
   ierr =VecScatterDestroy(scatter); CHKERRQ(ierr);

   // copy from vgradlocal to grad;
   double *ptVlocal;
   ierr =VecGetArray(Vlocal,&ptVlocal);CHKERRQ(ierr);

   int i;
   for(i=0;i<Mxyz;i++)
     pt[i] = ptVlocal[i];
   ierr =VecRestoreArray(Vlocal,&ptVlocal);CHKERRQ(ierr);   
  PetscFunctionReturn(0);
}




#undef __FUNCT__ 
#define __FUNCT__ "AddMuAbsorption"
PetscErrorCode AddMuAbsorption(double *muinv, Vec muinvpml, double Qabs, int add)
{
  //compute muinvpml/(1+i/Qabs)
  double Qinv = (add==0) ? 0.0: (1.0/Qabs);
  double d=1 + pow(Qinv,2);
  PetscErrorCode ierr;
  int N;
  ierr=VecGetSize(muinvpml,&N);CHKERRQ(ierr);

  double *ptmuinvpml;
  ierr=VecGetArray(muinvpml, &ptmuinvpml);CHKERRQ(ierr);

  int i;
  double a,b;
  for(i=0;i<N/2;i++)
    {
      a=ptmuinvpml[i];
      b=ptmuinvpml[i+N/2];      
      muinv[i]= (a+b*Qinv)/d;
      muinv[i+N/2]=(b-a*Qinv)/d;
    }
  ierr=VecRestoreArray(muinvpml,&ptmuinvpml);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef _FUNCT_
#define _FUNCT_ "TMprojmat"
PetscErrorCode TMprojmat(MPI_Comm comm, Mat *TMout, int Nxyz)
{
  Mat TM;
  int nnz = 1;
  PetscErrorCode ierr;
  int i, ns, ne;
  ierr = MatCreateMPIAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, 2*Nxyz, 6*Nxyz, nnz, NULL, nnz, NULL, &TM); CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(TM, &ns, &ne); CHKERRQ(ierr);

  int id;
  for(i=ns; i<ne; i++)
    {
      id = (ns<Nxyz)*(i+2*Nxyz) + (ns>=Nxyz)*(i+4*Nxyz);//note "4", not "5"
      ierr = MatSetValue(TM, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
    }

  ierr = MatAssemblyBegin(TM, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(TM, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  
  ierr = PetscObjectSetName((PetscObject) TM, "TMmatrix"); CHKERRQ(ierr);

  *TMout = TM;
  PetscFunctionReturn(0);
}


#undef _FUNCT_
#define _FUNCT_ "MatSetTwoDiagonals"
/* [M(1,1) + sign*epsC(1), M(1,2) - sign*epsC(2);
   M(2,1) + sign*epsC(2), M(2,2) + sign*epsC(1);] */
PetscErrorCode MatSetTwoDiagonals(Mat M, Vec epsC, Mat D, double sign)
{
  PetscErrorCode ierr;
  
  Vec epsCi;
  ierr=VecDuplicate(epsC, &epsCi); CHKERRQ(ierr);
  ierr=MatMult(D,epsC,epsCi); CHKERRQ(ierr);
  
  int N;
  ierr=VecGetSize(epsC,&N); CHKERRQ(ierr);
    
  int i, ns, ne;
  MatGetOwnershipRange(M, &ns, &ne); CHKERRQ(ierr);

  double *c, *ci;
  ierr = VecGetArray(epsC, &c); CHKERRQ(ierr);  
  ierr = VecGetArray(epsCi, &ci); CHKERRQ(ierr);

  double vr, vi;
  
  for (i = ns; i < ne; ++i) 
    {
      if(i<N/2)
	{ vr = c[i-ns];  // here vr is real;
	  vi = ci[i-ns]; // here vi is -imag; happen to be correct combination;
	}
      else
	{ vr = ci[i-ns]; // here vr is real;
	  vi = c[i-ns];  // here vr is imag;
	}
      
      //M = M + sign*epsC
      ierr = MatSetValue(M,i,i,sign*vr,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(M,i,(i+N/2)%N,sign*vi,ADD_VALUES);CHKERRQ(ierr);
    }
  
  ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  
  ierr = VecRestoreArray(epsC, &c); CHKERRQ(ierr);
  ierr = VecRestoreArray(epsCi, &ci); CHKERRQ(ierr);
  
 /* Destroy Vectors */
  ierr=VecDestroy(epsCi); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "CmpVecScale"
PetscErrorCode CmpVecScale(Vec vin, Vec vout, double a, double b, Mat D, Vec vini)
{
  PetscErrorCode ierr; 
  ierr=MatMult(D,vin,vini);CHKERRQ(ierr);
  VecAXPBYPCZ(vout,a,b,1.0, vin,vini); 
  PetscFunctionReturn(0);
}
