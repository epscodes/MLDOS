#include <petsc.h>
#include <complex.h>
#include <math.h>
typedef double complex dcomp;

extern int Nxyz;

/*--MoperatorGeneralBloch 2D codes only involving Ez, which handles the 2D TM case */

/*--MoperatorGeneralBloch contains -epsilon*omega^2 term--- different from MoperatorGeneral codes ----*/

/*
Generalize the MoperatorGeneral to handle the Bloch Boundary Condition;

This rountine generates sparse matrix for the operator Curl \times 1/mu \times Curl (finite difference with Yee grid for a cubic domain):

Input parameters: Nx, Ny, Nz (grid resolution)
                  hx, hy, hz (step-size)
		  bx[2], by[2], bz[2] ( boundary conditions: bx[lo/hi = 0/1] = 0/1/-1: dirichlet/even/odd; where lo/hi: lower/upper boundary;)
		  muinvvec ( 1/mu, the inverse of the permeability of the material; stored in a row-major order: muinvvec = [muinv_x_real; muinv_y_real; muinv_z_real; muinv_x_imag; muinv_y_imag; muinv_z_imag].
		  DimPeriod (for periodic boundary conditions)
		  DimPeriod = 1/2/3 for Periodic in x/y/z directions; 4 for all three directions 
		  DimPeriod = -1/-2/-3 for Periodic in non-x/non-y/non-z directions; (namely, -1 means periodic in both y and z direction, but not x direction) 
		  DimPeriod = 0 for non-periodic in all three directions


Output parameter: sparse matrix M.

 */


#undef __FUNCT__ 
#define __FUNCT__ "MoperatorGeneralBloch2D"
PetscErrorCode MoperatorGeneralBloch2D(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double hx, double hy, double hz, int bx[2], int by[2], int bz[2], double *muinv, int DimPeriod, double blochbc[3], Vec epsOmegasqr, Vec epsOmegasqri)
 /* bx[lo/hi = 0/1] = 0/1/-1: dirichlet/even/odd */
{
  Mat A;
  PetscErrorCode ierr;
  int Nc = 3;// Nr = 2;
  int ns, ne;
  int i,j,k, ic;
  double h[3]={hx,hy,hz}, hh;
  //int Nxyzc = Nx*Ny*Nz*Nc;
  //int Nxyzcr = Nx*Ny*Nz*Nc*Nr;
  int b[3][2][3]; /* b[x/y/z direction][lo/hi][Ex/Ey/Ez] */
  int Nxyzarray[3]={Nx,Ny,Nz};

  dcomp muinvcp1cmp, muinvcp1lcmp, muinvcp2cmp, muinvcp2lcmp;
  dcomp cidu_phase, cp1idu_phase, cp1idl_phase, cp2idu_phase, cp2idl_phase;
  double c1, c2, c3;
   /*-------------------------------------*/


  //ierr = MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, 2*Nxyz, 2*Nxyz, 10, NULL, 10, NULL, &A); CHKERRQ(ierr);

  MatCreate(comm, &A);
  MatSetType(A,MATMPIAIJ);
  MatSetSizes(A,PETSC_DECIDE, PETSC_DECIDE, 2*Nxyz, 2*Nxyz);
  MatMPIAIJSetPreallocation(A, 10, PETSC_NULL, 10, PETSC_NULL);

  ierr = MatGetOwnershipRange(A, &ns, &ne); CHKERRQ(ierr);
  

  // by default, I keep zero entries unless have ignore_zero_entries;
  PetscBool flg;
  ierr = PetscOptionsHasName(PETSC_NULL,"-ignore_zero_entries",&flg);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_WORLD,"the ignore_zero_entries option is %d \n",flg);
  if (flg) 
    {ierr = MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);}

  /* set up b ... */
 
  for(ic=0; ic<3; ic++)
    for(j=0; j<2; j++)
      for(k=0; k<3; k++){
	b[ic][j][k] =  ( (ic==0)*bx[j] + (ic==1)*by[j] + (ic==2)*bz[j])*( k==ic ? -1 :1);
      }

  /* convert epsOmegasqr to array */
  double *c, *ci;
  ierr = VecGetArray(epsOmegasqr, &c); CHKERRQ(ierr);
  ierr = VecGetArray(epsOmegasqri, &ci); CHKERRQ(ierr);


  for (i = ns; i < ne; ++i) {
    int ixyz[3], ic, ir, jr;
    int itmp;
    int cp1, cp2, icp1, icp2, cidu, cp1idu,cp1idl, cp2idu, cp2idl;
    	  
    ixyz[2] = (itmp = i) % Nz;
    ixyz[1] = (itmp /= Nz) % Ny;
    ixyz[0] = (itmp /= Ny) % Nx;
    ic= 2;//ic = (itmp /= Nx) % Nc;
    ir=itmp/Nx; //ir = itmp / Nc;
	  
    cp1 = (ic + 1) % Nc;
    cp2 = (ic + 2) % Nc;
    icp1 = i + (cp1 - ic) * (Nx*Ny*Nz);
    icp2 = i + (cp2 - ic) * (Nx*Ny*Nz);

    cidu = (ic==0)*Ny*Nz + (ic==1)*Nz + (ic==2);

    cp1idu = (ic==2)*Ny*Nz + (ic==0)*Nz + (ic==1);
    cp1idl = (ic==2)*Ny*Nz + (ic==0)*Nz + (ic==1);
    cp2idu = (ic==1)*Ny*Nz + (ic==2)*Nz + (ic==0);
    cp2idl = (ic==1)*Ny*Nz + (ic==2)*Nz + (ic==0);

   
    int cid, cp1id, cp2id;
    cid= (ic==0)*Ny*Nz + (ic==1)*Nz + (ic==2);
    cp1id = (ic==2)*Ny*Nz + (ic==0)*Nz + (ic==1);
    cp2id = (ic==1)*Ny*Nz + (ic==2)*Nz + (ic==0);
   

    cidu_phase = 1.0;
    cp1idu_phase = 1.0;
    cp1idl_phase = 1.0;
    cp2idu_phase = 1.0;
    cp2idl_phase = 1.0;
    
    for(jr=0; jr<2; jr++) { /* column real/imag parts */
      int jrd =  (jr-ir)*Nxyz;
               
      dcomp magicnum = (ir==jr)*1 + (ir<jr)*I + (ir>jr)*(-I); 

      /* d/dy muinv d/dx Ey */

      
      /* - d/dy muinv d/dy Ex */
      // it shares same muinvcp2cmp and muinvcp2lcmp as d/dx muinv d/dx Ey;

      if(ixyz[cp1] == Nxyzarray[cp1]-1)
	{
	  if( cp1== (DimPeriod-1) || DimPeriod == 4 || (DimPeriod<0 && cp1!=-(DimPeriod+1)) )
	    {
	      cp1idu = (1-Nxyzarray[cp1])*cp1id;
	      cp1idu_phase = cos(blochbc[cp1]) + I*sin(blochbc[cp1]);
	    }
	  else
	    {
	      cp1idu = 0;
	      cp1idu_phase = b[cp1][1][ic];
	    }
	}
	       
      if(ixyz[cp1] == 0)
	{
	  if(cp1!= (DimPeriod-1) && DimPeriod !=4 && ( DimPeriod >=0 || cp1 == -(DimPeriod+1)) )
	    {
	      cp1idl = -cp1idu; 
	      cp1idl_phase = b[cp1][0][ic];
	    }
	  else
	    {
	      cp1idl = (1-Nxyzarray[cp1])*cp1id;
	      cp1idl_phase = cos(blochbc[cp1]) - I*sin(blochbc[cp1]);
	    }
	 }
   
      //muinvcp2cmp = muinv[icp2%Nxyzc] + I*muinv[icp2%Nxyzc + Nxyzc];
      //muinvcp2lcmp = muinv[(icp2 - cp1idl)%Nxyzc] + I*muinv[(icp2 - cp1idl)%Nxyzc + Nxyzc];
      muinvcp2cmp = muinv[ i%Nxyz + Nxyz] + I*muinv[i%Nxyz + 4*Nxyz]; // muyijk
      muinvcp2lcmp = muinv[i%Nxyz - cp1idl + Nxyz] + I*muinv[i%Nxyz -cp1idl + 4*Nxyz]; //myuim1jk
  
      hh =  h[cp1]*h[cp1];
      c1 = -creal(cp1idu_phase * muinvcp2cmp * magicnum)/hh;
      c2 = +creal( (muinvcp2cmp + muinvcp2lcmp) * magicnum)/hh;
      c3 = -creal(cp1idl_phase * muinvcp2lcmp * magicnum)/hh;
     

      ierr = MatSetValue(A,i, i + cp1idu + jrd , c1, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, i + jrd , c2, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, i - cp1idl + jrd , c3, ADD_VALUES); CHKERRQ(ierr);

      /* d/dz muinv d/dx Ez */
             
      /* - d/dz muinv d/dz Ex */
      // it shares same muinvcp1cmp and muinvcp1lcmp as d/dz muinv d/dx Ez;
     
      if(ixyz[cp2] == Nxyzarray[cp2]-1)
	{
	  if (cp2== (DimPeriod-1) || DimPeriod == 4 || (DimPeriod<0 && cp2 !=-(DimPeriod+1)) )
	    {
	      cp2idu = (1-Nxyzarray[cp2])*cp2id;
	      cp2idu_phase = cos(blochbc[cp2]) + I * sin(blochbc[cp2]);
	    }
	  else
	    {
	      cp2idu = 0;
	      cp2idu_phase = b[cp2][1][ic];
	    }
	}
	       

      if(ixyz[cp2] == 0)
	{
	  if ( cp2!= (DimPeriod-1) && DimPeriod !=4 && ( DimPeriod >=0 || cp2 == -(DimPeriod+1)) )
	    {
	      cp2idl = -cp2idu;  
	      cp2idl_phase = b[cp2][0][ic];
	    }
	  else
	    {
	      cp2idl = (1-Nxyzarray[cp2])*cp2id;
	      cp2idl_phase = cos(blochbc[cp2]) - I*sin(blochbc[cp2]);
	    }
	}
   

      hh = h[cp2]*h[cp2];
      //muinvcp1cmp = muinv[icp1%Nxyzc] + I*muinv[icp1%Nxyzc + Nxyzc];
      //muinvcp1lcmp = muinv[(icp1-cp2idl)%Nxyzc] + I*muinv[(icp1-cp2idl)%Nxyzc + Nxyzc];  

      muinvcp1cmp =  muinv[ i%Nxyz + 2*Nxyz] + I*muinv[i%Nxyz + 5*Nxyz]; // muyijk
      muinvcp1lcmp = muinv[ i%Nxyz - cp2idl + 2*Nxyz] + I*muinv[i%Nxyz -cp2idl + 5*Nxyz]; //myuim1jk


      c1 = -creal(muinvcp1cmp * cp2idu_phase * magicnum)/hh;
      c2 = +creal((muinvcp1cmp + muinvcp1lcmp)*magicnum)/hh;
      c3 = -creal(muinvcp1lcmp * cp2idl_phase * magicnum)/hh;
   
      ierr = MatSetValue(A,i, i + cp2idu + jrd, c1, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, i + jrd, c2, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, i - cp2idl + jrd, c3, ADD_VALUES); CHKERRQ(ierr);

      /*---add tiny number to diagonals to keep nonzero positions on diagonal for future---*/
      if (flg && (jrd!=0))      
	ierr=MatSetValue(A,i,i+jrd,1e-125,ADD_VALUES);CHKERRQ(ierr);
    }

    /* M = M - esp*omega^2; add -epsomega^2 in the diagonals */
      double vr, vi;
      if (ir==0)
	{ 
	  vr = c[i-ns];
	  vi = -ci[i-ns];
	}
      else
	{
	   vr = ci[i-ns];
	   vi = c[i-ns];
	}
      ierr = MatSetValue(A,i,i,-vr,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(A,i,(i+Nxyz)%(2*Nxyz), (1.0*pow(-1,i/(Nxyz)))*1.0*vi,ADD_VALUES); CHKERRQ(ierr);  

  }
   
      /*---------------------------*/
     
     ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
     ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

     ierr = VecRestoreArray(epsOmegasqr, &c); CHKERRQ(ierr);
     ierr = VecRestoreArray(epsOmegasqri, &ci); CHKERRQ(ierr);

     ierr = PetscObjectSetName((PetscObject) A,
			       "InitialMOpGeneral"); CHKERRQ(ierr);

     *Aout = A;
     PetscFunctionReturn(0);
}

