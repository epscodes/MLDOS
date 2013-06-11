#include <petsc.h>

/*

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
#define __FUNCT__ "MoperatorGeneral"
PetscErrorCode MoperatorGeneral(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double hx, double hy, double hz, int bx[2], int by[2], int bz[2], double *muinv, int DimPeriod)
 /* bx[lo/hi = 0/1] = 0/1/-1: dirichlet/even/odd */
{
  Mat A;
  PetscErrorCode ierr;
  int Nc = 3, Nr = 2;
  int ns, ne;
  int i,j,k, ic;
  double h[3]={hx,hy,hz}, hh;
  int Nxyzc = Nx*Ny*Nz*Nc;
  int Nxyzcr = Nx*Ny*Nz*Nc*Nr;
  int b[3][2][3]; /* b[x/y/z direction][lo/hi][Ex/Ey/Ez] */
  int Nxyz[3]={Nx,Ny,Nz};
  double muinva, muinvl;


   /*-------------------------------------*/


  //ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, Nxyzcr, Nxyzcr, 26, NULL, 26, NULL, &A); CHKERRQ(ierr);
  MatCreate(comm, &A);
  MatSetType(A,MATMPIAIJ);
  MatSetSizes(A,PETSC_DECIDE, PETSC_DECIDE, Nxyzcr, Nxyzcr);
  MatMPIAIJSetPreallocation(A, 26, PETSC_NULL, 26, PETSC_NULL);

  ierr = MatGetOwnershipRange(A, &ns, &ne); CHKERRQ(ierr);
  

  // by default, I keep zero entries unless have ignore_zero_entries;
  PetscBool flg;
  ierr = PetscOptionsHasName(PETSC_NULL,"-ignore_zero_entries",&flg);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"the ignore_zero_entries option is %d \n",flg);
  if (flg) 
    {ierr = MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);}

  /* set up b ... */
 
  for(ic=0; ic<3; ic++)
    for(j=0; j<2; j++)
      for(k=0; k<3; k++){
	b[ic][j][k] =  ( (ic==0)*bx[j] + (ic==1)*by[j] + (ic==2)*bz[j])*( k==ic ? -1 :1);
      }
  
  for (i = ns; i < ne; ++i) {
    int ixyz[3], ic, ir, jr;
    int itmp;
    int cp1, cp2, icp1, icp2, cidu, cp1idu,cp1idl, cp2idu, cp2idl;
    double cid_usign, cp1id_usign, cp1id_lsign, cp2id_usign, cp2id_lsign;

	  
    ixyz[2] = (itmp = i) % Nz;
    ixyz[1] = (itmp /= Nz) % Ny;
    ixyz[0] = (itmp /= Ny) % Nx;
    ic = (itmp /= Nx) % Nc;
    ir = itmp / Nc;
	  
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
   

    cid_usign = 1.0;
    cp1id_usign = 1.0;
    cp1id_lsign = 1.0;
    cp2id_usign = 1.0;
    cp2id_lsign = 1.0;
    

    for(jr=0; jr<2; jr++) { /* column real/imag parts */
       int sign = (jr == 0) ? 1 : (ir == 0 ? -1 : 1); 
      int jrd =  (jr-ir)*Nxyzc;
      int jrdmu =  ((jr == 0) ? 0 : (ir == 0 ? 1 : -1))*Nxyzc;
     
             

      /* d/dy muinv d/dx Ey */

      if(ixyz[ic] == Nxyz[ic]-1)
	{
	  if(ic == (DimPeriod-1) || DimPeriod == 4 || (DimPeriod<0 && ic !=-(DimPeriod+1)) )
	    cidu = (1-Nxyz[ic])*cid; // periodic b.c; DimPeriod ==4 means period in all three dimensions;
	  else
	    {
	      cidu = 0;
	      cid_usign = b[ic][1][cp1];
	    }
	}
 
      muinva = muinv[icp2 + jrdmu]; 

      if(ixyz[cp1]==0)
	{
	  if(cp1 == (DimPeriod-1) || DimPeriod == 4 || (DimPeriod<0 && cp1!=-(DimPeriod+1)) )
	    {
	      cp1idl = (1-Nxyz[cp1])*cp1id;
	    }
	  else
	    {
	      cp1idl = 0;
	      cp1id_lsign = b[cp1][0][cp1];
	    }
	}

      muinvl = muinv[icp2 - cp1idl + jrdmu];

      hh = sign * h[ic]*h[cp1];

      ierr = MatSetValue(A,i, icp1 + cidu + jrd,  cid_usign*muinva/hh, ADD_VALUES); CHKERRQ(ierr);     
       ierr = MatSetValue(A,i, icp1 + jrd, -muinva/hh, ADD_VALUES); CHKERRQ(ierr);
       ierr = MatSetValue(A,i, icp1 + cidu - cp1idl + jrd, -cid_usign*cp1id_lsign*muinvl/ hh, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, icp1 - cp1idl + jrd, +cp1id_lsign*muinvl/ hh, ADD_VALUES); CHKERRQ(ierr);
	    
    
      /* - d/dy muinv d/dy Ex */

      muinva = muinv[icp2 + jrdmu];

      if(ixyz[cp1] == Nxyz[cp1]-1)
	{
	  if( cp1== (DimPeriod-1) || DimPeriod == 4 || (DimPeriod<0 && cp1!=-(DimPeriod+1)) )
	    cp1idu = (1-Nxyz[cp1])*cp1id;
	  else
	    {
	      cp1idu = 0;
	      cp1id_usign = b[cp1][1][ic];
	    }
	}
	       
      if(ixyz[cp1] == 0)
	{
	  if(cp1!= (DimPeriod-1) && DimPeriod !=4 && ( DimPeriod >=0 || cp1 == -(DimPeriod+1)) )
	    {
	      cp1idl = -cp1idu; 
	      cp1id_lsign = b[cp1][0][ic];
	      muinvl = muinva;
	    }
	  else
	    {
	      cp1idl = (1-Nxyz[cp1])*cp1id;
	      muinvl = muinv[icp2-cp1idl+jrdmu];
	    }
	 }
      else
	{
	  muinvl = muinv[icp2-cp1idl+jrdmu];
	}


      hh = sign * h[cp1]*h[cp1];
      ierr = MatSetValue(A,i, i + cp1idu + jrd , -cp1id_usign * muinva/hh, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, i + jrd , +(muinva + muinvl)/hh, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, i - cp1idl + jrd , - cp1id_lsign * muinvl/hh, ADD_VALUES); CHKERRQ(ierr);


      /* d/dz muinv d/dx Ez */
      
      if(ixyz[ic] == Nxyz[ic]-1)
	{
	  if(ic == (DimPeriod-1) || DimPeriod == 4 || (DimPeriod<0 && ic!=-(DimPeriod+1)) )
	    cidu = (1-Nxyz[ic])*cid;
	  else
	    {
	      cidu = 0;
	      cid_usign = b[ic][1][cp2];
	    }
	}

      muinva = muinv[icp1 + jrdmu];

      if(ixyz[cp2] == 0)
	{

	  if(cp2 == (DimPeriod-1) || DimPeriod == 4 || (DimPeriod<0 && cp2!=-(DimPeriod+1)) )
	    cp2idl = (1-Nxyz[cp2])*cp2id;
	  else
	    {
	      cp2idl = 0; 
	      cp2id_lsign = b[cp2][0][cp2];		   
	    }
	}

       muinvl = muinv[icp1 - cp2idl + jrdmu];

      hh = sign * h[ic]*h[cp2];
      ierr = MatSetValue(A,i, icp2 + cidu + jrd, cid_usign * muinva/hh, ADD_VALUES); CHKERRQ(ierr);          ierr = MatSetValue(A,i, icp2 + jrd, -muinva/hh, ADD_VALUES); CHKERRQ(ierr); 
      ierr = MatSetValue(A,i, icp2 + cidu - cp2idl + jrd, - cid_usign * cp2id_lsign * muinvl/hh, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, icp2 - cp2idl + jrd, + cp2id_lsign*muinvl/hh, ADD_VALUES); CHKERRQ(ierr);
	       
      /* - d/dz muinv d/dz Ex */
	 
      muinva = muinv[icp1 + jrdmu];
      if(ixyz[cp2] == Nxyz[cp2]-1)
	{
	  if (cp2== (DimPeriod-1) || DimPeriod == 4 || (DimPeriod<0 && cp2 !=-(DimPeriod+1)) )
	    cp2idu = (1-Nxyz[cp2])*cp2id;
	  else
	    {
	      cp2idu = 0;
	      cp2id_usign = b[cp2][1][ic];
	    }
	}
	       

      if(ixyz[cp2] == 0)
	{
	  if ( cp2!= (DimPeriod-1) && DimPeriod !=4 && ( DimPeriod >=0 || cp2 == -(DimPeriod+1)) )
	    {
	      cp2idl = -cp2idu;  
	      cp2id_lsign = b[cp2][0][ic];
	      muinvl = muinva;
	    }
	  else
	    {
	      cp2idl = (1-Nxyz[cp2])*cp2id;
	      muinvl = muinv[icp1 - cp2idl + jrdmu]; 
	    }
	}
      else
	{
	  muinvl = muinv[icp1 - cp2idl + jrdmu];
	}

      hh = sign * h[cp2]*h[cp2];

      ierr = MatSetValue(A,i, i + cp2idu + jrd, -cp2id_usign*muinva/hh, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, i + jrd, +(muinva + muinvl)/hh, ADD_VALUES); CHKERRQ(ierr);
      ierr = MatSetValue(A,i, i - cp2idl + jrd, -cp2id_lsign* muinvl/hh, ADD_VALUES); CHKERRQ(ierr);

      /*---add tiny number to diagonals to keep nonzero positions on diagonal for future---*/
      if (flg && (jrd!=0))      
	ierr=MatSetValue(A,i,i+jrd,1e-125,ADD_VALUES);CHKERRQ(ierr);
    }

  }
   
      /*---------------------------*/
     
     ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
     ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

     ierr = PetscObjectSetName((PetscObject) A,
			       "InitialMOpGeneral"); CHKERRQ(ierr);

     *Aout = A;
     PetscFunctionReturn(0);
}

