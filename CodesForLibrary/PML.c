#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <petsc.h>


double sqr(double a)
{
  return pow(a,2);
}

#undef __FUNCT__ 
#define __FUNCT__ "pmlsigma"
double pmlsigma(double RRT, double d)
{
  if (d==0)
    return 0; // if pml thickness is 0, return sigma=0;
  else
    return -3*log(RRT)/(4*d);
}

/* With Nx*Ny*Nz grid, pml is Npmlx, Npmly, Npmlz in each direction; The PML is defined at (i,j,k), where E is (i,j,k) + 0.5*ic, and H is (i+0.5,j+0.5,k+0.5) - 0.5*ic; */

/* EpsPMLFull and MuinvPMLFull mean that we added PML on both lower and upper boundaries. Later, we may only add PML on the upper boundary, while use even/odd BCs on the lower boundary, where PML is not needed.*/

#undef __FUNCT__ 
#define __FUNCT__ "EpsPMLFull"
PetscErrorCode EpsPMLFull(MPI_Comm comm, Vec epspml, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML)
{  
  PetscErrorCode ierr;
  int Nc = 3;
  int i, ns, ne;
  double sigma[3]={sigmax,sigmay,sigmaz};
  double Npml[3]={Npmlx,Npmly,Npmlz};
  int Nxyz[3]={Nx,Ny,Nz};
 
  VecGetOwnershipRange(epspml, &ns, &ne);

  for (i = ns; i < ne; ++i) {
    int ixyz[3], ic, ir;
    int itmp;
    int cp1, cp2;
    double dic, dcp1, dcp2, epsijk;
	  
    ixyz[2] = (itmp = i) % Nz;
    ixyz[1] = (itmp /= Nz) % Ny;
    ixyz[0] = (itmp /= Ny) % Nx;
    ic = (itmp /= Nx) % Nc;
    ir = itmp / Nc;
	  
    cp1 = (ic + 1) % Nc;
    cp2 = (ic + 2) % Nc;

    int npmlic, npmlcp1, npmlcp2;

    npmlic = Npml[ic];
    npmlcp1 = Npml[cp1];
    npmlcp2 = Npml[cp2];


    if (npmlic==0)
      dic = 0;
    else
      dic =  (LowerPML)* (ixyz[ic] < Npml[ic]) * (Npml[ic]-ixyz[ic]-0.5)/npmlic 
	+ ( ixyz[ic] > (Nxyz[ic]-Npml[ic]-1-1))*(ixyz[ic]-(Nxyz[ic]-Npml[ic]-1-1) -0.5 )/npmlic ;
  /* first -1 is for the postion of E, second -1 is for C index from 0 */
    /* I had same double -1 in the implementation for mu, but did not have it here; added it Apr 20 */


    //ierr=PetscPrintf(comm,"the dic value is %f \n",dic);
    if (npmlcp1==0)
      dcp1=0;
    else
      dcp1 =  (LowerPML)* (ixyz[cp1] < Npml[cp1] ) * (Npml[cp1]-ixyz[cp1])/npmlcp1	+ ( ixyz[cp1] > (Nxyz[cp1]-Npml[cp1]-1))* (ixyz[cp1] - (Nxyz[cp1]-Npml[cp1]-1))/npmlcp1;
    
    if (npmlcp2==0)
      dcp2=0;
    else
      dcp2 =  (LowerPML)* (ixyz[cp2] < Npml[cp2] ) * (Npml[cp2]-ixyz[cp2])/npmlcp2	+ (ixyz[cp2] > (Nxyz[cp2]-Npml[cp2]-1))* (ixyz[cp2] - (Nxyz[cp2]-Npml[cp2]-1))/npmlcp2;

   

    if(ir==0) // real part of epspml[ijk], actually,same as muinv[ijk]
      {	 
	epsijk = 1 + 1/sqr(omega)* 
	  (    sigma[ic]*sigma[cp1] * sqr(dic*dcp1) 
	       +  sigma[ic]*sigma[cp2] * sqr(dic*dcp2)
	       -  sigma[cp1]*sigma[cp2] * sqr(dcp1*dcp2)
	       );
      }
    else //imaginary part of epspml[ijk], just sign difference with muinv[ijk]
      {
	epsijk =(-sigma[ic]*sqr(dic) + sigma[cp1]*sqr(dcp1) + sigma[cp2]*sqr(dcp2)
		    + sigma[ic]*sigma[cp1]*sigma[cp2]/sqr(omega) * sqr(dic*dcp1*dcp2))/omega;
      }
    /* taking care of the denominator from multipling complex conjugate */
    epsijk /= (1 + sqr(sigma[ic]/omega*sqr(dic))) ;

    //ierr=PetscPrintf(comm,"the epsijk value is %f \n", epsijk);
    VecSetValue(epspml,i,epsijk,INSERT_VALUES);


    }
       /*---------------------------*/
     
     ierr = VecAssemblyBegin(epspml); CHKERRQ(ierr);
     ierr = VecAssemblyEnd(epspml); CHKERRQ(ierr);

     PetscFunctionReturn(0);

}




#undef __FUNCT__ 
#define __FUNCT__ "MuinvPMLFull"
PetscErrorCode MuinvPMLFull(MPI_Comm comm, Vec *muinvout, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML)
{
  Vec muinv;
  PetscErrorCode ierr;
  int Nc = 3, Nr = 2;
  int i, ns, ne;
  double sigma[3]={sigmax,sigmay,sigmaz};
  double Npml[3]={Npmlx,Npmly,Npmlz};
  int Nxyz[3]={Nx,Ny,Nz};

  int Nxyzc = Nx*Ny*Nz*Nc;
  int Nxyzcr = Nxyzc*Nr;
  ierr = VecCreate(comm,&muinv);CHKERRQ(ierr);
  ierr = VecSetSizes(muinv, PETSC_DECIDE, Nxyzcr);CHKERRQ(ierr);
  ierr = VecSetFromOptions(muinv);CHKERRQ(ierr);
  
  VecGetOwnershipRange(muinv, &ns, &ne);


  for (i = ns; i < ne; ++i) {
    int ixyz[3], ic, ir;
    int itmp;
    int cp1, cp2;
    double dic, dcp1, dcp2, muinvijk;
	  
    ixyz[2] = (itmp = i) % Nz;
    ixyz[1] = (itmp /= Nz) % Ny;
    ixyz[0] = (itmp /= Ny) % Nx;
    ic = (itmp /= Nx) % Nc;
    ir = itmp / Nc;
	  
    cp1 = (ic + 1) % Nc;
    cp2 = (ic + 2) % Nc;

    double npmlic, npmlcp1, npmlcp2;

    npmlic = Npml[ic];
    npmlcp1 = Npml[cp1];
    npmlcp2 = Npml[cp2];

    if (npmlic==0)
      dic = 0;
    else
      dic = (LowerPML)* (ixyz[ic] < Npml[ic] ) * (Npml[ic]-ixyz[ic])/npmlic
	+ (ixyz[ic] > (Nxyz[ic]-Npml[ic]-1))* (ixyz[ic] - (Nxyz[ic]-Npml[ic]-1))/npmlic;
    //ierr=PetscPrintf(comm,"the dic value in mu is %f \n", dic);


    if (npmlcp1==0)
      dcp1 = 0;
    else
      dcp1 =  (LowerPML)* (ixyz[cp1] < Npml[cp1] ) * (Npml[cp1]-ixyz[cp1]-0.5)/npmlcp1	+ ( ixyz[cp1] > (Nxyz[cp1]-Npml[cp1]-1-1))* (ixyz[cp1] - (Nxyz[cp1]-Npml[cp1]-1-1) -0.5)/npmlcp1;
    /* first -1 is for the postion of H, second -1 is for C index from 0 */

    if (npmlcp2==0)
      dcp2 = 0;
    else
      dcp2 =  (LowerPML)* (ixyz[cp2] < Npml[cp2] ) * (Npml[cp2]-ixyz[cp2]-0.5)/npmlcp2 	+ (ixyz[cp2] > (Nxyz[cp2]-Npml[cp2]-1-1))* (ixyz[cp2] - (Nxyz[cp2]-Npml[cp2]-1-1) -0.5)/npmlcp2;


    if(ir==0) // real part of mu[ijk]
      {	 
	muinvijk = 1 + 1/sqr(omega)* 
	  (    sigma[ic]*sigma[cp1] * sqr(dic*dcp1) 
	       +  sigma[ic]*sigma[cp2] * sqr(dic*dcp2)
	       -  sigma[cp1]*sigma[cp2] * sqr(dcp1*dcp2)
	       );
      }
    else //imaginary part of mu[ijk]
      {
	muinvijk =(sigma[ic]*sqr(dic) - sigma[cp1]*sqr(dcp1) - sigma[cp2]*sqr(dcp2)
		    -sigma[ic]*sigma[cp1]*sigma[cp2]/sqr(omega) * sqr(dic*dcp1*dcp2))/omega;
      }
    /* taking care of the denominator from multipling complex conjugate */
    muinvijk /= (1 + sqr(sigma[cp1]/omega*sqr(dcp1))) * (1 + sqr(sigma[cp2]/omega*sqr(dcp2)));


    VecSetValue(muinv,i,muinvijk,INSERT_VALUES);


    }


       /*---------------------------*/
     
     ierr = VecAssemblyBegin(muinv); CHKERRQ(ierr);
     ierr = VecAssemblyEnd(muinv); CHKERRQ(ierr);
     ierr = PetscObjectSetName((PetscObject) muinv,"MuinvPMLFull"); CHKERRQ(ierr);

     *muinvout = muinv;
     PetscFunctionReturn(0);
}



