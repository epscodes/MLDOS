#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <petsc.h>


#undef __FUNCT__ 
#define __FUNCT__ "SourceSingleSetX"
PetscErrorCode SourceSingleSetX(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp)
{
  PetscErrorCode ierr;

  int pos, Nxyz; 
  Nxyz = Nx*Ny*Nz;
  pos = scx*Ny*Nz + scy*Nz + scz;

  //VecSet(J,0.0);

  int ns, ne;
  ierr = VecGetOwnershipRange(J, &ns, &ne); CHKERRQ(ierr);
  
  if ( ns < pos+1 && ne > pos)
    {
      ierr=VecSetValue(J,pos, amp, ADD_VALUES); CHKERRQ(ierr); //Source in the x-direction
    }
  ierr = VecAssemblyBegin(J);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(J); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "SourceSingleSetY"
PetscErrorCode SourceSingleSetY(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp)
{
  PetscErrorCode ierr;

  int pos, Nxyz; 
  Nxyz = Nx*Ny*Nz;
  pos = scx*Ny*Nz + scy*Nz + scz;

  //VecSet(J,0.0);

  int ns, ne;
  ierr = VecGetOwnershipRange(J, &ns, &ne); CHKERRQ(ierr);
  
  if ( ns < pos+Nxyz+1 && ne > pos + Nxyz)
    {
      ierr=VecSetValue(J,pos+Nxyz, amp, ADD_VALUES); CHKERRQ(ierr); //Source in the y-direction
    }
  ierr = VecAssemblyBegin(J);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(J); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "SourceSingleSetZ"
PetscErrorCode SourceSingleSetZ(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp)
{
  PetscErrorCode ierr;

  int pos, Nxyz; 
  Nxyz = Nx*Ny*Nz;
  pos = scx*Ny*Nz + scy*Nz + scz;

  //VecSet(J,0.0);

  int ns, ne;
  ierr = VecGetOwnershipRange(J, &ns, &ne); CHKERRQ(ierr);
  
  if ( ns < pos+2*Nxyz+1 && ne > pos + 2*Nxyz)
      ierr = VecSetValue(J,pos+2*Nxyz, amp, ADD_VALUES); CHKERRQ(ierr);
  
  ierr = VecAssemblyBegin(J);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(J); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}




#undef __FUNCT__ 
#define __FUNCT__ "SourceSingleSetGlobal"
PetscErrorCode SourceSingleSetGlobal(MPI_Comm comm, Vec J, int globalpos, double amp)
{
  PetscErrorCode ierr;

  //VecSet(J,0.0);

  int ns, ne;
  ierr = VecGetOwnershipRange(J, &ns, &ne); CHKERRQ(ierr);
  
  if ( ns < globalpos+1 && ne > globalpos)
    {
      ierr=VecSetValue(J,globalpos, amp, ADD_VALUES); CHKERRQ(ierr);
    }
  ierr = VecAssemblyBegin(J);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(J); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}




#undef __FUNCT__ 
#define __FUNCT__ "SourceDuplicate"
PetscErrorCode SourceDuplicate(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp)
{
  int i, j, k;
  int halfNx, halfNy, halfNz, pos, N; 
  N = Nx*Ny*Nz;  
  halfNx = Nx/2;
  halfNy = Ny/2;
  halfNz = Nz/2;

  Vec b;
  PetscErrorCode ierr;
  ierr = VecCreate(comm,&b);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b, "Source");CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,6*N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b); CHKERRQ(ierr);
  VecSet(b,0.0);

  int ns, ne;
  ierr = VecGetOwnershipRange(b,&ns,&ne); CHKERRQ(ierr);


  for(i=0;i<2;i++)
    for(j=0;j<2;j++)
      for(k=0;k<2;k++)
	{
	  pos = (halfNx + pow(-1,i)*scx)*Ny*Nz + (halfNy + pow(-1,j)*scy)*Nz + (halfNz + pow(-1,k)*scz);
	  //VecSetValue(b,pos+0*N, pow(-1,i)*amp, INSERT_VALUES);
	  //VecSetValue(b,pos+1*N, pow(-1,j)*amp, INSERT_VALUES);

	  if ( ns < pos+2*N+1 && ne > pos + 2*N)
	  VecSetValue(b,pos+2*N, pow(-1,k)*amp, INSERT_VALUES);
	  // I use insert_values, instead of Add_values (fail when run in parallel); //May rewrite this program so that it only run once;
	}

  VecAssemblyBegin(b);
  VecAssemblyEnd(b); 

  *bout = b;
  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "SourceBlock"
PetscErrorCode SourceBlock(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, double hx, double hy, double hz, double lx, double ux, double ly, double uy, double lz, double uz, double amp)
{
  int i, j, k, pos, N;
  N = Nx*Ny*Nz;

  Vec b;
  PetscErrorCode ierr;
  ierr = VecCreate(comm,&b);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) b, "Source");CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,6*N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b); CHKERRQ(ierr);
  VecSet(b,0.0);

  int ns, ne;
  ierr = VecGetOwnershipRange(b, &ns, &ne); CHKERRQ(ierr);

  
  for (i=0;i<Nx;i++)
    if ((i*hx>lx) && (i*hx<ux))
      {for (j=0; j<Ny;j++)
	  if ((j*hy>ly) && (j*hy<uy))
	    { 
	      for (k=0; k<Nz; k++)
		//if ((k*hz>lz) && (k*hz<uz)) // uncomment this if z direction is not trivial;
		{ pos = i*Ny*Nz + j*Nz + k;
		  //VecSetValue(b,pos,amp,INSERT_VALUES);
		  //VecSetValue(b,pos+N,amp,INSERT_VALUES);
		   if ( ns < pos+2*N+1 && ne > pos + 2*N)
		  VecSetValue(b,pos+2*N,amp,INSERT_VALUES);
		}
	    }
	    
      }
  VecAssemblyBegin(b);
  VecAssemblyEnd(b); 
  
  *bout = b;
  PetscFunctionReturn(0);
  
}

