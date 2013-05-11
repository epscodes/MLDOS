#include "maxminnheader.h"

#undef __FUNCT__ 
#define __FUNCT__ "maxminnobjfun"
double maxminnobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  if(gradAll)
    {
      int i;
      for (i=0;i<DegFreeAll-1;i++)
	{
	  gradAll[i]=0;
	}
      gradAll[DegFreeAll-1]=1;
    }
  
  PetscPrintf(PETSC_COMM_WORLD,"**the current objective value is %.8e**\n",epsoptAll[DegFreeAll-1]);

  return epsoptAll[DegFreeAll-1];
}



#undef __FUNCT__ 
#define __FUNCT__ "ldosmaxminnconstraint"
double ldosmaxminnconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data)
{
  /* altoght epsopt now has one more element, ldoskernel (a variant of ResonantSolver) should be able to put correct results in the first DegFree-1 poistions */
  double tmpldos;
  tmpldos=ldosmaxminnkernel(DegFreeAll-1,epsoptAll,gradAll,data);
  
  if(gradAll)
    {
      int i;
      for(i=0;i<DegFreeAll-1;i++)
	{
	  gradAll[i]=-gradAll[i];
	}
      gradAll[DegFreeAll-1]=1;
    }

  PetscPrintf(PETSC_COMM_WORLD,"**the current constraint value is %.8e which is from t value %.8e and ldos value %.8e **\n",epsoptAll[DegFreeAll-1]-tmpldos,epsoptAll[DegFreeAll-1],tmpldos);

  return epsoptAll[DegFreeAll-1]-tmpldos;
}
