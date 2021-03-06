#include <petsc.h>

typedef struct{
  int Sidj;
  double Somega;
  Vec Sb, SweightedJ, Sepscoef;
} myfundatatypemaxminn;

/* in ldoskernel.c */
double ldosmaxminnkernel(int DegFree,double *epsopt, double *grad, void *data);

/* in ldosmaxminneignesolver.c */
PetscErrorCode MaxMinNEigenSolver(int Linear, int Eig, int maxeigit, void *data);

/* inshgobjandconstraint.c */
double maxminnobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);
double ldosmaxminnconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);
