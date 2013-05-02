#include <petsc.h>

typedef struct{
  double Somega;
  Vec Sb, SweightedJ, Sepscoef;
} myfundatatypemaxminn;

/* in ldoskernel.c */
double ldosmaxminnkernel(int DegFree,double *epsopt, double *grad, void *data);

/* inshgobjandconstraint.c */
double maxminnobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);
double ldosmaxminnconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);
