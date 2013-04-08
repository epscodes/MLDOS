#include <petsc.h>

typedef struct{
  double Somega;
  Vec Sb, SweightedJ, Sepscoef;
  KSP Sksp;
} myfundatatypeshg;

/* in ldoskernel.c */
double ldoskernel(int DegFree,double *epsopt, double *grad, void *data);

/* inshgobjandconstraint.c */
double maxminobjfun(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);
double ldosconstraint(int DegFreeAll,double *epsoptAll, double *gradAll, void *data);
