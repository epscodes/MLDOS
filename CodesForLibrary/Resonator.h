//define Global variables;

//define macros;
#undef PI
#define PI 3.14159265358979e+00

// from MoperatorGeneral.c
PetscErrorCode MoperatorGeneral(MPI_Comm comm, Mat *Mout, int Nx, int Ny, int Nz, double hx, double hy, double hz, int bx[2], int by[2], int bz[2], double *muinv,int DimPeriod);

// from SourceGeneration.c
PetscErrorCode SourceSingleSetX(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);

PetscErrorCode SourceSingleSetY(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);

PetscErrorCode SourceSingleSetZ(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);

PetscErrorCode SourceSingleSetGlobal(MPI_Comm comm, Vec J, int globalpos, double amp);

PetscErrorCode SourceDuplicate(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);

PetscErrorCode SourceBlock(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, double hx, double hy, double hz, double lx, double ux, double ly, double uy, double lz, double uz, double amp);

// from PML.c
double pmlsigma(double RRT, double d);

PetscErrorCode EpsPMLFull(MPI_Comm comm, Vec epspml, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML);

PetscErrorCode MuinvPMLFull(MPI_Comm comm, Vec *muinvout, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML);

// from Eps.c 
PetscErrorCode EpsCombine(Mat D, Vec weight, Vec epspml, Vec epspmlQ, Vec epsgrad, double Qabs,double omega);

PetscErrorCode ModifyMatDiagonals(Mat M, Mat A, Mat D, Vec epsSReal, Vec epspmlQ, Vec epsmedium, Vec epsC, Vec epsCi, Vec epsP, int Nxyz, double omega);

PetscErrorCode ModifyMatDiagonalsForOmega( Mat M, Mat A, Mat D, Vec epsSReal, Vec epspmlQ, Vec epsC, Vec epsCi, Vec epsP, int Nxyz, double omegasqr);

//PetscErrorCode  yee_interp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double x0, double y0, double z0, double x1, double y1, double z1, int Mx, int My, int Mz);

PetscErrorCode  myinterp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int Nxo, int Nyo, int Nzo, int Mx, int My, int Mz, int Mzslab);

//PetscErrorCode myinterpSlab(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, int Nxo, int Nyo, int Nzo, int Mx, int My, int Mz, int Mzslab);

//PetscErrorCode General_interp(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double hx, double hy, double hz, double x0, double y0, double z0, double x1, double y1, double z1, int Mx, int My, int Mz);

PetscErrorCode myinterpTM2D(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nxo, int Nyo, int Mx, int My);

// from MathTools.c
PetscErrorCode  CmpVecProd(Vec va, Vec vb, Vec vout, Mat D, int aconj, Vec vai, Vec vbi);

PetscErrorCode  ImagIMat(MPI_Comm comm, Mat *Dout, int N);

PetscErrorCode   CongMat(MPI_Comm comm, Mat *Cout, int N);

PetscErrorCode   GetWeightVec(Vec weight,int Nx, int Ny, int Nz);

PetscErrorCode GetMediumVec(Vec epsmedium, int Nz, int Mz, double epsair, double epssub);

PetscErrorCode  GetRealPartVec(Vec vR, int N);

PetscErrorCode  ArrayToVec(double *pt, Vec V);

PetscErrorCode VecToArray(Vec V, double *pt, VecScatter scatter, IS from, IS to, Vec Vlocal, int Mxyz);

PetscErrorCode AddMuAbsorption(double *muinv, Vec muinvpml, double Qabs, int add);

PetscErrorCode TMprojmat(MPI_Comm comm, Mat *TMout, int Nxyz);

PetscErrorCode MatSetTwoDiagonals(Mat M, Vec epsC, Mat D, double sign);

// from Output.c
PetscErrorCode  OutputVec(MPI_Comm comm, Vec x, const char *filenameComm, const char *filenameProperty);

PetscErrorCode  OutputMat(MPI_Comm comm, Mat A, const char *filenameComm, const char *filenameProperty);

PetscErrorCode RetrieveVecPoints(Vec x, int Npt, int *Pos, double *ptValues);

PetscErrorCode MyCheckAndOutputInt(PetscTruth flg, int CmdVar, const char *strCmdVar, const char *strCmdVarDetail);

PetscErrorCode MyCheckAndOutputDouble(PetscTruth flg, double CmdVar, const char *strCmdVar, const char *strCmdVarDetail);

PetscErrorCode MyCheckAndOutputChar(PetscTruth flg, char *CmdVar, const char *strCmdVar, const char *strCmdVarDetail);



PetscErrorCode GetIntParaCmdLine(int *ptCmdVar, const char *strCmdVar, const char *strCmdVarDetail);


// from ResonatorSolver.c
double ResonatorSolver(int Mxyz, double *epsopt, double *grad, void *data);

// from ldos.c
double ldos(int numofvar, double *epsopt, double *grad, void *data);
double ldosdiff(int numofvar, double *epsopt, double *grad, void *data);

// from EigenSolver.c
PetscErrorCode EigenSolver(void *data, int Linear, int Eig, int maxeigit);
PetscErrorCode RayleighQuotient(Mat M, Vec diagB, Vec x, Vec b, Vec vR, Mat D, Vec tmpa, Vec tmpb, int N, int i);

// from ResonatorSolverRHS.c
double ResonatorSolverRHS(int Mxyz,double *epsopt, double *grad, void *data);

// from ResonatorSolverPOLXY.c
double ResonatorSolverPOLXY(int Mxyz,double *epsopt, double *grad, void *data);

// from MoperatorGeneralBloch.c
PetscErrorCode MoperatorGeneralBloch(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double hx, double hy, double hz, int bx[2], int by[2], int bz[2], double *muinv, int DimPeriod, double blochbc[3], Vec epsOmegasqr, Vec epsOmegasqri);

// from MoperatorGeneralBloch2D.c
int MoperatorGeneralBloch2D(MPI_Comm comm, Mat *Aout, int Nx, int Ny, int Nz, double hx, double hy, double hz, int bx[2], int by[2], int bz[2], double *muinv, int DimPeriod, double blochbc[3], Vec epsOmegasqr, Vec epsOmegasqri);

// from ResonatorSolverSolar.c
double ResonatorSolverSolar(int Mxyz,double *epsopt, double *grad, void *data);
PetscErrorCode SolarComputeKernel(Vec epsCurrent, Vec epsOmegasqr, Vec epsOmegasqri, double blochbc[3], double *kdlos, Vec kepsgrad);
double ldossolar(int numofvar, double *varopt, double *grad, void *data);

// from ResonatorSolverSolar2D.c
double ResonatorSolverSolar2D(int Mxyz,double *epsopt, double *grad, void *data);
PetscErrorCode SolarComputeKernel2D(Vec epsCurrent, Vec epsOmegasqr, Vec epsOmegasqri, double blochbc[3], double *kdlos, Vec kepsgrad);
double ldossolar2D(int numofvar, double *varopt, double *grad, void *data);

// from SolarEgienvaluesSolver.c
int SolarEigenvaluesSolver(Mat M, Vec epsCurrent, Vec epspmlQ, Mat D);


// from ResonatorSolverSolar2D.c
double ResonatorSolverSolar2D(int Mxyz,double *epsopt, double *grad, void *data);

int SolarComputeKernel2D(Vec epsCurrent, Vec epsOmegasqr, Vec epsOmegasqri, double blochbc[3], double *ptkldos, Vec kepsgrad);

// datatype used for optimization of dielectric structure;
typedef struct { 
  int SNx, SNy, SNz; 
  double Shx, Shy, Shz;
  double Somega;
  KSP Sksp;
  Vec SepspmlQ, Sepscoef, Sepsmedium, SepsC, SepsCi, SepsP, Sx, Scglambda, Sb, SweightedJ, SvR, SepsSReal, Sepsgrad, Svgrad, Svgradlocal, Stmp, Stmpa, Stmpb;
  Mat SA, SD, SM;
  IS Sfrom, Sto;
  VecScatter Sscatter;
  char *SfilenameComm;
} myfundatatype;

// dataype used for optimization of frequency;
typedef struct { 
  int SNx, SNy, SNz; 
  double Shx, Shy, Shz;
  double Somega, Ssparedouble;
  double *Sspareptdouble;
  KSP Sksp;
  Vec SepspmlQ, Sepscoef, SepsC, SepsCi, SepsP, Sx, Scglambda, Sb, SweightedJ, SvR, SepsSReal, Sepsgrad, Svgrad, Svgradlocal, Stmp, Stmpa, Stmpb;
  Mat SA, SD, SM;
  IS Sfrom, Sto;
  VecScatter Sscatter;
  char *SfilenameComm;
} myfundatatypeq;

// datatype used for optimization of dielectric structure with mullitple RHS;
typedef struct { 
  int SNx, SNy, SNz, Sclx,Scly,Sclz, Sncx, Sncy, Sncz, SJdirection; 
  double Shx, Shy, Shz;
  double Somega;
  KSP Sksp;
  Vec SepspmlQ, Sepscoef, Sepsmedium, SepsC, SepsCi, SepsP, Sx, Scglambda, SJ, Sb, SweightedJ, SvR, SepsSReal, Sepsgrad, Svgrad, Svgradlocal, Stmp, Stmpa, Stmpb;
  Mat SA, SD, SM;
  IS Sfrom, Sto;
  VecScatter Sscatter;
  char *SfilenameComm;
} myfundataRHStype;

// dataype used for optimization of frequency with multiple RHS;
typedef struct { 
  int SNx, SNy, SNz,Sclx, Scly, Sclz, Sncx,Sncy,Sncz, SJdirection; 
  double Shx, Shy, Shz;
  double Somega, Ssparedouble;
  double *Sspareptdouble;
  KSP Sksp;
  Vec SepspmlQ, Sepscoef, SepsC, SepsCi, SepsP, Sx, Scglambda, SJ, Sb, SweightedJ, SvR, SepsSReal, Sepsgrad, Svgrad, Svgradlocal, Stmp, Stmpa, Stmpb;
  Mat SA, SD, SM;
  IS Sfrom, Sto;
  VecScatter Sscatter;
  char *SfilenameComm;
} myfundataRHStypeq;

// datatype used for optimization of dielectric structure with Polarization in both X and Y;
typedef struct { 
  int SNx, SNy, SNz; 
  double Shx, Shy, Shz;
  double Somega;
  KSP Skspx, Skspy;
  Vec SepspmlQ, Sepscoef, Sepsmedium, SepsC, SepsCi, SepsPx, SepsPy, Ssolx, Ssoly, Scglambda, Sbx, Sby, SweightedJx, SweightedJy, SvR, SepsSReal, Sepsgrad, Svgrad, Svgradlocal, Stmp, Stmpa, Stmpb;
  Mat SA, SD, SMatX, SMatY;
  IS Sfrom, Sto;
  VecScatter Sscatter;
  char *SfilenameComm;
} myfundataPOLXYtype;

// dataype used for optimization of frequency with Polarization in both X and Y;
typedef struct { 
  int SNx, SNy, SNz; 
  double Shx, Shy, Shz;
  double Somega, Ssparedouble;
  double *Sspareptdouble;
  KSP Skspx, Skspy;
  Vec SepspmlQ, Sepscoef, SepsC, SepsCi, SepsPx, SepsPy, Ssolx, Ssoly, Scglambda, Sbx, Sby, SweightedJx, SweightedJy, SvR, SepsSReal, Sepsgrad, Svgrad, Svgradlocal, Stmp, Stmpa, Stmpb;
  Mat SA, SD, SMatX, SMatY;
  IS Sfrom, Sto;
  VecScatter Sscatter;
  char *SfilenameComm;
} myfundataPOLXYtypeq;


// datatype used for optimization of dielectric structure with Solar Problem;
typedef struct { double Somega; double *Sptepsinput;
} myfundataSolartype;
