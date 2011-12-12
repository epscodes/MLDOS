#include <petsc.h>
#include <string.h>

#undef __FUNCT__ 
#define __FUNCT__ "OutputVec"
PetscErrorCode OutputVec(MPI_Comm comm, Vec x, const char *filenameComm, const char *filenameProperty)
{
  PetscErrorCode ierr;
 PetscViewer viewer;
 char filename[100]="";
 strcat(strcat(filename,filenameComm),filenameProperty);
 ierr = PetscViewerASCIIOpen(comm, filename, &viewer); CHKERRQ(ierr);
 ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
 ierr = VecView(x,viewer); CHKERRQ(ierr);
 ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
 PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "OutputMat"
PetscErrorCode OutputMat(MPI_Comm comm, Mat A, const char *filenameComm, const char *filenameProperty)
{
  PetscErrorCode ierr;
 PetscViewer viewer;
 char filename[100]="";
 strcat(strcat(filename,filenameComm),filenameProperty);
 ierr = PetscViewerASCIIOpen(comm, filename, &viewer); CHKERRQ(ierr);
 ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
 ierr = MatView(A,viewer); CHKERRQ(ierr);
 ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
 // When runing parallely, it may need -mat_ascii_output_large
 PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "RetrieveVecPointsValue"
PetscErrorCode RetrieveVecPoints(Vec x, int Npt, int *Pos, double *ptValues)
{
  PetscErrorCode ierr;
  Vec T;
  VecScatter scatter;
  IS from, to;
  ierr = VecCreateSeq(PETSC_COMM_SELF, Npt, &T);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,Npt, Pos, &from);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,Npt,0,1, &to);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,from,T,to,&scatter);CHKERRQ(ierr);
  ierr = VecScatterBegin(scatter,x,T,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,x,T,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
  int ix[Npt];
  int i;
  for(i=0; i<Npt; i++)
    ix[i]=i;
  
  ierr = VecGetValues(T,Npt,ix,ptValues);

  ierr = ISDestroy(from);CHKERRQ(ierr);
  ierr = ISDestroy(to);CHKERRQ(ierr);
 ierr =  VecScatterDestroy(scatter);CHKERRQ(ierr);
  ierr = VecDestroy(T);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "MyCheckAndOutputInt"
PetscErrorCode MyCheckAndOutputInt(PetscTruth flg, int CmdVar, const char *strCmdVar, const char *strCmdVarDetail)
{
  if (!flg) 
    { 
      char myerrmsg[100];
      sprintf(myerrmsg,"Please indicate %s with -%s option",strCmdVarDetail, strCmdVar);
      SETERRQ(PETSC_ERR_ARG_WRONG,myerrmsg);
    }
  else
    {PetscPrintf(PETSC_COMM_WORLD,"%s is %d \n",strCmdVarDetail,CmdVar);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "MyCheckAndOutputDouble"
PetscErrorCode MyCheckAndOutputDouble(PetscTruth flg, double CmdVar, const char *strCmdVar, const char *strCmdVarDetail)
{
  if (!flg) 
    { 
      char myerrmsg[100];
      sprintf(myerrmsg,"Please indicate %s with -%s option",strCmdVarDetail, strCmdVar);
      SETERRQ(PETSC_ERR_ARG_WRONG,myerrmsg);
    }
  else
    {PetscPrintf(PETSC_COMM_WORLD,"%s is %g \n",strCmdVarDetail,CmdVar);}
  PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "MyCheckAndOutputChar"
PetscErrorCode MyCheckAndOutputChar(PetscTruth flg, char *CmdVar, const char *strCmdVar, const char *strCmdVarDetail)
{
  if (!flg) 
    { 
      char myerrmsg[100];
      sprintf(myerrmsg,"Please indicate %s with -%s option",strCmdVarDetail, strCmdVar);
      SETERRQ(PETSC_ERR_ARG_WRONG,myerrmsg);
    }
  else
    {PetscPrintf(PETSC_COMM_WORLD,"%s is %s \n",strCmdVarDetail,CmdVar);}
  PetscFunctionReturn(0);
}





#undef __FUNCT__ 
#define __FUNCT__ "GetIntParaCmdLine"
PetscErrorCode GetIntParaCmdLine(int *ptCmdVar, const char *strCmdVar, const char *strCmdVarDetail)
{
  PetscTruth flg;
  PetscOptionsGetInt(PETSC_NULL,strCmdVar,ptCmdVar,&flg);
  if (!flg) 
    { 
      char myerrmsg[100];
      sprintf(myerrmsg,"Please indicate %s with -%s option",strCmdVarDetail, strCmdVar);
      SETERRQ(PETSC_ERR_ARG_WRONG,myerrmsg);
    }
  else
    {PetscPrintf(PETSC_COMM_WORLD,"%s is %d \n",strCmdVarDetail,*ptCmdVar);}
  PetscFunctionReturn(0);
}

