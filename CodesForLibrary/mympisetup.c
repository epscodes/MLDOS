#include <mpi.h>

extern int commsz, myrank, mygroup, myid, numgroups; 
extern MPI_Comm comm_group, comm_sum;

int mympisetup()
{
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &commsz);
  
  if (commsz%numgroups!=0 && myrank==0)
    printf("!!!=== not dividing workers evenly === !!! \n");
      
  mygroup = (myrank*numgroups)/commsz;
  myid = myrank%(commsz/numgroups);

  MPI_Comm_split(MPI_COMM_WORLD, mygroup, myrank, &comm_group);
  MPI_Comm_split(MPI_COMM_WORLD, myid, myrank, &comm_sum);
  
}
