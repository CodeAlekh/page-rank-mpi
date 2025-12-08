#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <mpi.h>
#include "mat_op.cpp"
#include "my_time.cpp"

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        printf("ERROR: Pass in the .pm file name as argument\n");
        return 0;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    char *filename = argv[1];
    int iter = 100;
    double tolerance = 1e-4;

    if (argc > 2)
    {
        tolerance = atof(argv[2]);
        printf("Tolerance: %lf\n", tolerance);
    }

    if (argc > 3)
    {
        iter = atoi(argv[3]);
    }

    int rank, num_procs;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);
    std::string path = std::string(SRC_DIR) + "/data/" + filename;

    Time overall, file_read_time, page_rank_time;
    overall.start_time();

    file_read_time.start_time();
    LocalCSR A = readParallelPM(path.c_str(), comm);
    double f_read = file_read_time.get_time();

    std::vector<double> v(A.local_rows, 1.0 / A.global_rows);

    page_rank_time.start_time();
    page_rank(A, v, comm, iter, tolerance);
    double p_rank = page_rank_time.get_time();

    double overall_time = overall.get_time();

    MPI_Allreduce(MPI_IN_PLACE, &f_read, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &p_rank, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &overall_time, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0)
    {
        printf("File Name %s\n", filename);
        printf("File read time: %lf \n", f_read);
        printf("Page Rank convergence time: %lf \n", p_rank);
        printf("Total time: %lf \n", overall_time);
    }

    // print_vector(v, rank);

    MPI_Finalize();
}