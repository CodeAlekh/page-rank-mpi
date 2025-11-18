#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <mpi.h>
#include "mat_op.cpp"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string path = std::string(SRC_DIR) + "/data/small.pm";
    LocalCSR A = readParallelPM(path.c_str(), MPI_COMM_WORLD);

    std::vector<double> v(A.local_rows, 1/A.global_rows);
    page_rank(A, v, MPI_COMM_WORLD);

    print_vector(v, rank);

    MPI_Finalize();
}