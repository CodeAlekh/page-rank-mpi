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
    std::vector<double> x(A.local_cols, 1);
    std::vector<double> x_remote(A.global_cols - A.local_cols, 1);
    std::vector<double> y(A.local_cols, 0.0);

    vec_mul(A, x, y, MPI_COMM_WORLD);

    print_vector(y, rank);

    MPI_Finalize();
}