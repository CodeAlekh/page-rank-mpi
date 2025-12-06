#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <mpi.h>
#include "mat_op.cpp"

int main(int argc, char** argv) {

     if (argc < 2) {
        printf("ERROR: Pass in the .pm file name as argument\n");
        return 0;
    }

    MPI_Init(&argc, &argv);

    char* filename = argv[1];

    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string path = std::string(SRC_DIR) + "/data/" + filename;
    LocalCSR A = readParallelPM(path.c_str(), MPI_COMM_WORLD);

    std::vector<double> x(A.local_rows, 1.0/A.global_rows);

    std::vector<double> y(A.local_rows, 0);
    vec_mul(A, x, y, MPI_COMM_WORLD);

    print_vector(y, rank);

    MPI_Finalize();
}