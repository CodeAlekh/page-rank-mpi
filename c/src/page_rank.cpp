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
    int iter = 100;
    double tolerance = 1e-4;

    if (argc > 2) {
        tolerance = atof(argv[2]);
        printf("Tolerance: %lf\n", tolerance);
    }

    if (argc > 3) {
        iter = atoi(argv[3]);
    }

    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string path = std::string(SRC_DIR) + "/data/" + filename;
    LocalCSR A = readParallelPM(path.c_str(), MPI_COMM_WORLD);

    std::vector<double> v(A.local_rows, 1/A.global_rows);
    page_rank(A, v, MPI_COMM_WORLD, iter, tolerance);

    MPI_Finalize();
}