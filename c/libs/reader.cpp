#include <mpi.h>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <unordered_map>

// PETSc matrix binary header constant
#define PETSC_MAT_CODE 1211216

// Helper for endian swap
template <typename T>
void endian_swap(T* objp)
{
    unsigned char* memb = reinterpret_cast<unsigned char*>(objp);
    std::reverse(memb, memb + sizeof(T));
}

void print_arr_int(int* arr, int n, int rank) {
    printf("\n================ Rank %d =================\n", rank);
    for (int i = 0; i < n; i++) {
        std::cout << "[" << i << "] = " <<arr[i] << "\n";
    }
    fflush(stdout);
}

void print_arr(double* arr, int n, int rank) {
    printf("\n================ Rank %d =================\n", rank);
    for (int i = 0; i < n; i++) {
        std::cout << "[" << i << "] = " <<arr[i] << "\n";
    }
    fflush(stdout);
}

void print_vector_int(const std::vector<int> &v, int rank) {
    printf("\n================ Rank %d =================\n", rank);
    for (int i = 0; i < v.size(); i++) {
        std::cout << "[" << i << "] = " << v[i] << "\n";
    }
    fflush(stdout);
}

void printLocalMap(int rank, const std::unordered_map<int, double>& mp)
{
    std::cout << "==== Rank " << rank << " map ====\n";
    for (const auto& p : mp) {
        std::cout << "  index = " << p.first 
                  << "  value = " << p.second << "\n";
    }
    std::cout << std::endl;
}

void print_vector(const std::vector<double> &v, int rank) {
    printf("\n================ Rank %d =================\n", rank);
    for (int i = 0; i < v.size(); i++) {
        std::cout << "[" << i << "] = " << v[i] << "\n";
    }
    fflush(stdout);
}

struct CSR {
    std::vector<int> rowptr;
    std::vector<int> colind;
    std::vector<double> vals;
};

// Simple struct to hold local CSR data
struct LocalCSR {
    int global_rows, global_cols;
    int local_rows, local_cols;
    int first_row, first_col;
    int total_nnz, local_nnz;
    std::vector<int> rowptr;
    std::vector<int> colind;
    std::vector<double> vals;

    CSR on_proc;
    CSR off_proc;
};

void printLocalCSR(const LocalCSR &A, int rank) {
    printf("\n================ Rank %d =================\n", rank);
    printf("Global size: %d x %d\n", A.global_rows, A.global_cols);
    printf("Local rows: %d (first_row = %d)\n", A.local_rows, A.first_row);
    printf("Total nnz: %d (Local nnz = %d)\n", A.total_nnz, A.local_nnz);

    // Row pointer
    printf("\nRowPtr (%ld entries):\n", A.rowptr.size());
    for (int i = 0; i < A.rowptr.size(); i++) {
        printf("%d ", A.rowptr[i]);
    }
    printf("\n");

    // Full colind/vals
    printf("\nFull colind (%ld entries):\n", A.colind.size());
    for (int i = 0; i < A.colind.size(); i++) {
        printf("%d ", A.colind[i]);
    }
    printf("\n");

    printf("\nFull vals (%ld entries):\n", A.vals.size());
    for (int i = 0; i < A.vals.size(); i++) {
        printf("%.6lf ", A.vals[i]);
    }
    printf("\n");

    // On-proc part
    printf("\nOn-proc rowPtr (%ld entries):\n", A.on_proc.rowptr.size());
    for (int i = 0; i < A.on_proc.rowptr.size(); i++) {
        printf("%d ", A.on_proc.rowptr[i]);
    }
    printf("\n");

    printf("\nOn-proc colind (%ld entries):\n", A.on_proc.colind.size());
    for (int i = 0; i < A.on_proc.colind.size(); i++) {
        printf("%d ", A.on_proc.colind[i]);
    }
    printf("\n");

    printf("\nOn-proc vals (%ld entries):\n", A.on_proc.vals.size());
    for (int i = 0; i < A.on_proc.vals.size(); i++) {
        printf("%.6lf ", A.on_proc.vals[i]);
    }
    printf("\n");

    // Off-proc part
    printf("\nOff-proc rowPtr (%ld entries):\n", A.off_proc.rowptr.size());
    for (int i = 0; i < A.off_proc.rowptr.size(); i++) {
        printf("%d ", A.off_proc.rowptr[i]);
    }
    printf("\n");

    printf("\nOff-proc colind (%ld entries):\n", A.off_proc.colind.size());
    for (int i = 0; i < A.off_proc.colind.size(); i++) {
        printf("%d ", A.off_proc.colind[i]);
    }
    printf("\n");

    printf("\nOff-proc vals (%ld entries):\n", A.off_proc.vals.size());
    for (int i = 0; i < A.off_proc.vals.size(); i++) {
        printf("%.6lf ", A.off_proc.vals[i]);
    }
    printf("\n===========================================\n");

    fflush(stdout); // ensure clean ordering when printed from multiple ranks
}


LocalCSR readParallelPM(const char* filename, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    FILE* f = fopen(filename, "rb");
    if (!f) {
        std::cerr << "Rank " << rank << ": Cannot open file " << filename << "\n";
        MPI_Abort(comm, 1);
    }

    // Read header: code, nrows, ncols, nnz_total
    int32_t header[4];
    if (fread(header, sizeof(int32_t), 4, f) != 4) {
        std::cerr << "Rank " << rank << ": Error reading header\n";
        MPI_Abort(comm, 1);
    }

    int32_t code = header[0];
    int32_t nrows = header[1];
    int32_t ncols = header[2];
    int32_t nnz_total = header[3];

    bool need_endian_swap = false;
    if (code != PETSC_MAT_CODE) {
        // header likely big-endian relative to host; swap all header fields
        endian_swap(&code);
        endian_swap(&nrows);
        endian_swap(&ncols);
        endian_swap(&nnz_total);
        need_endian_swap = true;
    }

    // Row partition (1D row-wise)
    int rows_per_proc = nrows / num_procs;
    int row_remainder = nrows % num_procs;
    int local_rows = rows_per_proc + (rank < row_remainder ? 1 : 0);
    int first_row = rank * rows_per_proc + std::min(rank, row_remainder);
    int last_row = first_row + local_rows - 1;

    // For vector distribution align columns ownership to rows:
    int first_col = first_row;
    int local_cols = local_rows;

    // Read row sizes for this process (each row's nnz)
    std::vector<int32_t> row_sizes(local_rows);
    const size_t sizeof_i32 = sizeof(int32_t);
    const size_t sizeof_dbl = sizeof(double);

    // Row sizes are stored starting immediately after header (4 ints)
    // Seek to the first row size for this rank
    int64_t pos = (int64_t)(4 + first_row) * (int64_t)sizeof_i32;
    if (fseek(f, (long)pos, SEEK_SET) != 0) {
        std::cerr << "Rank " << rank << ": fseek row sizes failed\n";
        MPI_Abort(comm, 1);
    }
    if (fread(row_sizes.data(), sizeof_i32, local_rows, f) != (size_t)local_rows) {
        std::cerr << "Rank " << rank << ": fread row sizes failed\n";
        MPI_Abort(comm, 1);
    }

    if (need_endian_swap) {
        for (int i = 0; i < local_rows; ++i)
            endian_swap(&row_sizes[i]);
    }

    // Compute local_nnz
    int local_nnz = 0;
    for (int i = 0; i < local_rows; ++i) local_nnz += row_sizes[i];

    // Gather local nnz to compute offsets into global col index and value arrays
    std::vector<int> proc_nnz(num_procs);
    MPI_Allgather(&local_nnz, 1, MPI_INT, proc_nnz.data(), 1, MPI_INT, comm);

    long first_nnz = 0;
    for (int p = 0; p < rank; ++p) first_nnz += proc_nnz[p];

    long total_nnz = 0;
    for (int p = 0; p < num_procs; ++p) total_nnz += proc_nnz[p];

    // Read column indices for this process
    std::vector<int32_t> col_indices_i32(local_nnz);
    pos = (int64_t)(4 + nrows + first_nnz) * (int64_t)sizeof_i32;
    if (fseek(f, (long)pos, SEEK_SET) != 0) {
        std::cerr << "Rank " << rank << ": fseek col indices failed\n";
        MPI_Abort(comm, 1);
    }
    if (local_nnz > 0) {
        if (fread(col_indices_i32.data(), sizeof_i32, local_nnz, f) != (size_t)local_nnz) {
            std::cerr << "Rank " << rank << ": fread col indices failed\n";
            MPI_Abort(comm, 1);
        }
    }

    // Read values for this process
    std::vector<double> vals(local_nnz);
    // values begin after header + nrows row sizes + total_nnz col indices (all int32),
    // stored as doubles contiguous for all nnz.
    int64_t vals_start = (int64_t)(4 + nrows + total_nnz) * (int64_t)sizeof_i32;
    pos = vals_start + (int64_t)first_nnz * (int64_t)sizeof_dbl;
    if (fseek(f, (long)pos, SEEK_SET) != 0) {
        std::cerr << "Rank " << rank << ": fseek values failed\n";
        MPI_Abort(comm, 1);
    }
    if (local_nnz > 0) {
        if (fread(vals.data(), sizeof_dbl, local_nnz, f) != (size_t)local_nnz) {
            std::cerr << "Rank " << rank << ": fread values failed\n";
            MPI_Abort(comm, 1);
        }
    }

    if (need_endian_swap) {
        for (int i = 0; i < local_nnz; ++i) {
            endian_swap(&col_indices_i32[i]);
            endian_swap(&vals[i]);
        }
    }

    fclose(f);

    // Build full rowptr (counts per local row are in row_sizes)
    std::vector<int> rowptr(local_rows + 1, 0);
    for (int i = 0; i < local_rows; ++i)
        rowptr[i + 1] = rowptr[i] + row_sizes[i];

    // Convert col indices to int (global indices)
    std::vector<int> col_indices;
    col_indices.reserve(local_nnz);
    for (int i = 0; i < local_nnz; ++i)
        col_indices.push_back(static_cast<int>(col_indices_i32[i]));

    // Prepare on/off proc CSR pieces
    std::vector<int> on_proc_rowptr(local_rows + 1, 0);
    std::vector<int> off_proc_rowptr(local_rows + 1, 0);
    std::vector<int> on_proc_colind;   on_proc_colind.reserve(local_nnz);
    std::vector<double> on_proc_vals;  on_proc_vals.reserve(local_nnz);
    std::vector<int> off_proc_colind;  off_proc_colind.reserve(local_nnz);
    std::vector<double> off_proc_vals; off_proc_vals.reserve(local_nnz);

    int on_count = 0;
    int off_count = 0;
    for (int r = 0; r < local_rows; ++r) {
        int start = rowptr[r];
        int end = rowptr[r + 1];
        for (int j = start; j < end; ++j) {
            int col = col_indices[j];
            double v = vals[j];

            // on-proc if column index belongs to this rank's vector partition
            if (col >= first_col && col < first_col + local_cols) {
                on_proc_colind.push_back(col);
                on_proc_vals.push_back(v);
                ++on_count;
            } else {
                off_proc_colind.push_back(col);
                off_proc_vals.push_back(v);
                ++off_count;
            }
        }
        on_proc_rowptr[r + 1] = on_count;
        off_proc_rowptr[r + 1] = off_count;
    }

    // Build LocalCSR struct
    LocalCSR A;
    A.global_rows = nrows;
    A.global_cols = ncols;
    A.local_rows = local_rows;
    A.local_cols = local_cols;
    A.first_row = first_row;
    A.first_col = first_col;
    A.total_nnz = total_nnz;
    A.local_nnz = local_nnz;

    A.rowptr = std::move(rowptr);
    A.colind = std::move(col_indices);
    A.vals = std::move(vals);

    A.on_proc.rowptr = std::move(on_proc_rowptr);
    A.on_proc.colind = std::move(on_proc_colind);
    A.on_proc.vals = std::move(on_proc_vals);

    A.off_proc.rowptr = std::move(off_proc_rowptr);
    A.off_proc.colind = std::move(off_proc_colind);
    A.off_proc.vals = std::move(off_proc_vals);

    return A;
}
