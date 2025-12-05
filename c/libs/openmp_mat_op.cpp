#include "mat_op.cpp"

void enhanced_spmv_serial_threaded(const CSR &A, const std::vector<double> &x, std::vector<double> &y, int rank, int num_procs, int total_cols) {
    int nrows = A.rowptr.size() - 1;
    // y.assign(nrows, 0.0);

    #pragma omp for
    for (int i = 0; i < nrows; i++) {
        double sum = 0.0;

        int row_start = A.rowptr[i];
        int row_end   = A.rowptr[i + 1];

        for (int k = row_start; k < row_end; k++) {
            int col = A.colind[k];
            int local_col = local_index_from_global(
                                col, 
                                rank,
                                num_procs,
                                total_cols
                            );
            sum += A.vals[k] * x[local_col];
        }
        y[i] = sum;
    }
}

void enhanced_spmv_serial_threaded(const CSR &A, double* ordered_remote_vals, std::vector<double> &y) {
    int nrows = A.rowptr.size() - 1;
    // y.assign(nrows, 0.0);

    #pragma omp for
    for (int i = 0; i < nrows; i++) {
        double sum = 0.0;

        int row_start = A.rowptr[i];
        int row_end   = A.rowptr[i + 1];

        for (int k = row_start; k < row_end; k++) {
            int col = A.colind[k];
            sum += A.vals[k] * ordered_remote_vals[col];
        }

        y[i] = sum;
    }
}

// void enhanced_vec_mul_threaded(LocalCSR& A, vector<double>& x_local, std::vector<double>& y, CommunicationDetails& c, MPI_Comm comm) {
//     int rank, num_procs;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &num_procs);

//     // 1. Gather remote vector

//     double* remote_values_ordered  = nullptr;
//     std::vector<double> y_local;
//     std::vector<double> y_remote;

//     #pragma omp single
//     {
//         y_local.assign(A.on_proc.rowptr.size() - 1, 0.0);
//     }
//     #pragma omp barrier

//     // 2. Compute on-proc part
//     enhanced_spmv_serial_threaded(A.on_proc, x_local, y_local, rank, num_procs, A.global_cols);

//     #pragma omp single
//     {
//         remote_values_ordered = gather_value_from_communication_details(A, x_local, c, rank, num_procs, comm);
//         y_remote.assign(A.off_proc.rowptr.size() - 1, 0.0);
//     }
//     #pragma omp barrier

//     // 3. Compute off-proc part
//     enhanced_spmv_serial_threaded(A.off_proc, remote_values_ordered, y_remote);

//     // 4. Add them
//     int n = A.local_rows;
//     #pragma omp single
//     {
//         y.resize(n);
//     }
//     #pragma omp barrier

//     #pragma omp for
//     for (int i = 0; i < n; i++)
//         y[i] = y_local[i] + y_remote[i];
// }

void enhanced_vec_mul_threaded(
    LocalCSR& A, 
    vector<double>& x_local, 
    std::vector<double>& y, 
    CommunicationDetails& c, 
    MPI_Comm comm,
    std::vector<double>& y_local_buf,  // SHARED: Passed from caller
    std::vector<double>& y_remote_buf, // SHARED: Passed from caller
    double*& remote_values_ordered     // SHARED: Reference to a shared pointer
) {
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    #pragma omp single
    {
        y_local_buf.assign(A.on_proc.rowptr.size() - 1, 0.0);
    }
    enhanced_spmv_serial_threaded(A.on_proc, x_local, y_local_buf, rank, num_procs, A.global_cols);

    #pragma omp single
    {
        remote_values_ordered = gather_value_from_communication_details(A, x_local, c, rank, num_procs, comm);
        
        y_remote_buf.assign(A.off_proc.rowptr.size() - 1, 0.0);
    }
    #pragma omp barrier 

    enhanced_spmv_serial_threaded(A.off_proc, remote_values_ordered, y_remote_buf);

    int n = A.local_rows;
    #pragma omp single
    {
        y.resize(n);
    }
    #pragma omp barrier

    #pragma omp for
    for (int i = 0; i < n; i++)
        y[i] = y_local_buf[i] + y_remote_buf[i];
}

void enhanced_scalar_multiply_add_threaded(vector<double>& vec, double mul, double add) {
    #pragma omp for
    for (int i = 0; i < vec.size(); i++) {
        vec[i] *= mul;
        vec[i] += add;
    }
}

double enhanced_dangling_sum_local_threaded(LocalCSR& A, vector<double>& page_rank_local, vector<int>& outdegree) {
    double sum = 0.0;
    for (int i = 0; i < A.local_rows; i++) {
        int global_index = A.first_row + i;
        if (outdegree[global_index] == 0) {
            sum += page_rank_local[i];
        }
    }
    return sum;
}

// void enhanced_page_rank_step_threaded(LocalCSR& A, vector<double>& rank_vec, vector<int>& outdegree, double teleportation_factor, CommunicationDetails& c, int rank, int num_procs, MPI_Comm comm) {
//     vector<double> next(A.local_rows, 0.0);
//     double local_sum = 0.0;
//     double total_dangling_sum = 0.0;
//     double dangling_correction = 0.0;
    
//     #pragma omp single 
//     {
//         local_sum = enhanced_dangling_sum_local_threaded(A, rank_vec, outdegree);
//         MPI_Allreduce(&local_sum, &total_dangling_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
//         dangling_correction =  DAMPING * (total_dangling_sum / A.global_rows);
//     }

//     #pragma omp barrier

//     enhanced_vec_mul_threaded(A, rank_vec, next, c, comm);

//     //Dont need arrays for dangling and teleportation since they're same for every elements
//     enhanced_scalar_multiply_add_threaded(next, DAMPING, (dangling_correction + teleportation_factor));

//     #pragma omp single
//     {
//         rank_vec = next;
//     }
//     #pragma omp barrier
// }

void enhanced_page_rank_step_threaded(
    LocalCSR& A, 
    vector<double>& rank_vec, 
    vector<int>& outdegree, 
    double teleportation_factor, 
    CommunicationDetails& c, 
    int rank, 
    int num_procs, 
    MPI_Comm comm,
    vector<double>& next,              // SHARED: Passed in
    vector<double>& y_local_buf,       // SHARED: Passed in
    vector<double>& y_remote_buf,      // SHARED: Passed in
    double*& remote_vals_ptr           // SHARED: Passed in
) {
    double local_sum = 0.0;
    
    double dangling_correction = 0.0;
    
    #pragma omp single copyprivate(dangling_correction)
    {
        double total_dangling_sum = 0.0;
        local_sum = enhanced_dangling_sum_local_threaded(A, rank_vec, outdegree);
        MPI_Allreduce(&local_sum, &total_dangling_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        dangling_correction =  DAMPING * (total_dangling_sum / A.global_rows);
    }

    // Call the modified vec_mul with SHARED buffers
    enhanced_vec_mul_threaded(A, rank_vec, next, c, comm, y_local_buf, y_remote_buf, remote_vals_ptr);

    enhanced_scalar_multiply_add_threaded(next, DAMPING, (dangling_correction + teleportation_factor));

    #pragma omp single
    {
        rank_vec = next;
    }
    #pragma omp barrier
}

// void enhanced_page_rank_threaded(LocalCSR& A, vector<double>& rank_vec, MPI_Comm comm, int max_iters = 100, double tol = 1e-4) {

//     int rank, num_procs;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &num_procs);
//     vector<double> old(A.local_rows);

//     vector<int> outdegree(A.global_cols, 0);
//     normalize_outdegrees_allreduce(A, rank, num_procs, outdegree, comm);

//     CommunicationDetails c = build_comm_details(A, rank_vec, comm);

//     //Modify to local index so that we wont need map every iteration
//     renumber_off_proc_columns(A,c);

//     double teleportation_factor = (1-DAMPING)/A.global_rows;
//     bool converged = false;

//     #pragma omp parallel shared(rank_vec, old, converged)
//     {
//         for (int iter = 0; iter < max_iters; iter++) {

//             // Savinf previous value for tolerance
//             #pragma omp single 
//             {
//                 old = rank_vec;
//             }
//             #pragma omp barrier

//             // One PageRank iteration
//             enhanced_page_rank_step_threaded(A, rank_vec, outdegree, teleportation_factor, c, rank, num_procs, comm);

//             #pragma omp single 
//             {
//                 // Compute local L1 diff
//                 double local_diff = 0.0;
//                 for (int i = 0; i < A.local_rows; i++) {
//                     local_diff += std::fabs(rank_vec[i] - old[i]);
//                 }

//                 // All Reduce to global differece to check tolerance
//                 double global_diff = 0.0;
//                 MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, comm);
//                 if (global_diff < tol) converged = true;
//             }
//             #pragma omp barrier

//             if (converged) {
//                 break;
//             }
//         }
//     }
// }

void enhanced_page_rank_threaded(LocalCSR& A, vector<double>& rank_vec, MPI_Comm comm, int max_iters = 100, double tol = 1e-4) {

    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);
    
    vector<double> old(A.local_rows);
    vector<int> outdegree(A.global_cols, 0);
    normalize_outdegrees_allreduce(A, rank, num_procs, outdegree, comm);

    CommunicationDetails c = build_comm_details(A, rank_vec, comm);
    renumber_off_proc_columns(A,c);

    double teleportation_factor = (1-DAMPING)/A.global_rows;
    bool converged = false;

    // --- HOISTING VARIABLES ---
    // We create these variables HERE so they are shared by default in the parallel region.
    vector<double> next(A.local_rows, 0.0);
    vector<double> y_local_buf;   // Scratchpad for vec_mul
    vector<double> y_remote_buf;  // Scratchpad for vec_mul
    double* remote_vals_ptr = nullptr; // Shared pointer for MPI buffer

    // Pass them into the parallel region as SHARED
    #pragma omp parallel shared(rank_vec, old, converged, outdegree, c, next, y_local_buf, y_remote_buf, remote_vals_ptr)
    {
        for (int iter = 0; iter < max_iters; iter++) {

            #pragma omp single 
            {
                old = rank_vec;
            }
            #pragma omp barrier

            // Pass the shared buffers down
            enhanced_page_rank_step_threaded(A, rank_vec, outdegree, teleportation_factor, c, rank, num_procs, comm, next, y_local_buf, y_remote_buf, remote_vals_ptr);

            #pragma omp single 
            {
                double local_diff = 0.0;
                for (int i = 0; i < A.local_rows; i++) {
                    local_diff += std::fabs(rank_vec[i] - old[i]);
                }

                double global_diff = 0.0;
                MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, comm);
                if (global_diff < tol) converged = true;
            }
            #pragma omp barrier

            if (converged) {
                break;
            }
        }
    }
}