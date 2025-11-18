#include "reader.cpp"
#include <set>
#include <cmath>
#define DAMPING 0.85

using namespace std;

int calculate_process_rank(int num_procs, int total_cols, int col_index) {
    int remainder = total_cols % num_procs;
    int per_proc_cols = total_cols / num_procs;
    if (col_index < remainder * (per_proc_cols + 1)) {
        return col_index / (per_proc_cols + 1);
    } else {
        return remainder + (col_index - remainder * (per_proc_cols + 1)) / per_proc_cols;
    }
}

int local_index_from_global(int global_idx, int rank, int num_procs, int total_cols) {
    int per_proc = total_cols / num_procs;
    int rem = total_cols % num_procs;

    int start;
    if (rank < rem) {
        start = rank * (per_proc + 1);
    } else {
        start = rem * (per_proc + 1) + (rank - rem) * per_proc;
    }
    return global_idx - start;
}

std::unordered_map<int, double> index_to_value_map(
        const int* indices, 
        const double* values, 
        int n
    )
{
    std::unordered_map<int, double> mp;
    mp.reserve(n);

    for (int i = 0; i < n; i++) {
        mp[indices[i]] = values[i];
    }

    return mp;
}

unordered_map<int,double> gather_off_proc_vectors(LocalCSR &A, std::vector<double> &x_local, MPI_Comm comm) {
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int* my_required_count = (int*) calloc(num_procs, sizeof(int));
    int* other_required_count = (int*) calloc(num_procs, sizeof(int));
    map<int, std::vector<int>> owner_to_columns;

    set<int> unique_col_index_set(A.colind.begin(), A.colind.end());
    vector<int> unique_cols(unique_col_index_set.begin(), unique_col_index_set.end());

    for (int i = 0; i < unique_cols.size(); i++) {
        int col_idx = unique_cols[i];
        int owner = calculate_process_rank(num_procs, A.global_cols,col_idx );
        owner_to_columns[owner].push_back(col_idx);
        my_required_count[owner]++;
    }
    MPI_Alltoall(my_required_count, 1, MPI_INT, other_required_count, 1, MPI_INT, comm);

    //Now we have how many element each process needs to send to the others. We start by sending indices first
    vector<int> s_index_buffer;
    s_index_buffer.reserve(A.global_cols); //Just to prevent reallocation
    int* s_index_count = (int*) calloc(num_procs, sizeof(int));
    int* s_index_disp = (int*) calloc(num_procs, sizeof(int));
    int s_offset = 0;
    int r_offset = 0;
    int* r_index_count = (int*) calloc(num_procs, sizeof(int));
    int* r_index_disp = (int*) calloc(num_procs, sizeof(int));
    vector<int> r_index_buffer;
    r_index_buffer.reserve(A.global_cols); //Just to prevent reallocation
    for (int proc = 0; proc < num_procs; proc++) {
        if (rank == proc) {
            s_index_disp[proc] = s_offset;
            s_index_count[proc] = 0;

            r_index_disp[proc] = r_offset;
            r_index_count[proc] = 0;

        } else {
            vector<int> col_idx_list = owner_to_columns[proc];
            int size = col_idx_list.size();
            s_index_disp[proc] = s_offset;
            s_index_count[proc] = size;
            s_offset += size;
            for (int col_idx: col_idx_list) {
                s_index_buffer.push_back(col_idx);
            }
            // Receiving count and displacements
            r_index_disp[proc] = r_offset;
            r_index_count[proc] = other_required_count[proc];
            r_offset += other_required_count[proc];
        }
    }
    //Size of the receiving buffer is final. So we do this.
    r_index_buffer.resize(r_offset);

    MPI_Alltoallv(
        s_index_buffer.data(), s_index_count, s_index_disp, MPI_INT,
        r_index_buffer.data(), r_index_count, r_index_disp, MPI_INT, comm
    );


    //Now we have all required indices which we must send to each procses. Send it using alltoallv
    double* s_val_buffer = (double*) calloc(r_offset, sizeof(double));
    int* s_val_count = (int*) calloc(num_procs, sizeof(int));
    int* s_val_disp = (int*) calloc(num_procs, sizeof(int));
    int s_val_offset = 0;

    double* r_val_buffer = (double*) calloc(s_offset, sizeof(double));
    int* r_val_count = (int*) calloc(num_procs, sizeof(int));
    int* r_val_disp = (int*) calloc(num_procs, sizeof(int));
    int r_val_offset = 0;

    for (int proc = 0; proc < num_procs; proc++) {
        if (rank == proc) {
        } else {
            for (int i = 0; i < r_index_count[proc]; i++) {
                int global_index = r_index_buffer[r_index_disp[proc] + i];
                int local_index = local_index_from_global(global_index,rank,num_procs, A.global_cols);
                s_val_buffer[r_index_disp[proc] + i] = x_local[local_index];
            }
        }
    }

    MPI_Alltoallv(
        s_val_buffer, r_index_count, r_index_disp, MPI_DOUBLE,
        r_val_buffer, s_index_count, s_index_disp, MPI_DOUBLE, comm
    );

    return  index_to_value_map(s_index_buffer.data(), r_val_buffer, s_index_buffer.size());
    
}

void spmv_serial(const CSR &A, const std::vector<double> &x, std::vector<double> &y, int rank, int num_procs, int total_cols) {
    int nrows = A.rowptr.size() - 1;
    y.assign(nrows, 0.0);

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
            // if(rank == 1) {
            //     printf("k: %d, Col: %d -> local_col: %d, A.vals[k] = %lf, x[local_col] = %lf \n", k, col, local_col, A.vals[k], x[local_col]);
            //     fflush(stdout);
            // }
            sum += A.vals[k] * x[local_col];
        }

        y[i] = sum;
    }
}

void spmv_serial(const CSR &A, unordered_map<int,double> index_to_value, std::vector<double> &y) {
    int nrows = A.rowptr.size() - 1;
    y.assign(nrows, 0.0);

    for (int i = 0; i < nrows; i++) {
        double sum = 0.0;

        int row_start = A.rowptr[i];
        int row_end   = A.rowptr[i + 1];

        for (int k = row_start; k < row_end; k++) {
            int col = A.colind[k];
            sum += A.vals[k] * index_to_value[col];
        }

        y[i] = sum;
    }
}


void vec_mul(LocalCSR& A, vector<double>& x_local, std::vector<double>& y, MPI_Comm comm) {
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // 1. Gather remote vector
    std::vector<double> x_remote(A.global_cols - A.local_cols);
    unordered_map<int, double> index_to_value = gather_off_proc_vectors(A, x_local, comm);

    // 2. Compute on-proc part
    std::vector<double> y_local(A.local_rows, 0.0);
    spmv_serial(A.on_proc, x_local, y_local, rank, num_procs, A.global_cols);


    // 3. Compute off-proc part
    std::vector<double> y_remote;
    spmv_serial(A.off_proc, index_to_value, y_remote);

    // 4. Add them
    int n = A.local_rows;
    y.resize(n);
    for (int i = 0; i < n; i++)
        y[i] = y_local[i] + y_remote[i];

}

unordered_map<int,bool> normalize_outdegrees_allgather(LocalCSR& A, int rank, int num_procs, MPI_Comm comm) {
    map<int, int> local_outdegree;
    vector<int> send_col_idx;

    for (int i = 0; i < A.rowptr.size() -1; i++) {
        for (int j = A.rowptr[i]; j < A.rowptr[i +1]; j++) {
            int col = A.colind[j];
            local_outdegree[col] += 1;
        }
    }

    vector<int> s_count(num_procs, 0);
    vector<int> r_count(num_procs);

    vector<int> s_col_idx;
    vector<int> s_col_count;

    for (auto x: local_outdegree) {
        s_col_idx.push_back(x.first);
        s_col_count.push_back(x.second);
    }

    // Again we send the counts first
    int send_count = s_col_idx.size();
    MPI_Allgather(&send_count, 1, MPI_INT, r_count.data(), 1, MPI_INT, comm);

    // print_vector_int(r_count, rank);

    int total_receive = 0;
    for (int r: r_count) {
        total_receive += r;
    }

    vector<int> r_col_index(total_receive);
    vector<int> r_col_count(total_receive);

    vector<int> r_disp(num_procs, 0);

    // // Calculate displcaments
    for (int i = 1; i<num_procs; i++) {
        r_disp[i] = r_disp[i-1] + r_count[i-1];
    }
    
    // MPI gather the non zero column count from all
    MPI_Allgatherv(s_col_idx.data(), send_count, MPI_INT, r_col_index.data(), r_count.data(), r_disp.data(), MPI_INT, comm);
    MPI_Allgatherv(s_col_count.data(), send_count, MPI_INT, r_col_count.data(), r_count.data(), r_disp.data(), MPI_INT, comm);

    map<int, int> idx_to_count_map;

    for (int i = 0; i < r_col_index.size(); i++) {
        if ( idx_to_count_map.count(r_col_index[i]) == 0)
            idx_to_count_map[r_col_index[i]] = r_col_count[i];
        else {
            idx_to_count_map[r_col_index[i]] += r_col_count[i];
        }
    }


    for (int i = 0; i < A.colind.size(); i++) {
        if (idx_to_count_map.count(A.colind[i]) != 0) {
            A.vals[i] += idx_to_count_map[A.colind[i]];
        }
    }

    unordered_map<int, bool> is_dangling_map;

    for (int col = 0; col < A.global_cols; col++) {
        if (idx_to_count_map.count(col) == 0) {
            is_dangling_map[col] = 1;
        }
    }

    // balance duplicate addition of own weight
    for (int i = 0; i < A.vals.size(); i++) {
        A.vals[i] -= 1;
        A.vals[i] = 1 / A.vals[i];
    }

    return is_dangling_map;

}

void normalize_outdegrees_allreduce(LocalCSR &A, int rank, int num_procs,  vector<int>& outdegree, MPI_Comm comm)
{
    // 1. Local outdegree array (size N, initialized to 0)
    int N = A.global_cols;
    vector<int> local_outdeg(N, 0);

    // Count outgoing edges for each global column
    int local_rows = A.rowptr.size() - 1;
    for (int i = 0; i < local_rows; i++) {
        for (int k = A.rowptr[i]; k < A.rowptr[i + 1]; k++) {
            int col = A.colind[k];     // global column index
            local_outdeg[col]++;       // partial outdegree
        }
    }

    // 2. Global outdegree = sum of all partial outdegs
    MPI_Allreduce(local_outdeg.data(),
                  outdegree.data(),
                  N,
                  MPI_INT,
                  MPI_SUM,
                  comm);

    // 3. Normalize local values in CSR. A.vals[k] = 1 / global_outdeg[col]
    for (int i = 0; i < local_rows; i++) {
        for (int k = A.rowptr[i]; k < A.rowptr[i + 1]; k++) {

            int col = A.colind[k];
            int od = outdegree[col];
            if (od > 0) {
                A.vals[k] = 1.0 / od;
            }
        }
    }
    update_on_off_values(A, rank, num_procs);
}

void scalar_multiply(vector<double>& vec, double d) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] *= d;
    }
}

vector<double> teleportation_vector(int local_size, int global_size) {
    return vector<double>(local_size, (1-DAMPING)/global_size);
}

double dangling_sum_local(LocalCSR& A, vector<double>& page_rank_local, vector<int>& outdegree) {
    double sum = 0.0;
    for (int i = 0; i < A.local_rows; i++) {
        int global_index = A.first_row + i;
        if (outdegree[global_index] == 0) {
            sum += page_rank_local[i];
        }
    }
    return sum;
}

void add_vec(vector<double>& main, vector<double>& add) {
    for (int i = 0; i < main.size(); i++) {
        main[i] += add[i];
    }
}

void page_rank_step(LocalCSR& A, vector<double>& rank_vec, int rank, int num_procs, MPI_Comm comm) {
    vector<int> outdegree(A.global_cols);
    normalize_outdegrees_allreduce(A, rank, num_procs, outdegree, comm);
    vector<double> next(A.local_rows, 0.0);
    vector<double> tp = teleportation_vector(A.local_rows, A.global_rows);
    double local_sum = dangling_sum_local(A, rank_vec, outdegree);
    double total_dangling_sum = 0.0;
    MPI_Allreduce(&local_sum, &total_dangling_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    vector<double> dangling_correction(A.local_rows, DAMPING * (total_dangling_sum / A.global_rows));

    vec_mul(A, rank_vec, next, comm);
    scalar_multiply(next, DAMPING);

    add_vec(next, dangling_correction);
    add_vec(next, tp);
    rank_vec = next;
}

void page_rank(LocalCSR& A, vector<double>& rank_vec, MPI_Comm comm, int max_iters = 100, double tol = 1e-4) {

    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);
    vector<double> old(A.local_rows);

    for (int iter = 0; iter < max_iters; iter++) {

        // Savinf previous value for tolerance
        old = rank_vec;

        // One PageRank iteration
        page_rank_step(A, rank_vec, rank, num_procs, comm);

        // Compute local L1 diff
        double local_diff = 0.0;
        for (int i = 0; i < A.local_rows; i++) {
            local_diff += std::fabs(rank_vec[i] - old[i]);
        }

        // All Reduce to global differece to check tolerance
        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, comm);

        // Check convergence
        if (rank == 0) {
            printf("Iter %d: diff = %.12f\n", iter, global_diff);
        }

        if (global_diff < tol)
            break;
    }
}