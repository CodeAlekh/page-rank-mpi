#include <iostream>
#include <fstream>
#include <mpi.h>

// Custom installed libraries
#include <fast_matrix_market/app/Eigen.hpp>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>

void pagerank(Eigen::SparseMatrix<double> &mat,
              Eigen::VectorXd &pgrnk, //Pagerank vector
              Eigen::VectorXd &zero_vec, //Vector of zeros
              int process, //MPI
              int global   //MPI
            )
{
  // Convergence step

  return;
}

/*
  compute_col_weights(
    Eigen::SparseMatrix<double>& mat = the matrix that we will be preforming SpMV
    int start    = the row we will start at
    int num_cols =
  )

  Computes the column weights and updates the matrix, returns a list of
  all columns that have zero for zipper merge when doing SpMV at a later function
*/
void compute_col_weights(Eigen::SparseMatrix<double> &mat,
                         int start, int num_cols, Eigen::VectorXd &zero_cols)
{
  //
  int c_size = mat.cols();
  for (int i = 0; i < num_cols; i++)
  {
    double sum = 0;
    double start_weight = 0;
    for (Eigen::SparseMatrix<double>::InnerIterator it(mat, (start + i)); it; ++it)
    {
      sum += it.value();
    }
    if (sum == 0)
    {
      zero_cols[i] = (1 / c_size);
    }
    else
    {
      start_weight = 1 / sum;
      for (Eigen::SparseMatrix<double>::InnerIterator it(mat, (start + i)); it; ++it)
      {
        it.valueRef() *= start_weight;
      }
    }
  }
}

int main(int argc, char **argv)
{
  Eigen::SparseMatrix<double> mat;
  std::ifstream input_stream("../edges/web-Google.mtx");
  fast_matrix_market::read_matrix_market_eigen(input_stream, mat);
  int ROWS = mat.rows();
  int COLS = mat.cols();
  // Sparse vector of zero columns, used when zipper merging columns
  Eigen::VectorXd zeroList(ROWS); //Initialize zeros
  compute_col_weights(mat, 0, COLS, zeroList);
  Eigen::VectorXd pgrnk(COLS);
  pgrnk.setConstant(1/COLS);

  return 0;
}