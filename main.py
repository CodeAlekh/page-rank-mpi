
# Made for the purpouse of preforming PageRank algorithim in paralell
from scipy.sparse import csr_matrix,csc_matrix
from scipy.sparse import lil_matrix
from scipy.io import mmread
import numpy as np

# Load the matrix in at specified path
# and convert to csc
matrix = mmread("edges/web-Google.mtx").tocsc()
matrix = matrix.astype(float)
N_1 = 1/matrix.shape[0]

# Create the vector for the matrix vector multiplication
vector = np.full((matrix.shape[0],1),N_1)

# Calculate the number of nonzero elements in each row and set
# their value equal to 1/nonzero. if the row is all zeros, set
# it to 1/N for all values
for i in range(0,matrix.shape[1]):
  print(i,"/",matrix.shape[1])
  col_val = len(matrix.getcol(i).nonzero()[0])
  if(col_val > 0):
    # given start end set the value of the column to 1/col_val
    start = matrix.indptr[i]
    end = matrix.indptr[i+1]
    if end-start > 0:
      matrix.data[start:end] = 1/col_val
  else:
    # Avoid sink by filling with 1/n
    matrix[:,i] = np.full((matrix.shape[0],1),N_1)

print("Starting Pagerank")
# Begin the actual iteration of the pagerank algorithim

m_it = 100 #max iterations
# matrix = matrix.tocsr()
for i in range(0,m_it):
  print("Iter: ",i)
  vector = matrix @ vector