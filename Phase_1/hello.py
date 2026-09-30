# The fundamentals of this algorithm are adapted from the following paper:
# Zdunek, R.; Fonał, K.
# Incremental Nonnegative Tucker
# Decomposition with BlockCoordinate Descent and Recursive
# Approaches. Symmetry 2022, 14, 113.
# https://doi.org/10.3390/
# An adaptive factor is used to reduce the influence of previous information when the core and factor 
# matrices are updated at each time step

# The hyper spectral data used in this script is from the following database:
# Fasnacht, L., Vogt, ML., Renard, P. et al. 
# A 2D hyperspectral library of mineral reflectance, from 900 to 2500 nm. Sci Data 6, 268 (2019). 
# https://doi.org/10.1038/s41597-019-0261-9 


import numpy as np
from scipy.optimize import nnls
import tensorly as tl
from tensorly.decomposition import non_negative_tucker_hals
import h5py
from tensorly.metrics.regression import RMSE
from pathlib import Path

tl.set_backend('numpy')
EPS = 1e-12

# This function will load the required HDF5 hyper cubes and remove NaN values.
# Using the provided transposition code provided by Fasnatcht et al.

def load_and_clean(folder):

    with h5py.File(folder, 'r') as f:
        data = f['/hdr'][()]
    data = np.transpose(data, (1, 2, 0)) 
    width, height, bands = data.shape
    flat = data.reshape(-1, bands)
    for i in range(flat.shape[0]):
        row = flat[i, :]
        nan = np.isnan(row)
        if np.any(nan):
            good = np.where(~nan)[0]
            if len(good) == 0:
                row[:] = 0
            else:
                row[nan] = np.interp(np.where(nan)[0], good, row[good])
            if np.any(np.isnan(row)):
                    row[nan] = row[good[0]]
            flat[i, :] = row

    return data.reshape(width, height, bands)

# This function allows us to load the data cubes on demand without storing them in RAM, by using the keyword next
# Allowing for the sufficient without a bottleneck in processing speed 

def cube_generator(folder):
    for path in Path(folder).glob('*.h5'):
        yield load_and_clean(str(path))


def recursive_update(X_n, U, G, P, Q, start, nonneg=True, k_inner=10, gamma=1e-3, lam = 1):
    N = len(U)
    block_size = X_n.shape[-1]
    last = N - 1

    # The goal is the increment of the growing mode ( sub set of growing tensors being streamed )

    W = G
    for m in range(N - 1):
        W = tl.tenalg.mode_dot(W, U[m], mode=m)

    Wn = tl.unfold(W, mode=last)         
    Xn = tl.unfold(X_n, mode=last)    

    # Solve (Wn Wnᵀ) U_newᵀ = Wn Xnᵀ   (each column independently)
    A_last = Wn @ Wn.T
    U_new = np.zeros((block_size, Wn.shape[0]))
    for row in range(block_size):
        rhs = Wn @ Xn[row, :]
        sol, _ = nnls(A_last, rhs)
        U_new[row, :] = sol

    # Insert new rows into the full factor matrices along each mode N unfolding 
    # So we grow the tensor along each mode [0 - 2]
    
    U[last][start:start + block_size, :] = U_new

    # At this point we have managed to accumulate the new data blocks of information
    # Stay with me here guys, keep up 

    U_block_last = U_new
 
    
    for n in range(N - 1):
        
        W = G
        for m in range(N):
            if m != n and m != last:
                W = tl.tenalg.mode_dot(W, U[m], mode=m)
        W = tl.tenalg.mode_dot(W, U_block_last, mode=last)

        Xn = tl.unfold(X_n, mode=n)
        Wn = tl.unfold(W, mode=n)

        # The auxiliary matrices are updated using the accumulated information from each
        # mode of the growing tensor.
        
        P[n] = lam*P[n] + Xn @ Wn.T   # Equation 20
        Q[n] = lam*Q[n] + Wn @ Wn.T   # The variable lambda affects the effect of past information in updating the core and factor matrices
                                      # A value of 1 means that there is no decaying factor, a value of 0.1 means that past information 
                                      # has no influence on current information 

        # Implementation of the Gauss Seidel update for solving the least squares problem

        for _ in range(k_inner):
            Q_reg = Q[n] + gamma * np.eye(Q[n].shape[0])
            for j in range(U[n].shape[1]):

                num = P[n][:, j] - U[n] @ Q_reg[:, j]

                U[n][:, j] = U[n][:, j] + num / (Q_reg[j, j] + EPS)

                if nonneg:

                    U[n][:, j] = np.maximum(U[n][:, j], EPS)
        
        U[n] = U[n] / (U[n].sum(axis=0, keepdims=True) + EPS)

    # Update the core tensor using the current block of information

    A_core = 1
    Z_core = tl.tensor(X_n)
    grams = []   
    for n in range(N):
        if n == last:
            U_n = U_block_last
        else:
            U_n = U[n]
        
        Z_core = tl.tenalg.mode_dot(Z_core, U_n.T, mode=n)
        grams.append(U_n.T @ U_n)   

    A_core = grams[0]
    for g in grams[1:]:
        A_core = np.kron(A_core, g)

    z_vec = Z_core.reshape(-1)
    if nonneg:
        g_vec, _ = nnls(A_core, z_vec)
    else:
        g_vec = np.linalg.solve(A_core, z_vec)
    G = g_vec.reshape(G.shape)

    # Equation 30 - this is the vectorization form of the tensor 

    return U, G, P, Q


if __name__ == "__main__":

    data_folder = 'C:\\Users\\Marcus\\Desktop\\TuckerDecomp\\data'

    # Replace this with your file directory 

    ranks = [3, 3, 3]

    # Multi dimensional Tucker tensor rank for the core tensor it allows us to set the compression ration for the input data
    # For classification Andile, KK set this value to [20,20,15]. It allows us to set the expected end members expected in an ore stream.
    # Basically we expect 20 minerals along the width, 20 along the height and 15 different minerals in the abundance mapping.
    # Ideally we want to identify at least 4 end members

    Lt = 100   # Set the number of data cube slices 
    L = 100
    theta = 10
    overlap = int(L * theta / 100)

    # Each data cube is processed in splitted data of 100 slice increments with an overlap
    # to ensure that, thee data is processed without corruption.
    # The value is calculated by the overlap value in this code that value is set to 10
    # This is set as the default value because Zdunek .et al use the same value in their testing implementation 

    print("Starting data cube conveyor belt: ")

    for cube in cube_generator(data_folder):
        
        data = cube 

        # Splitting of the data cube into the first 100 along the last mode 
        X_burn = data[..., :Lt]
        tensor_hals, errors_ = non_negative_tucker_hals( X_burn,rank=ranks,algorithm='fista',n_iter_max=30, tol=1e-5, init='random',
        return_errors=True
        )

        # The initial data cube is split into the first 100 batches and then and a Non Negative Tucker decomposition is implemented using the FISTA method
        # This creates an initial core and factor matrices that are then updated with each increase in information

        # tensor_reconstruction = tl.tucker_to_tensor(tensor_hals)
        # print("Reconstruction Error : " + " " + str(RMSE(tensor_hals tucker_reconstruction_mu)))
        # Ummm guys uncomment this during the so you can get the reconstruction error, regardless of the forgetting factor the reconstruction error  should not
        # be exponentially high 
        

        U = list(tensor_hals.factors)   # [U1, U2, U3] factor matrices 
        G = np.asarray(tensor_hals.core)   # Core tensor 
        N = len(U)
        last = N - 1

        # Initialisation of auxiliary matrices P[n] and Q[n] for updating the projection matrices U[n]

        P = [None] * N
        Q = [None] * N
        for n in range(N - 1):

            # As stated by Zdunek et.al the tensor is is built up to the time stamp t-1 and the data provided at t > t - 1 is considered the streaming information
            # So we build the concatenated tensor produced by equations 9-10
            W = G
            for m in range(N):
                if m != n and m != last:
                    W = tl.tenalg.mode_dot(W, U[m], mode=m)
            
            W = tl.tenalg.mode_dot(W, U[last], mode=last)

            Xn = tl.unfold(X_burn, mode=n)
            Wn = tl.unfold(W, mode=n)
            P[n] = Xn @ Wn.T
            Q[n] = Wn @ Wn.T

        total_bands = data.shape[-1]
        U_full = U.copy()
        U_full[last] = np.vstack([U[last], np.zeros((total_bands - Lt, U[last].shape[1]))])

        # Loop controls the incremental step of data cubes until the last data cube is processed
        # Ensures that we dont process beyond the initial data cube and we dont reuse it 
        start = Lt
        block_idx = 0
        while start < total_bands:
            end = min(start + L, total_bands)
            X_n = data[..., start:end]
            print(f"  Block {block_idx+1}: bands {start+1}–{end} (size={end-start})")
            U_full, G, P, Q = recursive_update( X_n, U_full, G, P, Q, start=start,nonneg=True,k_inner=10,
            gamma=1e-3, lam=1
            )  # Pass a different value of lamda to test the effectiveness in the reconstruction error 
               # The value should not blow up when printed out 
            start += (L - overlap)
            block_idx += 1

        # Classification of the data cubes should happen in this loop
        # Ideally it should be, data cube in, decompose, classify then call the next data cube 

        print(f"Done. Core shape {G.shape}, factor shapes: {[u.shape for u in U_full]}")

        # An adaptive forgetting factor update will be presented for more adapt testing
        # A forgertting factor of 0.9 was found to give a consistent coherence between data cubes that have different minerals 
        
        