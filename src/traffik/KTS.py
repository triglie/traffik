import numpy as np

from numba import jit

def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Main interface
    
    Detect change points automatically selecting their number
        K       - kernel between each pair of frames in video
        ncp     - maximum ncp
        vmax    - special parameter
    Optional arguments:
        lmin     - minimum segment length
        lmax     - maximum segment length
        desc_rate - rate of descriptor sampling (vmax always corresponds to 1x)

    Note:
        - cps are always calculated in subsampled coordinates irrespective to
            desc_rate
        - lmin and m should be in agreement
    ---
    Returns: (cps, costs)
        cps   - best selected change-points
        costs - costs for 0,1,2,...,m change-points
        
    Memory requirement: ~ (3*N*N + N*ncp)*4 bytes ~= 16 * N^2 bytes
    That is 1,6 Gb for the N=10000.
    """
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)
    
    N = K.shape[0]
    N2 = N*desc_rate  # length of the video before subsampling
    
    penalties = np.zeros(m+1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m+1)
    penalties[1:] = (vmax*ncp/(2.0*N2))*(np.log(float(N2)/ncp)+1)
    
    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)

    return (cps, costs)
    

# ------------------------------------------------------------------------------
# Extra functions (currently not used)

def estimate_vmax(K_stable):
    """K_stable - kernel between all frames of a stable segment"""
    n = K_stable.shape[0]
    vmax = np.trace(centering(K_stable)/n)
    return vmax


def centering(K):
    """Apply kernel centering"""
    mean_rows = np.mean(K, 1)[:, np.newaxis]
    return K - mean_rows - mean_rows.T + np.mean(mean_rows)


def eval_score(K, cps):
    """ Evaluate unnormalized empirical score
        (sum of kernelized scatters) for the given change-points """
    N = K.shape[0]
    cps = [0] + list(cps) + [N]
    V1 = 0
    V2 = 0
    for i in range(len(cps)-1):
        K_sub = K[cps[i]:cps[i+1], :][:, cps[i]:cps[i+1]]
        V1 += np.sum(np.diag(K_sub))
        V2 += np.sum(K_sub) / float(cps[i+1] - cps[i])
    return (V1 - V2)


def eval_cost(K, cps, score, vmax):
    """ Evaluate cost function for automatic number of change points selection
    K      - kernel between all frames
    cps    - selected change-points
    score  - unnormalized empirical score (sum of kernelized scatters)
    vmax   - vmax parameter"""
    
    N = K.shape[0]
    penalty = (vmax*len(cps)/(2.0*N))*(np.log(float(N)/len(cps))+1)
    return score/float(N) + penalty


def calc_scatters(K):
    """
    Calculate scatter matrix:
    scatters[i,j] = {scatter of the sequence with starting frame i and ending frame j} 
    """
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1); # TODO: use the fact that K - symmetric

    scatters = np.zeros((n, n))

    return calc_scatters2(K1, K2, scatters, n)


@jit(nopython=True)
def calc_scatters2(K1, K2, scatters, n):
    for i in range(n):
        for j in range(i, n):
            scatters[i,j] = K1[j+1]-K1[i] - (K2[j+1,j+1]+K2[i,i]-K2[j+1,i]-K2[i,j+1])/(j-i+1)

    return scatters


def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=True,
    out_scatters=None):
    """ Change point detection with dynamic programming
    K - square kernel matrix 
    ncp - number of change points to detect (ncp >= 0)
    lmin - minimal length of a segment
    lmax - maximal length of a segment
    backtrack - when False - only evaluate objective scores (to save memory)
    
    Returns: (cps, obj)
        cps - detected array of change points: mean is thought to be constant on [ cps[i], cps[i+1] )    
        obj_vals - values of the objective function for 0..m changepoints
        
    """   
    m = int(ncp)  # prevent numpy.int64
    
    (n, n1) = K.shape
    assert(n == n1), "Kernel matrix awaited."    
    
    assert(n >= (m + 1)*lmin)
    assert(n <= (m + 1)*lmax)
    assert(lmax >= lmin >= 1)
    
    if verbose:
        #print "n =", n
        print ("Precomputing scatters...")
    J = calc_scatters(K)
    
    if out_scatters != None:
        out_scatters[0] = J

    if verbose:
        print ("Inferring best change points...")
    # I[k, l] - value of the objective for k change-points and l first frames
    I = 1e101*np.ones((m+1, n+1))
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]

    if backtrack:
        # p[k, l] --- "previous change" --- best t[k] when t[k+1] equals l
        p = np.zeros((m+1, n+1), dtype=int)
    else:
        p = np.zeros((1,1), dtype=int)
    
    cpd_nonlin2(m, n, p, I, J, lmin, lmax, int(1) if backtrack else int(0))

    # Collect change points
    cps = np.zeros(m, dtype=int)
    
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]

    scores = I[:, n].copy() 
    scores[scores > 1e99] = np.inf
    return cps, scores
    

@jit(nopython=True)
def cpd_nonlin2(m, n, p, I, J, lmin, lmax, backtrack):
    for k in range(1, m+1):
        for l in range((k+1)*lmin, n+1):
            I[k, l] = 1e100 #nearly infinity
            for t in range(max(k*lmin, l-lmax), l-lmin+1):
                c = I[k-1, t] + J[t, l-1]
                if (c < I[k, l]):
                    I[k, l] = c
                    if (backtrack == 1):
                        p[k, l] = t