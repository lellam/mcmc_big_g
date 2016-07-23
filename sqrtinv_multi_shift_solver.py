import numpy as np

class ShiftConjugateGradient(object):
    """ Multi-shift conjugate gradient solver for (A+aI)x = b """
    
    def __init__(self, shifts):
        self.shifts = shifts
        self.n_shift = len(self.shifts)
        return
        
        
    def solve(self, A, b, it_max, tol):

        # Initialize CG
        n_vec = len(b)
        r = b.copy()
        p = b.copy()
        beta_old = 1.
        alpha = 0.
        c_cur = np.dot(r, r)

        x_k = np.zeros((self.n_shift, len(b)))
        p_k = np.zeros((self.n_shift, n_vec))
        p_k = np.tile(b, (self.n_shift, 1))
        alpha_k=np.zeros(self.n_shift)
        beta_k=np.zeros(self.n_shift)
        xi_new_k = np.zeros(self.n_shift)
        xi_cur_k = np.ones(self.n_shift)
        xi_old_k = np.ones(self.n_shift)

        # CG Algorithm here
        ii=0
        while(ii < it_max):

            a_p = A.dot(p)
            beta_cur = -c_cur/np.dot(p, a_p)
            
            denom = beta_cur*alpha*(xi_old_k-xi_cur_k)
            denom += beta_old*xi_old_k*(1.-self.shifts*beta_cur)    
            xi_new_k = beta_old*xi_cur_k*xi_old_k/denom

            xi_cur_k[xi_cur_k==0] += np.finfo(float).eps  # Avoid div 0
            beta_k=beta_cur*xi_new_k/xi_cur_k

            x_k = x_k-beta_k[:,None]*p_k

            #Update residual, every 50 steps recompute from scratch
            if(ii%50 > 0):
                r = r+beta_cur*a_p
            else:
                r = b-A.dot(x_k[0])

            # Termination criterion
            norm=np.linalg.norm(r/b)
            if(norm < tol):
                if(ii%50 > 0):              #Recompute residual and norm
                    r = b-A.dot(x_k[0])
                    norm=np.linalg.norm(r/b)
                if(norm < tol):             #Final termination check
                    break

            c_new = np.dot(r, r)
            alpha = c_new/c_cur
            p=r+alpha*p

            alpha_k=alpha*xi_new_k*beta_k/(beta_cur*xi_cur_k)

            p_k = xi_new_k[:,None]*r[None,:]+alpha_k[:,None]*p_k

            xi_old_k=xi_cur_k
            xi_cur_k=xi_new_k.copy()
            beta_old=beta_cur
            c_cur = c_new

            ii+=1
        
        return x_k
    