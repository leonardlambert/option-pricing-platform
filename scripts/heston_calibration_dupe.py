import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.legendre import leggauss

class HestonCalibrator:
    """
    calibration of the Heston model using the method described in the paper REDACT THIS PART
    """

    def __init__(self, S0, r, q=0, nodes=64):
        self.S0 = S0
        self.r = r
        self.q = q

        self.nodes, self.weights = leggauss(nodes)
        self.u_max = 200
        self.u = 0.5 * (self.nodes + 1) * self.u_max
        self.w = self.weights * 0.5 * self.u_max

    def get_heston_intermediates(self, u_mesh, T_mesh, v0, vbar, rho, kappa, sigma):
        """
        computes the Heston intermediary quantities xi, d, A, A1, A2 and D as defined in the paper REDACT THIS PART
        """
        #basic complex quantity
        i = 1j 
        u = u_mesh

        #intermediate quantities -------------------------------------------------------
        xi = kappa - sigma * rho * iu #(eq 11a)
        d = np.sqrt(xi**2 + sigma**2 * (iu + u**2)) #(eq 11b)

        dT_half = (d * T_mesh)/2
        cosh_dT = np.cosh(dT_half)
        sinh_dT = np.sinh(dT_half)

        #A term construction
        A1 = (u**2 + iu) * sinh_dT #(eq 15b)
        A2 = (d/v0) * cosh_dT + (xi/v0) * sinh_dT #(eq 15c)

        A = A1 / A2 #(eq 15a)

        #B term construction
        B = (d * np.exp(kappa*T_mesh/2) / (v0*A2)) #(eq 15d)

        #D term construction
        D = np.log((d/v0) + ((kappa - d) * T_mesh) / 2) - np.log(((d + xi) / (2 * v0)) + ((d - xi) / (2 * v0)) * np.exp(-d * T_mesh)) #(eq 17b)

        #partial derivatives ------------------------------------------------------------------

        # w.r.t rho
        PD_d_rho = - (xi * rho * i * u) / d #(eq 27a)
        PD_A2_rho = - ((sigma * i * u * (2 + xi * T_mesh)) / (2 * d * v0)) * (xi * cosh_dT + d * sinh_dT) #(eq 27b)
        PD_B_rho = (np.exp(kappa * T_mesh / 2) / v0) * (((1 / A2) * PD_d_rho) - (d / A2**2) * PD_A2_rho) #(eq 27c)
        PD_A1_rho = - ((i * u * (u**2 + i * u) * T_mesh * xi * sigma) / (2 * d)) * cosh_dT #(eq 27d)
        PD_A_rho = ((1 / A2) * PD_A1_rho) - ((A / A2) * PD_A2_rho) #(eq 27e)

        # w.r.t kappa
        PD_A_kappa = (i / sigma * u) * PD_A_rho #(eq 28a)
        PD_B_kappa = (i / sigma * u) * PD_B_rho + (B * T_mesh) / 2 #(eq 28b)

        # w.r.t sigma
        PD_d_sigma = ((sigma / rho) - (1 / xi)) * PD_d_rho + (sigma * u**2) / d #(eq 30a)
        PD_A1_sigma = (((u**2 + i * u) * T_mesh) / 2) * PD_d_sigma * cosh_dT #(eq 30b)
        PD_A2_sigma = (rho / sigma) * PD_A2_rho - ((2 + T_mesh * xi) / (v0 * T_mesh * xi * i * u)) * PD_A1_rho + ((sigma * T_mesh * A1) / (2 * v0)) #(eq 30c)
        PD_A_sigma = (1 / A2) * PD_A1_sigma - (A / A2) * PD_A2_sigma #(eq 30d)

        return xi, d, A, A1, A2, D, PD_d_rho, PD_A_rho, PD_A2_rho, PD_B_kappa, PD_d_sigma, PD_A_sigma, PD_A2_sigma

    def characteristic_function(self, u, T, v0, vbar, rho, kappa, sigma): #Cui, Del Bano Rollin, Germano (2017)
        """
        Computes the Heston characteristic function phi(u, T, v0, kappa, vbar, xi, rho) as defined in the paper REDACT THIS PART
        """

        iu = 1j * u
        F = self.S0 * np.exp((self.r - self.q) * T) 
        xi, d, A, A1, A2, D = self.heston_intermediates(u, T, v0, vbar, rho, kappa, sigma)
        
        return np.exp(iu * np.log(F/self.S0) + (kappa*vbar*rho*T*iu)/sigma - A + (2*kappa*vbar)/sigma**2 * D) #(eq 18)
        
    def get_prices_and_gradients(self, K_vec, T_vec, params):
        """
        Computes the price and gradient of a European call or put option using the Heston model.
        """
        
        v0, vbar, rho, kappa, sigma = params
        i = 1j 

        T_mesh = T_vec[:, np.newaxis]
        K_mesh = K_vec[np.newaxis, :]
        u_mesh = self.u[np.newaxis, :]
        w_mesh = self.w[np.newaxis, :]
        iu = 1j * u_mesh
        
        #get intermediates
        xi, d, A, A1, A2, D, PD_d_rho, PD_A_rho, PD_A2_rho, PD_B_kappa, PD_d_sigma, PD_A_sigma, PD_A2_sigma = \
            self.get_heston_intermediates(u_mesh, T_mesh, v0, vbar, rho, kappa, sigma)
        
        #cf
        F_mesh = self.S0 * np.exp((self.r - self.q) * T_mesh)
        phi = np.exp(iu * np.log(F_mesh / self.S0) + (kappa * vbar * rho * T_mesh * iu) / sigma \
            - A + (2*kappa*vbar) / sigma**2 * D)
        
        #gradients
        h1 = -A / v0 # (eq 23a)

        h2 = (2 * kappa * D) / sigma**2 - ((kappa * rho * T * iu) / sigma) # eq 23b

        h3 = - PD_A_rho + ((2 * kappa * vbar) / d * sigma**2) * (PD_d_rho - (d / A2) * (PD_A2_rho)) \
            - ((kappa * vbar * rho * T * iu) / sigma**2) # eq 23c

        h4 = (1 / (sigma * iu)) * PD_A_rho + ((2 * vbar * D) / sigma**2) + ((2 * kappa * vbar) / \
            (B * sigma**2)) * PD_B_kappa - ((vbar * rho * T * iu) / sigma) # eq 23d

        h5 = - PD_A_sigma - ((4 * kappa * vbar * D) / sigma**3) + ((2 * kappa * vbar) / (d * sigma**2)) * \
             ( PD_d_sigma  - (d / A2) * PD_A2_sigma) + ((kappa * vbar * rho * T * iu) / sigma**2) # eq 23e

        #fourier integration
        K_ratio = np.log(K_mesh / self.S0)
        common_term = np.exp(-i * u_mesh * K_ratio) / (iu)

        price_integrands = np.sum(w_mesh * np.real(common_term * phi), axis=1)
        prices = 0.5 * (self.S0 * np.exp(-self.q * T_vec) - K_vec * np.exp(-self.r * T_vec)) + \
                (np.exp(-self.r * T_vec) / np.pi) * price_integrands

        #gradients computation
        grads = []
        for h in [h1, h2, h3, h4, h5]:
            grads_ints = np.sum(w_mesh * np.real(common_term * phi * h), axis=1)
            grads.append((np.exp(-self.r * T_vec) / np.pi) * grads_ints)

        jacobian = np.column_stack(grads)

        return prices, jacobian


    def calibration(self, market_prices, strikes, maturities, initial_guess):
        """
        calibration loop using Levenberg-Marquardt algorithm
        """
        
        K_vec = np.array(strikes)
        T_vec = np.array(maturities)
        market_prices = np.array(market_prices)
        
        def objective_function(params):
            prices, _ = self.get_prices_and_gradients(K_vec, T_vec, params)
            return prices - market_prices

        def jacobian(params):
            _, jacobian = self.get_prices_and_gradients(K_vec, T_vec, params)
            return jacobian

        res = least_squares(objective_function, initial_guess, jac=jacobian, method='lm',
                            xtol=1e-10, ftol=1e-10, gtol=1e-10)

        return res
        