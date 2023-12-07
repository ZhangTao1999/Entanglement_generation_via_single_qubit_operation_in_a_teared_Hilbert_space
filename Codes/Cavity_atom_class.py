# %% Import libs
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from math import factorial
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from matplotlib.colors import Normalize
import cmath
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import dual_annealing
# %% Defining functions
# σ matrices
I = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j], [1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
n = np.array([[0,0],[0,1]])

def kronecker_delta(x,y):
    if abs(x-y)<1e-5: return 1
    else: return 0

# Defining Physical constants
hbar = 6.62607015*(10**(-34))/2/np.pi   # Yet I'd like to set hbar = 1 in the calculation follows
# Defining Atomic Constants
Gamma = 2*np.pi*6*10**6    # 6MHz linewidth of the ^{87}Rb upper state 5P3/2
kappa = 2*np.pi*0.1*10**6   # 0.1MHz linewidth of the cavity

class cavity_atom_light_system():   # Note that here most of the parameters are alterable, except for the atomic parameters.
    def __init__(self, atom_number, cooperativity, Delta, Omega):
        self.atom_number = atom_number              
        self.cooperativity = cooperativity              
        self.Delta = Delta                              # Delta should be much smaller than 6.8GHz/2, otherwise self.t() no longer holds
        self.omega_s = self.cooperativity*kappa*Gamma / (4*self.Delta)     # omega_s = g^2/Δ = ηκΓ/(4Δ)
        self.Omega = Omega                              # Note that we set Omega as a fixed parameter here.
        self.state_index = np.arange(self.atom_number+1)[::-1]    # The first one being the Dicke state with spin_z==+N/2
        # Calculating S_xyz, be ware of the order of Dicke states
        self.S_z = np.diag(np.array([self.atom_number/2 - iter for iter in np.arange(atom_number+1)]))
        S_plus  = np.array( [ [np.sqrt(atom_number/2 * (atom_number/2 +1) - jj * ( jj + 1 ) ) * kronecker_delta(ii, jj+1) for jj in np.arange(atom_number/2,-atom_number/2-1,-1)] for ii in np.arange(atom_number/2,-atom_number/2-1,-1)] )
        S_minus = np.array( [ [np.sqrt(atom_number/2 * (atom_number/2 +1) - jj * ( jj - 1 ) ) * kronecker_delta(ii, jj-1) for jj in np.arange(atom_number/2,-atom_number/2-1,-1)] for ii in np.arange(atom_number/2,-atom_number/2-1,-1)] )
        self.S_plus = S_plus
        self.S_minus = S_minus
        self.S_x = (S_plus + S_minus) / 2
        self.S_y = (S_plus - S_minus) / (2j)

        self.psi0 = self.initialize_psi0()
        self.psi = deepcopy(self.psi0)  # Remember to reset psi after each evolution turn

    def initialize_psi0(self):      # Initialize a CSS orienting at the north pole, that is, [1,0,0,0,...,0] in the Dicke representation
        psi0 = np.zeros(self.atom_number+1)
        psi0[0] = 1
        return psi0.reshape(-1,1)     # State vectors are column vectors, of course
    
    def H_AC(self, delta, order): # "delta" is the depth of the AC stark shift, and "order" is the Dicke state order with which the AC light resonates
        H_prime_exp = delta*(1 - 1j*Gamma / (2*self.Delta) ) * np.diag([n*np.abs(self.t(order*self.omega_s, n))**2 for n in range(self.atom_number+1)[::-1]]) / (order * np.abs(self.t(order * self.omega_s, order))**2 )   # Check that the first one being +N/2
        return H_prime_exp  # order = 1 recovers the Eq.6 in the main text of Creation of Greenberger-Horne-Zeilinger states with thousands of atoms by entanglement amplification

    def evaluation(self, mode = "GHZ"):
        if mode == "GHZ":
            density_mat = self.psi@np.transpose(np.conj(self.psi))
            fidelity = 1/2 * np.abs(density_mat[0,0] + density_mat[self.atom_number, self.atom_number] + np.abs(density_mat[0, self.atom_number] + density_mat[self.atom_number, 0]) )
            return fidelity
        elif mode == "SSS": 
            # # This is ξs^2, but cannot reveal the importance of the unitarity.
            # rho = self.psi@np.transpose(np.conj(self.psi))
            # Jzmean = np.real(np.trace(self.S_z@rho))
            # Jxmean = np.real(np.trace(self.S_x@rho))
            # Jymean = np.real(np.trace(self.S_y@rho))
            # Jabs = np.real(np.sqrt(Jzmean**2 + Jxmean**2 + Jymean**2))
            # print("Jzmean:", Jzmean)
            # print("Jxmean:", Jxmean)
            # print("Jymean:", Jymean)
            # print("Jabs:", Jabs)
            # Polar_Angle = np.arccos(Jzmean/Jabs)
            # temp = Jxmean/(Jabs*np.sin(Polar_Angle))
            # if temp>1:temp-=1e-6
            # if temp<-1:temp+=1e-6
            # Azimuthal_Angle = np.arccos(temp)
            # if Jymean<=0: Azimuthal_Angle = 2*np.pi - Azimuthal_Angle
            # n_1 = np.array([-np.sin(Azimuthal_Angle), np.cos(Azimuthal_Angle), 0])
            # n_2 = np.array([np.cos(Polar_Angle)*np.cos(Azimuthal_Angle), np.cos(Polar_Angle)*np.sin(Azimuthal_Angle), -np.sin(Polar_Angle)])
            # J_n1 = self.S_x*n_1[0] + self.S_y*n_1[1] + self.S_z*n_1[2]
            # J_n2 = self.S_x*n_2[0] + self.S_y*n_2[1] + self.S_z*n_2[2]
            # covariance = 1/2 * np.real(np.trace(rho@J_n1@J_n2+rho@J_n2@J_n1))
            # Squeezing_Parameter = 2/self.atom_number * (  np.real(np.trace(rho@(J_n1@J_n1+J_n2@J_n2))) - np.sqrt((np.real(np.trace(rho@(J_n1@J_n1 - J_n2@J_n2 + 4*covariance**2 ))))**2 ) )
            # return {'Squeezing_Parameter' : Squeezing_Parameter, 'Polar_Angle:' : Polar_Angle, 'Azimuthal_Angle': Azimuthal_Angle}
            
            # This is ξ_R^2, which is sensitive to the unitarity for the most.
            rho = self.psi@np.transpose(np.conj(self.psi))
            unitarity = np.real(1-np.trace(rho))
            Jzmean = np.real(np.trace(self.S_z@rho))
            Jxmean = np.real(np.trace(self.S_x@rho))
            Jymean = np.real(np.trace(self.S_y@rho))
            Jabs = np.real(np.sqrt(Jzmean**2 + Jxmean**2 + Jymean**2))
            # print("Jzmean:", Jzmean)
            # print("Jxmean:", Jxmean)
            # print("Jymean:", Jymean)
            # print("Jabs:", Jabs)
            Polar_Angle = np.arccos(Jzmean/Jabs)
            if Polar_Angle == 0:
                Azimuthal_Angle = 0
            else:
                temp = Jxmean/(Jabs*np.sin(Polar_Angle))
                if temp>1:temp-=1e-6
                if temp<-1:temp+=1e-6
                Azimuthal_Angle = np.arccos(temp)
                if Jymean<=0: Azimuthal_Angle = 2*np.pi - Azimuthal_Angle
            n_1 = np.array([-np.sin(Azimuthal_Angle), np.cos(Azimuthal_Angle), 0])
            n_2 = np.array([np.cos(Polar_Angle)*np.cos(Azimuthal_Angle), np.cos(Polar_Angle)*np.sin(Azimuthal_Angle), -np.sin(Polar_Angle)])
            J_n1 = self.S_x*n_1[0] + self.S_y*n_1[1] + self.S_z*n_1[2]
            J_n2 = self.S_x*n_2[0] + self.S_y*n_2[1] + self.S_z*n_2[2]
            covariance = 1/2 * np.real(np.trace(rho@J_n1@J_n2+rho@J_n2@J_n1))
            Squeezing_Parameter = self.atom_number/(2*Jabs**2) * (  np.real(np.trace(rho@(J_n1@J_n1+J_n2@J_n2))) - np.sqrt(np.real(np.trace(rho@(J_n1@J_n1 - J_n2@J_n2))**2 + 4*covariance**2 ))   )
            Squeezing_Parameter = Squeezing_Parameter*unitarity + 1 - unitarity
            return {'Squeezing_Parameter' : Squeezing_Parameter, 'Polar_Angle:' : Polar_Angle, 'Azimuthal_Angle': Azimuthal_Angle}

    def calc_SP_from_scratch(self, unitarity=0):
        rho = self.psi @ np.transpose(np.conj(self.psi))
        rho_unitarity = np.trace(rho)   # We should judge the unitarity from trace(ρ), which marks the loss of atoms by scattering Γ
        Jzmean = np.real(np.trace(self.S_z @ rho))
        Jxmean = np.real(np.trace(self.S_x @ rho))
        Jymean = np.real(np.trace(self.S_y @ rho))
        # print([Jzmean,Jxmean,Jymean])
        Jabs = np.real(np.sqrt(Jzmean**2 + Jxmean**2 + Jymean**2))
        # print("Jabs:",Jabs)
        # if Jabs < self.atom_number/2*unitarity :
        #     return [1000,Jabs]
        if rho_unitarity < unitarity:
            return [1000,rho_unitarity]
        else:
            Polar_Angle = np.arccos(Jzmean/Jabs)
            if Polar_Angle == 0:
                Azimuthal_Angle = 0
            else:
                temp = Jxmean/(Jabs*np.sin(Polar_Angle))
                if temp>1:temp-=1e-6
                if temp<-1:temp+=1e-6
                Azimuthal_Angle = np.arccos(temp)
                if Jymean<=0: Azimuthal_Angle = 2*np.pi - Azimuthal_Angle
            # print("Jz,Jx,Jy,Jabs:",Jzmean,Jxmean,Jymean,Jabs )
            n1 = np.array([-np.sin(Azimuthal_Angle), np.cos(Azimuthal_Angle), 0])
            n2 = np.array([np.cos(Polar_Angle) * np.cos(Azimuthal_Angle), np.cos(Polar_Angle) * np.sin(Azimuthal_Angle), -np.sin(Polar_Angle)])
            sin_phi = np.array([np.sin(phi) for phi in np.arange(0, np.pi, 0.01)])
            cos_phi = np.array([np.cos(phi) for phi in np.arange(0, np.pi, 0.01)])

            # Calculating variances for different directions
            variances = []
            for sinp, cosp in zip(sin_phi, cos_phi):
                direction = sinp * n1 + cosp * n2
                J_n = self.S_x * direction[0] + self.S_y * direction[1] + self.S_z * direction[2]
                var = np.real(np.trace(rho @ (J_n @ J_n))) - np.real(np.trace(rho @ J_n)) ** 2
                variances.append(var)

            # Finding the minimum variance
            min_variance = min(variances)

            # Calculating squeezing parameters
            xi_S_squared = 4 * min_variance / self.atom_number
            xi_H_squared = (self.atom_number / (2 * Jabs)) * xi_S_squared
            xi_R_squared = (self.atom_number ** 2 / (4 * Jabs ** 2)) * xi_S_squared

            return [xi_S_squared, rho_unitarity]

    def OAT(self, xit):
        self.psi = expm(-1j*xit*self.S_z@self.S_z)@self.psi

    def TACT(self, xit):
        self.psi = expm(-0.5*xit*(self.S_plus@self.S_plus - self.S_minus@self.S_minus))@self.psi
        # self.psi = expm(-1j*xit*(self.S_x@self.S_x - self.S_y@self.S_y))@self.psi


    # Cavity amplitude transmission function t
    def t(self, ksi, n_up):   # ksi is the detuning between light and cavity mode ω-ω_c
        return 1 / ( 1 + (n_up*self.cooperativity) / (1+4*(self.Delta+ksi)**2/Gamma**2) - 2*1j* ( ksi/kappa - n_up*self.cooperativity*(self.Delta+ksi)/Gamma / (1+4*(self.Delta+ksi)**2/Gamma**2) ) )

    def show_occupation(self):
        plt.figure()
        plt.scatter(self.state_index-self.atom_number/2, np.abs(np.transpose(self.psi))**2)     # Occupation is the probability distribution, for sure
        plt.show()

    def system_reset(self):
        self.psi = deepcopy(self.psi0)

    def semi_customized_evolution_trajectory(self, params): # This is for BO, resembling that of Eq.(9) in SI
        # self.psi = deepcopy(self.psi0)
        t1, t2, t3, t4, phi3, phi4, delta2, delta4 = params
        # self.show_occupation()
        self.psi = expm(-1j*self.Omega*self.S_x*t1) @ self.psi
        # self.show_occupation()
        self.psi = expm(-1j*(self.Omega*self.S_x+self.H_AC(delta2,order = 1))*t2) @ self.psi
        # self.show_occupation()
        self.psi = expm(1j*self.S_z*phi3) @ self.psi
        self.psi = expm(-1j*self.Omega*self.S_x*t3) @ self.psi
        self.psi = expm(1j*self.S_z*(phi4-phi3) ) @ self.psi
        # self.show_occupation()
        self.psi = expm(-1j*(self.Omega*self.S_x+self.H_AC(delta4,order = 1))*t4) @ self.psi
        # self.show_occupation()
        # print(np.sum(np.abs(self.psi)**2))
        return self.evaluation()

    def one_Sx_evolution(self, params):
        t_x = params
        self.psi = expm(-1j*self.Omega*self.S_x*t_x) @ self.psi

    def one_sequence_evolution(self, params): # Sx——Sx+Hac
        t_x, order, delta, t = params
        self.psi = expm(-1j*(self.Omega*self.S_x + self.H_AC(delta, order)) * t) @ expm(-1j*self.Omega*self.S_x*t_x) @ self.psi
    
    def one_H_AC_evolution(self, params):
        t, order, delta = params
        self.psi = expm(-1j * self.H_AC(delta, order)*t) @ self.psi

    def one_step(self, params): # Sx+Hac
        t, diagonal_element = params
        self.psi = expm(-1j * ( self.Omega*self.S_x + np.diag(diagonal_element) ) * t) @ self.psi

    def husimi_q(self, theta, phi):
        q_func = np.zeros((len(theta), len(phi)), dtype=np.complex128)
        for j, t in enumerate(theta):
            for k, p in enumerate(phi):
                alpha = np.cos(t / 2) * np.exp(1j * p / 2)
                beta = np.sin(t / 2) * np.exp(-1j * p / 2)
                coherent_state = np.array([alpha ** m * beta ** (self.atom_number - m) * np.sqrt(factorial(self.atom_number) / (factorial(m) * factorial(self.atom_number - m))) for m in range(self.atom_number + 1)[::-1]]).reshape(-1, 1)
                q_func[k,j] = np.abs(np.conj(coherent_state).T @ self.psi) ** 2
        return np.abs(q_func)

    def visualize_origin(self):
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 100)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        q_func = self.husimi_q(theta, phi)
        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)
        # Normalize the q_func values for the colormap
        norm = Normalize(vmin=np.min(q_func), vmax=np.max(q_func))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.jet(norm(q_func)), alpha=0.6, linewidth=0, antialiased=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Husimi-Q Representation')
        plt.show()

    def visualize(self):
        # Create a colormap with colors from white to yellow.
        colormap = LinearSegmentedColormap.from_list("", ["white","red"])

        theta = np.linspace(0, np.pi, 1000)
        phi = np.linspace(0, 2 * np.pi, 1000)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        q_func = self.husimi_q(theta, phi)
        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)
        # Normalize the q_func values for the colormap
        norm = Normalize(vmin=np.min(q_func), vmax=np.max(q_func))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colormap(norm(q_func)), alpha=0.6, linewidth=0, antialiased=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Husimi-Q Representation')

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()


