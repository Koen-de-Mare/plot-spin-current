import math

import numpy as np
import matplotlib.pyplot as plt

# discretization step size
h = 0.1

# distribution and convolution infrastructure   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class SampledFunction:
    def __init__(self, samples):
        assert(samples > 0.0)
        samples = math.ceil(samples) # allows non-natural numbers to be passed

        self.samples = samples
        self.N = 1 + 2 * samples
        self.values = np.zeros(self.N)

    def clone(self):
        result = SampledFunction(self.samples)

        for n in range(self.N):
            result.values[n] = self.values[n]

        return result

    def multiply(self, c):
        result = self.clone()
        for n in range(result.N):
            result.values[n] *= c

        return result

    def sample(self, n):  # here the index is the 0 centered index; the sample with index 0 is at z=0
        return self.values[self.samples + n]

    def index_to_z(self, n): # here the index is the index of self.values, so the array that starts at 0
        return h * (n - self.samples)

    def make_z_list(self):
        zlist = []
        for n in range(self.N):
            zlist.append(self.index_to_z(n))
        return zlist

    def plot(self, zmax):
        plt.plot(self.make_z_list(), self.values)
        plt.xlim(0.0, zmax)
        plt.hlines(0.0, 0.0, zmax)
        plt.show()


def convolve(f1, f2):
    result = SampledFunction(f1.samples + f2.samples)
    for n in range(result.N):
        accumulator = 0.0

        # these are inclusive min and max
        m_min = max(0, n - 2 * f2.samples)
        m_max = min(2 * f1.samples, n)
        for m in range(m_min, m_max + 1):
            accumulator += f1.values[m] * f2.values[n - m]

        result.values[n] = accumulator * h

    return result


def add(f1, f2):
    if f1.samples > f2.samples:
        return add_2(f1, f2)
    else:
        return add_2(f2, f1)


def add_2(f_large, f_small):
    result = SampledFunction(f_large.samples)

    for n in range(f_large.N):
        result.values[n] = f_large.values[n]

    for n in range(f_small.N):
        result.values[(n - f_small.samples) + f_large.samples] += f_small.values[n]

    return result

# custom mathematical functions   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def a(x):
    if x == 0.0:
        return 0.0

    x2 = math.fabs(x)

    if x > 0.0:
        sign = 1.0
    else:
        sign = -1.0

    num_samples = 100
    accumulator = 0.0
    for n in range(1, num_samples + 1):
        u = n / num_samples
        accumulator += math.exp(-x2 / u)

    return sign * accumulator / num_samples


def b(x):
    if x == 0.0:
        return 0.0

    x2 = math.fabs(x)

    num_samples = 100
    accumulator = 0.0
    for n in range(1, num_samples + 1):
        u = n / num_samples
        accumulator += (1.0 / u) * math.exp(-x2 / u)

    return accumulator / num_samples

# main script   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == '__main__':
    # system properties   ----------------------------------------------------------------------------------------------

    material = "Ni"

    if material == "Fe":
        ds_up = 55.2
        ds_dn = 21.2

        alpha_up = 19.5
        alpha_dn = 22.3

        tau = 3681.0
    elif material == "Co":
        ds_up = 12.8
        ds_dn = 71.1

        alpha_up = 78
        alpha_dn = 5.2

        tau = 3560
    elif material == "Ni":
        ds_up = 13.7
        ds_dn = 152.5

        alpha_up = 48
        alpha_dn = 9.6

        tau = 157
    else:
        assert False

    ds_tot = ds_up + ds_dn
    alpha_tot = alpha_up + alpha_dn

    # called lambda in the formulae, but lambda is a protected token in python
    l_sd = math.sqrt(tau * ds_tot * alpha_up * alpha_dn / (alpha_tot * ds_up * ds_dn))

    vf = 1.0

    tau_up = 7.0
    tau_dn = 3.0

    E_up = alpha_up
    E_dn = alpha_dn
    E_tot = E_up + E_dn

    E_p = 1.0
    L = 5.0  # laser decay length
    P0 = 1.0  # laser power

    # set up distributions   -------------------------------------------------------------------------------------------

    # laser excitation profile
    laser_excitation = SampledFunction(10.0 * L / h)
    for n in range(laser_excitation.N):
        z = laser_excitation.index_to_z(n)
        laser_excitation.values[n] = P0 * math.exp(-math.fabs(z) / L)

    # hot electron current by delta_0 excitation
    J_hot_up_G = SampledFunction(10.0 * vf * tau_up / h)
    for n in range(J_hot_up_G.N):
        z = J_hot_up_G.index_to_z(n)
        J_hot_up_G.values[n] = a(z / (vf * tau_up)) * E_up / (2.0 * E_tot * E_p)

    J_hot_dn_G = SampledFunction(10.0 * vf * tau_dn / h)
    for n in range(J_hot_dn_G.N):
        z = J_hot_dn_G.index_to_z(n)
        J_hot_dn_G.values[n] = a(z / (vf * tau_dn)) * E_dn / (2.0 * E_tot * E_p)

    # spin current due to thermal electron diffusion from delta_0 hot electron decay
    J_diffusion_G = SampledFunction(10.0 * l_sd / h)
    for n in range(J_diffusion_G.N):
        z = J_diffusion_G.index_to_z(n)

        sign = 0.0
        if z > 0.0:
            sign = 1.0
        if z < 0.0:
            sign = -1.0

        J_diffusion_G.values[n] = sign * math.exp(-math.fabs(z) / l_sd) * ds_up * ds_dn / ds_tot

    # source term of thermal electron system, decay component (total includes delta component)
    diffusion_source_G = SampledFunction(10.0 * vf * max(tau_up, tau_dn) / h)
    for n in range(diffusion_source_G.N):
        z = diffusion_source_G.index_to_z(n)

        diffusion_source_G.values[n] = \
            ds_tot / (E_p * alpha_tot * 2.0 * vf * E_tot * ds_up * ds_dn) * (
                b(z / (vf * tau_up)) * E_up * alpha_dn / tau_up -
                b(z / (vf * tau_dn)) * E_dn * alpha_up / tau_dn
            )

    dirac_size = (
            (alpha_up / ds_up - alpha_dn / ds_dn) / alpha_tot -
            (E_up     / ds_up - E_dn     / ds_dn) / E_tot
        ) / E_p

    # perform convolutions   -------------------------------------------------------------------------------------------

    J_hot_up = convolve(J_hot_up_G, laser_excitation)
    J_hot_dn = convolve(J_hot_dn_G, laser_excitation)

    J_s_hot = add(J_hot_up, J_hot_dn.multiply(-1.0))
    J_s_screening = add(J_hot_up, J_hot_dn).multiply((alpha_dn - alpha_up) / alpha_tot)
    J_s_screened_hot = add(J_s_hot, J_s_screening)

    diffusion_source_a = convolve(diffusion_source_G, laser_excitation)
    diffusion_source_b = laser_excitation.multiply(dirac_size)
    diffusion_source = add(diffusion_source_a, diffusion_source_b)

    J_s_diffusion = convolve(J_diffusion_G, diffusion_source)

    J_s = add(add(J_s_hot, J_s_screening), J_s_diffusion)

    # plotting   -------------------------------------------------------------------------------------------------------

    if False:
        print("thermal diffusion system source term")
        print("red: hot electron decay")
        print("green: hot electron excitation")
        print("blue: total")
        plt.plot(diffusion_source_a.make_z_list(), diffusion_source_a.values, 'r', \
             diffusion_source_b.make_z_list(), diffusion_source_b.values, 'g', \
             diffusion_source.make_z_list(),   diffusion_source.values, 'b')
        plt.xlim(0.0, 100.0)
        plt.show()

    print("spin currents")

    plt.plot(J_s_hot.make_z_list(), J_s_hot.values, label='J_s_hot')
    plt.plot(J_s_screening.make_z_list(), J_s_screening.values, label='J_s_screening')
    plt.plot(J_s_screened_hot.make_z_list(), J_s_screened_hot.values, label='J_s_screened_hot')
    plt.plot(J_s_diffusion.make_z_list(), J_s_diffusion.values, label='J_s_diffusion')
    plt.plot(J_s.make_z_list(), J_s.values, label='J_s')
    plt.legend()
    plt.xlim(0.0, 25.0)
    plt.hlines(0.0, 0.0, 100.0)
    plt.show()

