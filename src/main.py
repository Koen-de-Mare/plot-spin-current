import math

import numpy as np
import matplotlib.pyplot as plt

# discretization step size
h = 0.25

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
        # plt.ylim(-1.0, 1.0)
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

# custom mathematical functions


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


if __name__ == '__main__':
    material = "Fe"

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

    tau_up = 10.0
    tau_dn = 4.0

    # currently not allowed to be anything else
    # the derivation for the thermal electron system evolution assumes E_sigma = Ds_sigma
    E_up = ds_up
    E_dn = ds_dn
    E_tot = E_up + E_dn

    E_p = 1.0
    L = 15.0  # laser decay length
    P0 = 1.0  # laser power

    # set up distributions

    # laser excitation profile
    f1 = SampledFunction(10.0 * L / h)
    for n in range(f1.N):
        z = f1.index_to_z(n)
        f1.values[n] = P0 * math.exp(-math.fabs(z) / L)

    print("f1")
    f1.plot(50.0)

    # spin current by hot electrons excited by delta_0
    f2 = SampledFunction(10.0 * vf * max(tau_up, tau_dn) / h)
    for n in range(f2.N):
        z = f2.index_to_z(n)
        f2.values[n] = (E_up * a(z / (vf * tau_up)) - E_dn * a(z / (vf * tau_dn))) / (2.0 * E_tot * E_p)

    print("f2")
    f2.plot(20.0)

    # spin current due to thermal diffusion diffusion from delta_0 hot electron decay
    f3 = SampledFunction(10.0 * l_sd / h)
    for n in range(f3.N):
        z = f3.index_to_z(n)

        sign = 0.0
        if z > 0.0:
            sign = 1.0
        if z < 0.0:
            sign = -1.0

        f3.values[n] = sign * math.exp(-math.fabs(z) / l_sd) * ds_up * ds_dn / ds_tot

    print("f3")
    f3.plot(50.0)

    # decay positions of hot electrons excited by delta_0
    f4a = SampledFunction(10.0 * vf * max(tau_up, tau_dn) / h)
    for n in range(f4a.N):
        z = f4a.index_to_z(n)

        f4a.values[n] = \
            ds_tot / (E_p * alpha_tot * 2.0 * vf * E_tot * ds_up * ds_dn) * (
                b(z / (vf * tau_up)) * E_up * alpha_dn / tau_up -
                b(z / (vf * tau_dn)) * E_dn * alpha_up / tau_dn
            )

    #f4b = SampledFunction(10.0 * vf * max(tau_up, tau_dn) / h)
    # Dirac delta at 0, note factor h^-1 for normalization
    #f4b.values[f4b.samples] = (alpha_up / ds_up - alpha_dn / ds_dn) / (E_p * alpha_tot * h)

    #print("f4")
    #f4.plot(20.0)

    # perform convolutions
    M_hot = convolve(f1, f2)

    #source_diffusion = convolve(f1, f4)
    #M_thermal = convolve(source_diffusion, f3)

    source_diffusion_a = convolve(f1, f4a)
    source_diffusion_b = f1.multiply((alpha_up / ds_dn - alpha_dn / ds_dn) / (E_p * alpha_tot))
    source_diffusion = add(source_diffusion_a, source_diffusion_b)

    M_thermal = convolve(source_diffusion, f3)

    # add hot and thermal
    M_total = add(M_hot, M_thermal)

    print("thermal system source term")

    plt.plot(source_diffusion_a.make_z_list(), source_diffusion_a.values, 'r', \
             source_diffusion_b.make_z_list(), source_diffusion_b.values, 'b')
    plt.show()

    print("spin currents")

    plt.plot(M_hot.make_z_list(), M_hot.values, 'r', \
             M_thermal.make_z_list(), M_thermal.values, 'g', \
             M_total.make_z_list(), M_total.values, 'b', \
             source_diffusion.make_z_list(), source_diffusion.values, 'y')
    plt.xlim(0.0, 100.0)
    plt.hlines(0.0, 0.0, 100.0)
    plt.show()
