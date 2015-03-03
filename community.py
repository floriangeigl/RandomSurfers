__author__ = 'dhelic'

import numpy as np
import scipy.sparse.linalg as linalg
import pylab
import random
import math
import abc

def laplacian_matrix(A):
    return deg_matrix(A) - A

def spectrum(M, k=2, which="SR"):
    l, v = linalg.eigs(M, k=k, which=which)
    return l, v

def deg_vector(A):
    d = np.sum(A, axis=0)
    return d

def deg_matrix(A):
    d = deg_vector(A)
    D = np.diag(d)
    return D

def modularity_matrix(A):
    m = sum(sum(A)) * 0.5
    Dp = degree_product_matrix(A)
    Dp *= 0.5 / m
    B = A - Dp
    return B

def degree_product_matrix(A):
    d = deg_vector(A)
    return np.outer(d, d)

def kronecker_delta(s):
    y = []
    for v in s:
        y.append(s == v)
    return np.array(y)

def spectral_clustering(A):
    L = laplacian_matrix(A)
    l, v = spectrum(L)
    lamb = [a.real for a in l]
    v2 = v[:, 1]
    v2 = [a.real for a in v2]
    return lamb[1], v2

def modularity_clustering(A):
    B = modularity_matrix(A)
    l, v = spectrum(B, k=1, which="LR")
    b = [a.real for a in l]
    v1 = v[:, 0]
    v1 = [a.real for a in v1]
    return b[0], v1

def init_randomly(n, c=2):
    s = np.zeros(n)
    for i in range(1, c):
        count = 0
        while count < (n / c):
            index = random.randint(0, n - 1)
            while s[index] == i:
                index = random.randint(0, n - 1)
            s[index] = i
            count += 1
    return s

def eval_modularity(B, s, m):
    kdelta = kronecker_delta(s)
    mod_matrix = B * kdelta
    mod = sum(sum(mod_matrix)) * 0.5 / m
    return mod

def eval_likelihood(A, s):
    L = 0
    LP = 0

    labels = np.unique(s)
    block_count = len(labels)
    block_indices = []
    for label in labels:
        block_indices.append(np.where(s == label)[0])

    for i in range(block_count):
        for j in range(i, block_count):
            block = A[np.ix_(block_indices[i], block_indices[j])]
            m = sum(sum(block))
            x, y = block.shape
            n = x * y
            if i == j:
                m *= 0.5
                n = x * (x - 1) * 0.5
            if m > 0 and n > m:
                L += m * math.log(m) + (n - m) * math.log(n - m) - n * math.log(n)
                LP += m * (math.log(m) - math.log(n))

    return L, LP

def simmulated_annealing(A, energy, alpha=1.0, c=2):
    print "number of divisions"
    print c

    n, n = A.shape
    s = init_randomly(n, c)
    T0 = 0.02 * alpha
    T = T0

    max_move = n / 5

    E = energy.eval_energy(s)
    Emin = E

    nsteps = 10000
    q = 0.8

    for k in range(2, 20):
        count = 0
        for i in range(nsteps):
            move_count = random.randint(1, max_move)
            move_index = random.sample(range(n), move_count)
            move_destination = np.array([random.randint(0, c - 1) for j in range(n)])
            s[move_index] = (s[move_index] + move_destination[move_index]) % c
            Enew = energy.eval_energy(s)

            delta_energy = Enew - E
            if random.uniform(0.0, 1.0) < math.exp(-delta_energy / T):  # accept
                E = Enew
                count += 1
                if E < Emin:
                    Emin = E
                    sb = np.copy(s)
            else:  # reject
                s[move_index] = (s[move_index] - move_destination[move_index]) % c

        print "acceptance rate"
        print float(count) / nsteps
        print "E"
        print E

        if count == 0:
            break
        T = T0 / k ** q

    print "last E"
    print E
    print "best"
    print Emin
    return Emin, sb

def kernighan_lin_modularity_bisection(A):
    n, n = A.shape
    m = sum(sum(A)) * 0.5
    s = init_randomly(n)
    B = modularity_matrix(A)
    mod_max = eval_modularity(B, s, m)

    round = 0
    while True:
        moved = np.zeros(n)
        sc = {}
        modc = {}
        for j in range(n):
            mod = {}
            for i in range(n):
                x = np.copy(s)
                if moved[i] == 0:
                    x[i] = (x[i] + 1) % 2
                    mod[i] = eval_modularity(B, x, m)

            index = max(mod, key=mod.get)
            moved[index] = 1
            s[index] = (s[index] + 1) % 2
            sc[j] = np.copy(s)
            modc[j] = mod[index]

        index = max(modc, key=modc.get)
        print "round"
        print round
        print modc[index]
        print s
        round += 1
        if modc[index] <= mod_max:
            break
        s = sc[index]
        mod_max = modc[index]

    mod = eval_modularity(B, s, m)
    print "max modularity"
    print mod
    return s

def plot_clustering(A, l, v, type="spectral"):
    fig1 = pylab.figure(1)
    x = range(len(v))
    pylab.plot(x, v)
    if type == "spectral":
        pylab.title("v2: $\lambda_2$=%f"%l)
    elif type == "modularity":
        pylab.title("v1: $\lambda_1$=%f"%l)
    else:
        pylab.title("v1")
    pylab.ylabel("Vector component")
    fig1.show()

    fig2 = pylab.figure(2)
    pylab.plot(x, sorted(v))
    if type == "spectral":
        pylab.title("v2 sorted: $\lambda_2$=%f"%l)
    elif type == "modularity":
        pylab.title("v1: $\lambda_1$=%f"%l)
    else:
        pylab.title("v1")
    pylab.ylabel("Vector component")
    fig2.show()

    sorted_indices = sorted((e, i) for i, e in enumerate(v))
    indices = [i for (e, i) in sorted_indices]

    fig3 = pylab.figure(3)
    pylab.spy(A)
    pylab.title("Adjacency Matrix")
    fig3.show()

    A1 = A[:, indices][indices]

    fig4 = pylab.figure(4)
    pylab.spy(A1)
    pylab.title("Adjacency Matrix Sorted")
    fig4.show()

    pylab.show()

def blocks1():
    blocks = np.array([45, 55])
    blockp = np.matrix("0.5, 0.01; 0.01, 0.4")
    sbm = SBMGenerator(blocks, blockp)
    A = sbm.generate()
    return A

def blocks2():
    blocks = np.array([40, 30, 30])
    blockp = np.matrix("0.8, 0.01, 0.05; 0.01, 0.7, 0.05; 0.05, 0.05, 0.6")
    sbm = SBMGenerator(blocks, blockp)
    A = sbm.generate()
    return A

def blocks3():
    blocks = np.array([4, 3, 3])
    blockp = np.matrix("0.8, 0.01, 0.05; 0.01, 0.7, 0.05; 0.05, 0.05, 0.6")
    sbm = SBMGenerator(blocks, blockp)
    A = sbm.generate()
    return A

def toy_example():
    A = np.genfromtxt("/home/dhelic/work/courses/netsci_slides/examples/community/ipython/toy.txt", delimiter=" ")
    return A

def toy_example2(mod=True):
    A = np.array([[0., 1., 1., 0., 0., 0.],
         [1., 0., 1., 0., 0., 0.],
         [1., 1., 0., 1., 0., 0.],
         [0., 0., 1., 0., 1., 1.],
         [0., 0., 0., 1., 0., 1.],
         [0., 0., 0., 1., 1., 0.]])
    s1 = np.array([1, 1, 1, 2, 2, 2])
    s2 = np.array([1, 1, 1, 1, 2, 2])

    if mod:
        B = modularity_matrix(A)
        m = sum(sum(A)) * 0.5
        print "good modularity"
        print eval_modularity(B, s1, m)
        print "bad modularity"
        print eval_modularity(B, s2, m)
    else:
        print "good likelihood"
        L, LP = eval_likelihood(A, s1)
        print L, LP
        print math.exp(L), math.exp(LP)
        print "bad likelihood"
        L, LP = eval_likelihood(A, s2)
        print L, LP
        print math.exp(L), math.exp(LP)

def spectral_example(A, plt=True):
    lamb2, v2 = spectral_clustering(A)
    if plt:
        plot_clustering(A, lamb2, v2)
    else:
        print v2

def modularity_example(A, plt=True):
    b, v = modularity_clustering(A)
    if plt:
        plot_clustering(A, b, v, type="modularity")
    else:
        print b
        print v

def kl_example(A, plt=True):
    s = kernighan_lin_modularity_bisection(A)
    if plt:
        plot_clustering(A, 1, s, type="kl")
    else:
        print "s"
        print s

def sa_example(A, energy, alpha=1.0, plt=True, cc=5):
    energies = []
    divisions = []
    for i in range(2, cc):
        mod, s = simmulated_annealing(A, energy, alpha=alpha, c=i)
        energies.append(mod)
        divisions.append(s)
    index = np.array(energies).argmin()
    s = divisions[index]

    print index
    print energies[index]

    if plt:
        plot_clustering(A, 1, s, type="sa")
    else:
        print "s"
        print s

class Energy:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def eval_energy(self, s):
        return

class ModularityEnergy(Energy):

    def __init__(self, A):
        self.A = A
        self.B = modularity_matrix(A)
        self.m = sum(sum(A)) * 0.5

    def eval_energy(self, s):
        return -eval_modularity(self.B, s, self.m)

class LikelihoodEnergy(Energy):

    def __init__(self, A, poisson=False):
        self.A = A
        self.poisson = poisson

    def eval_energy(self, s):
        L, LP = eval_likelihood(self.A, s)

        if self.poisson:
            return -LP
        return -L

class SBMGenerator:

    def __init__(self, blocks, block_prob):
        self.blocks = blocks
        self.block_prob = block_prob
        self.indices = []

    def generate(self, rindices=True):
        n = sum(self.blocks)
        block_count = len(self.blocks)
        if rindices:
            self.indices = np.random.permutation(range(n))
        else:
            self.indices = range(n)
        block_indices = self.create_indices()
        A = np.zeros(shape = (n, n))

        for i in range(block_count):
            for j in range(block_count):
                block_random = np.random.rand(len(block_indices[i]), len(block_indices[j]))
                block_random[block_random > self.block_prob[i, j]] = -1
                block_random[block_random != -1] = 1
                block_random[block_random == -1] = 0
                A[np.ix_(block_indices[i], block_indices[j])] = block_random

        A = np.triu(A, 1)
        A = A + A.T
        return A

    def generate_degree_corrected(self, degree_sequence):
        n = sum(self.blocks)
        block_count = len(self.blocks)
        block_random = np.zeros((n, n))

        #normalize degree_sequence
        i = 0
        for block_size in self.blocks:
            degrees = degree_sequence[i:(i + block_size)]
            #print sum(degrees)
            degree_sequence[i:(i + block_size)] /= float(sum(degrees))
            i += block_size

        self.indices = range(n)
        block_indices = self.create_indices()

        #init blocks
        for i in range(block_count):
            for j in range(block_count):
                block_random[np.ix_(block_indices[i], block_indices[j])] = self.block_prob[i, j]

        #degree correction
        for i in range(n):
            for j in range(n):
                block_random[i][j] *= degree_sequence[i] * degree_sequence[j] * 10000

        #print block_random

        # draw a graph
        A = np.zeros((n, n))
        A = np.random.rand(n, n)
        A[A < block_random] = 1
        A[A != 1] = 0

        #print sum(sum(A))
        A = np.triu(A, 1)
        A = A + A.T
        return A

    def create_indices(self):
        block_indices = []
        i = 0
        for block_size in self.blocks:
                block_indices.append(self.indices[i:(i + block_size)])
                i += block_size

        return block_indices

def example1():
    A = toy_example()
    spectral_example(A, plt=False)

def example2():
    A = blocks1()
    spectral_example(A)

def example3():
    A = blocks2()
    spectral_example(A)

def example4():
    toy_example2()

def example5():
    A = blocks1()
    kl_example(A)

def example6():
    A = toy_example()
    modularity_example(A, plt=False)

def example7():
    A = blocks1()
    modularity_example(A)

def example8():
    A = blocks2()
    modularity_example(A)

def example9():
    toy_example2(mod=False)

def example10():
    A = blocks1()
    energy = ModularityEnergy(A)
    sa_example(A, energy)

def example11():
    A = blocks2()
    energy = ModularityEnergy(A)
    sa_example(A, energy)

def example12():
    A = blocks1()
    energy = LikelihoodEnergy(A)
    sa_example(A, energy, alpha=800, cc=3)

def example13():
    A = blocks2()
    energy = LikelihoodEnergy(A)
    sa_example(A, energy, alpha=800, cc=4)

def example14():
    A = blocks3()
    energy = LikelihoodEnergy(A)
    sa_example(A, energy, alpha=200, cc=11)

#example14()