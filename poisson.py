#!/usr/bin/env python

from ucgrad import *
from integral import *
from scipy.special import legendre

# Class representing moment expansions of the Green's
# function for the Poisson kernel, G(r) = 1/4 pi |r|
# using all K moments up to order K-1
class Poisson:
    def __init__(self, K):
        self.L = Lebedev(2*(K-1)+1) # add one since Lebedev is odd
        self.K = K
        self.BEM()

    # q, x are source charge and locations, y are destinations
    def calc(self, q, x, y):
        return tensordot(q, sum((x-y[...,newaxis,:])**2, 2)**-0.5, \
                axes=[0,-1])*0.25/pi

    def calc2(self, q, x, y): # sum_x -q_x dG(x-y)/dy = sum_x q_x dG(x-y)/dx
        return tensordot(q, (y[...,newaxis,:]-x)*(sum((x-y[...,newaxis,:])**2, 2)**-1.5)[...,newaxis], \
                axes=[0,-2])*0.25/pi

    # If d = 1 (forward),
    # return {w} such that sum q_a*L_n(x_a,y) = sum w_i*L_n(x_i,y)
    # for all K moments L_0, ..., L_{K-1}
    #
    # If d = -1 (reverse),
    # return {w} such that sum q_a*L_n(x,y_a) = sum w_i*L_n(x,y_i)
    # for all K moments L_0, ..., L_{K-1}
    #
    # -- scales as O(K * len(x) * N)
    def solve_moments(self, q,x,R,d=1):
        w = zeros(self.L.N)
        for n,P in enumerate(self.mom(q,x/R,d)):
            w += (2*n+1)*P
        return w*self.L.wt*(0.25/pi)

    # Shift the origin of the outer expansion
    # t = (x_(new origin) - x_(old origin))
    def oshift(self, w, R, t, R1=None): # *(0) -> O(t) shift
        assert len(w) == len(self.L.wt), "Wrong weight shape!"
        if R1 == None:
            R1 = R+sqrt(sum(t*t))
        return self.solve_moments(w, R*self.L.x - t, R1)

    # Shift the origin of the inner / outer expansion -> inner expansion
    # t = (x_(new origin) - x_(old origin))
    def ishift(self, w, R, t, R1=None): # *(0) -> I(t) shift
        assert len(w) == len(self.L.wt), "Wrong weight shape!"
        if R1 == None:
            R1 = abs(R-sqrt(sum(t*t)))
        return self.solve_moments(w, R*self.L.x - t, R1, -1)
        w2 = zeros(self.L.N)
        w3 = zeros(self.L.N)
        for n,P in enumerate(self.mom(w, self.L.x)):
            w3 += (2*n+1)*P * R**(-n-1)
        w3 *= self.L.wt*(0.25/pi)
        for n,P in enumerate(self.mom(w3, R*self.L.x - t, -1)):
            w2 += (2*n+1)*P
        return w2*self.L.wt*(0.25/pi)

    # Create an inner expansion by inverting G
    # t = (x_(new origin) - x_(old origin))
    # d == -1 : inner -> inner (exact)
    # d == 1  : outer -> inner (inexact)
    def ginv(self, w, R, t, d=-1):
        assert len(w) == len(self.L.wt), "Wrong weight shape!"
        if d == -1:
            g = self.L.wt*self.ig_quad(w, R, self.L.x+t)
        else:
            g = self.L.wt*self.g_quad(w, R, self.L.x+t)

        w2 = zeros(len(self.L.wt))
        for n,P in enumerate(e2poly(self.L.x, self.L.x, self.K, -1)):
            w2 += (2*n+1)**2*dot(P, g)
        return w2*self.L.wt*0.25/pi

    # Single-layer and double-layer potentials
    # Calculate matrices exactly integrating the singular
    # and hypersingular operators:
    # int { 1/4pi |x-y| f(y) dy }
    # and int { n(y).del_y (1/|x-y|) f(y) dy }
    # (where f(y) is a degree K/2 polynomial) over the unit sphere.
    def BEM(self):
        G = zeros((self.L.N, self.L.N))
        Fe = zeros((self.L.N, self.L.N))
        Fi = zeros((self.L.N, self.L.N))
        for n,P in enumerate(e2poly(self.L.x, self.L.x, self.K, -1)):
            G += P
            Fe += n*P # outside of sphere
            Fi -= (n+1)*P # inside of sphere
        self.G = G*(0.25/pi)
        self.Fe = Fe*(0.25/pi)
        self.Fi = Fi*(0.25/pi)

    # Integrate F(y, x) w(y) dy from the outer expansion
    def f_quad(self, w, R, x):
        assert len(w) == len(self.L.wt), "Wrong weight shape!"
        G = zeros(x.shape[:-1])
        for n,P in enumerate(e2poly(R*self.L.x, x, self.K, -1)):
            G += n*tensordot(w, P, axes=[0,0])
        return G*(0.25/pi)

    # Integrate F(y, x) w(y) dy from the inner expansion
    def if_quad(self, w, R, x):
        assert len(w) == len(self.L.wt), "Wrong weight shape!"
        G = zeros(x.shape[:-1])
        for n,P in enumerate(e2poly(x, R*self.L.x, self.K, -1)):
            G -= (n+1)*tensordot(w, P, axes=[0,1])
        return G*(0.25/pi)

    # Integrate G(y, x) w(y) dy from the outer expansion
    def g_quad(self, w, R, x):
        assert len(w) == len(self.L.wt), "Wrong weight shape!"
        G = zeros(x.shape[:-1])
        for P in e2poly(R*self.L.x, x, self.K, -1):
            G += tensordot(w, P, axes=[0,0])
        return G*(0.25/pi)

    # Integrate G(x, y) w(y) dy from the inner expansion
    def ig_quad(self, w, R, x):
        G = zeros(x.shape[:-1])
        for P in e2poly(x, R*self.L.x, self.K, -1):
            G += tensordot(P, w, axes=[-1,0])
        return G*(0.25/pi)

    # Calculate the polynomials in series:
    # sum_k q_k L_n(x_k, x_i) [d=1]
    # sum_k q_k L_n(x_i, x_k) [d=-1]
    def mom(self, q, x, d=1):
        if d == -1:
            for P in e2poly(self.L.x, x, self.K, -1):
                yield dot(P, q)
        else:
            for P in e2poly(x, self.L.x, self.K, -1):
                yield dot(q, P)

    # Approximation error in the potential represented by multipoles.
    # should be order (x-d)**-(K+1)
    def pot_err(self, q, x, R, dests):
        pot = sum(q*sum((dests[:,newaxis,:]-x[newaxis,:,:])**2,2)**(-0.5),1)
        w = self.solve_moments(q, x, R)

        def mpot(r):
            return sum((dests-r)**2, 1)**(-0.5)
        
        def mpot_l(r): # legendre expansion up to order k/2
            return reduce(lambda a,b: a+b, e2poly(R*r, dests, self.K, -1), 0.0)

        #pe = self.L.quad(w, mpot)
        #pe = self.L.quad(w, mpot_l)
        pe = self.g_quad(w, R, dests)*4*pi
        
        return sqrt(sum((pe - pot)**2)/sum(pot**2))

    # Test the reproducing kernel for representing moments.
    # should be exact.
    def rep_err(self, q,x, R):
        w = self.solve_moments(q,x,R)
        err = 0.0
        for P1,P2 in zip(self.mom(w, self.L.x), self.mom(q, x/R)):
            err += sum((P1-P2)**2)
        
        return sqrt(err/float(self.K))

def rand_sources(Npts=4000, R=1.0):
    Npts = 4000
    q = rand.random(Npts)-0.5
    x = R*(rand.random((Npts,3)) - 0.5)*2/sqrt(3.0)
    return q,x

def test(R):
    q, x = rand_sources(R=R)

    # check potential at sinks some distance away
    Nsink = 500
    dests = R*(rand.random((Nsink,3))*2 + array([1.5, 0.0, 0.0]))

    def rep_err(P):
        err = P.rep_err(q,x,R)
        if err > 1e-8:
            print "  FAIL: moment err = %g"%(err)
            return 1
        print "  matched %d moments, err = %g"%(P.K, err)
        return 0
    def pot_err(P):
        err = P.pot_err(q,x,R, dests)
        print "  moment integral error = %g"%(err)
        return 0
    def outer_shift(P):
        d = array([-1., 0., 0.])
        w = P.solve_moments(q, x + d*R*0.5, R*1.5)
        w2 = P.oshift(w, R*1.5, d*R*0.5, R)
        phi = P.calc(q, x, dests)
        err = sqrt(sum((P.g_quad(w2, R, dests)-phi)**2) / sum(phi**2))
        print "  shift err = %g"%(err)
        return 0
    def inner_shift(P):
        d = array([1., 0., 0.])
        q2 = rand.random(Nsink)-0.5
        w = P.solve_moments(q2, dests, R, -1)
        phi = P.calc(q2, dests, x)
        err = sqrt(sum((P.ig_quad(w, R, x)-phi)**2) / sum(phi**2))
        print "  inner exp err = %g"%(err)
        w = P.solve_moments(q2, dests + d*R*0.25, R*1.3, -1)
        w2 = P.ishift(w, R*1.3, d*R*0.25, R)
        #w2 = P.ginv(w, 0.05)
        err = sqrt(sum((P.ig_quad(w2, R, x)-phi)**2) / sum(phi**2))
        print "  ishift err = %g"%(err)
        return 0

    R2 = 10.0 #0.5*sqrt(3.0)*1.5
    qy = rand.random(100)-0.5
    x2 = R2*(rand.random((100, 3))-0.5)
    y2 = R2*(rand.random((100, 3))-0.5)
    def io_shift(P):
        t = R2*array([2.0, 0.0, 0.0])
        w2 = P.ishift(P.solve_moments(qy, y2, R2), R2, -t, R2)
        phi = P.calc(qy, y2+t, x2)
        err = sqrt(sum((P.ig_quad(w2, R2, x2)-phi)**2) / sum(phi**2))
        print "  M->L err = %g"%err
        return 0

    rules = [ 3, 4, 7, 8, 12, 13 ]
    tests = [rep_err, pot_err, outer_shift, inner_shift, io_shift]
    F = 0
    for k in rules:
      print "--- Poisson(%d) ---"%k
      L = Poisson(k)
      for t in tests:
        F += t(L)
      print

    print "%d/(%d*%d) failed tests."%(F,len(rules),len(tests))

# test(10.0)

