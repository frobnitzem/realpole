#!/usr/bin/env python

from ucgrad import *
from scipy.special import gamma
import os

cwd = os.path.dirname(os.path.abspath(__file__))

def read_rule(K):
    r = read_matrix("%s/rules/lebedev_%03d.txt"%(cwd,K))
    r[:,:2] *= pi/180.0 # th, phi

    N = len(r)
    x = zeros((N,3)) # cos(th) sin(phi), sin(th) sin(phi), cos(phi)
    x[:,0] = cos(r[:,0])
    x[:,1] = sin(r[:,0])
    x[:,:2] *= sin(r[:,1])[:,newaxis]
    x[:,2] = cos(r[:,1])
    w = r[:,-1]*4*pi
    return x, w

class Lebedev:
    def __init__(self, K):
        self.x, self.wt = read_rule(K)
        self.K = K
        self.N = len(self.x)

    # == quad(self.wt, f)
    def integrate(self, f):
        return reduce(lambda a,i: a + self.wt[i]*f(self.x[i]), \
                        range(self.N), 0.0)

    # Integrate over the distribution given by w
    def quad(self, w, f):
        assert len(w) == len(self.wt), "Wrong weight shape!"
        return reduce(lambda a,i: a + w[i]*f(self.x[i]), \
                        range(self.N), 0.0)

# Yield n derivatives of (-x.d)^n . |y|^a/n! in series.
# the result is a power series in x^n with expansion coeffs in QQ[y]
def e2poly(x, y, n, a):
    if n < 1:
        raise StopIteration
    r2 = sum(y*y, -1)
    r = zeros((3,) + x.shape[:-1]+y.shape[:-1], dtype=x.dtype)
    r[0] = pow(r2, 0.5*a)
    yield r[0].copy()

    x2 = sum(x*x, -1)
    for d in y.shape[:-1]:
        x2 = x2[...,newaxis]+zeros(d)
    x2 /= r2
    xy = tensordot(x, y, axes=[-1,-1])/r2
    r[1] = -a*r[0]*xy
    yield r[1].copy()
    
    for i in range(2, n):
        r[i%3] = (2*i-2-a)/float(i)*r[(i+2)%3]*xy + (a+2-i)/float(i)*r[(i+1)%3]*x2
        yield r[i%3].copy()

# Yield tensor expansions: (u.dx) (v.dx) (-x.dy)^n . |y|^a/n! in series,
def e2poly_T1(x, y, n, a):
    if n < 1:
        raise StopIteration
    r = zeros((3,) + x.shape[:-1]+y.shape[:-1], dtype=x.dtype)
    dr = zeros((3,) + x.shape[:-1]+y.shape[:-1] + (3,), dtype=x.dtype)

    r2 = sum(y*y, -1)
    r[0] = pow(r2, 0.5*a)
    r2 = 1./r2
    y = y*r2[...,newaxis]
    yield r[0].copy(), dr[0].copy() #, -y*r[0,...,newaxis]
    if n < 2:
        raise StopIteration

    x2 = sum(x*x, -1)
    xy = tensordot(x, y, axes=[-1,-1])
    for d in y.shape[:-1]:
        x = x[...,newaxis,:]+zeros(d)[:,newaxis]
        x2 = x2[...,newaxis]+zeros(d)
    x *= r2[...,newaxis]
    x2 *= r2

    r[1] = -a*r[0]*xy
    dr[1] = -a*r[0,...,newaxis]*y
    yield r[1].copy(), dr[1].copy() #, x*r[0,...,newaxis] - 3*y*r[1,...,newaxis]

    for i in range(2, n):
        c1 = (2*i-2-a)/float(i)
        c2 = (a+2-i)/float(i)
        r[i%3] = c1*r[(i+2)%3]*xy + c2*r[(i+1)%3]*x2
        dr[i%3] = c1*r[(i+2)%3,...,newaxis]*y + 2*c2*r[(i+1)%3,...,newaxis]*x \
                + c1*dr[(i+2)%3]*xy[...,newaxis] + c2*dr[(i+1)%3]*x2[...,newaxis]
        yield r[i%3].copy(), dr[i%3].copy() #, i*i*x*r[(i+2)%3,...,newaxis] \
                                 #- (2*i+1)*y*r[i%3,...,newaxis] - i*dr[(i+2)%3]*x2[...,newaxis]
                #2*i*i*x*r[(i+2)%3,...,newaxis] \

# Yield tensor expansions: (u.dx) (v.dx) (-x.dy)^n . |y|^a/n! in series,
#  -- numerically validated!
def e2poly_T2(x, y, n, a):
    if n < 1:
        raise StopIteration
    r = zeros((3,) + x.shape[:-1]+y.shape[:-1], dtype=x.dtype)
    dr = zeros((3,) + x.shape[:-1]+y.shape[:-1] + (3,), dtype=x.dtype)
    d2r = zeros((3,) + x.shape[:-1]+y.shape[:-1] + (3,3), dtype=x.dtype)

    r2 = sum(y*y, -1)
    r[0] = pow(r2, 0.5*a)
    yield r[0].copy(), dr[0].copy(), d2r[0].copy()
    if n < 2:
        raise StopIteration

    x2 = sum(x*x, -1)
    r2 = 1./r2
    y = y*r2[...,newaxis]
    xy = tensordot(x, y, axes=[-1,-1])
    for d in y.shape[:-1]:
        x = x[...,newaxis,:]+zeros(d)[:,newaxis]
        x2 = x2[...,newaxis]+zeros(d)
    x *= r2[...,newaxis]
    x2 *= r2

    r[1] = -a*r[0]*xy
    dr[1] = -a*r[0,...,newaxis]*y
    yield r[1].copy(), dr[1].copy(), d2r[1].copy()

    ltr = range(len(d2r[0].shape)) # transpose last axis
    ltr[-1] = len(ltr)-2
    ltr[-2] = len(ltr)-1

    for i in range(2, n):
        c1 = -(a-2*(i-1))/float(i)
        c2 = (a+2-i)/float(i)
        r[i%3] = c1*r[(i+2)%3]*xy + c2*r[(i+1)%3]*x2
        dr[i%3] = c1*r[(i+2)%3,...,newaxis]*y + 2*c2*r[(i+1)%3,...,newaxis]*x \
                + c1*dr[(i+2)%3]*xy[...,newaxis] + c2*dr[(i+1)%3]*x2[...,newaxis]
        Y =    c1*dr[(i+2)%3,...,:,newaxis]*y[...,newaxis,:] \
            + 2*c2*dr[(i+1)%3,...,:,newaxis]*x[...,newaxis,:] \
            + c2*(r[(i+1)%3]*r2)[...,newaxis,newaxis]*identity(3)
        d2r[i%3] = Y + transpose(Y, ltr) \
                + c1*d2r[(i+2)%3]*xy[...,newaxis,newaxis] \
                + c2*d2r[(i+1)%3]*x2[...,newaxis,newaxis]
        yield r[i%3].copy(), dr[i%3].copy(), d2r[i%3].copy()

##################### tests ####################

# Testing of Lebedev.integrate
def test_rule(L):
    def test_mon(quad, a, b, c):
        # Exact monomial integral
        def mon_int(e):
          if e[0] == 0 and e[1] == 0 and e[2] == 0:
            #integral = 2.0 * pi**1.5 / gamma(1.5)
            return 4*pi

          if e[0] % 2 == 1 or e[1] % 2 == 1 or e[2] % 2 == 1:
            return 0.0

          integral = 2.0
          for i in range(3):
              integral *= gamma(0.5 * (e[i] + 1))
          integral /= gamma(0.5 * (e[0] + e[1] + e[2] + 3))

          return integral

        def poly(z):
            return z[0]**a * z[1]**b * z[2]**c
        I1 = quad(poly)
        I2 = mon_int([a, b, c])
        #print "%f %f %e"%(I1, I2, abs(I1-I2))
        return (I1-I2)**2

    # iterate over all nonneg. sequences adding to `n'
    def iter3(n):
        a = n
        b = 0
        c = 0
        yield a, b, c
        while 1:
          if b == 0:
            if a == 0:
                break
            a -= 1
            b = c+1
            c = 0
            yield a, b, c
          b -= 1
          c += 1
          yield a, b, c

    quad = L.integrate
    #print "a b c   Quad  Exact  Err"
    err = 0.0
    S = 0
    for n in range(L.K+1):
        #print "--------------------"
        for a,b,c in iter3(n):
          #print "%d %d %d"%(a,b,c),
          err += test_mon(quad, a, b, c)
          S += 1
    return sqrt(err/float(S))

def test_legendre(M):
    from scipy.special import legendre
    #x = arange(12)/6.0 - 1.0 + 1./12.0
    #x2 = zeros((len(x), 2))
    #x2[:,0] = x
    #x2[:,1] = sqrt(1.0 - x*x)
    #y = array([1.0, 0.0])
    L = Lebedev(9)
    y = L.x[int(rand.random()*L.N)]
    #lp = array([legendre(n)(x) for n in range(M)])
    lp = array([legendre(n)(dot(L.x,y)) for n in range(M)])
    lp2 = zeros(lp.shape)
    #for n,u in enumerate(e2poly(x2, y, M, -1)):
    for n,u in enumerate(e2poly(L.x, y, M, -1)):
        lp2[n] = u

    return abs(lp-lp2).max()

# Numerical differentiation test of e2poly_T2
def test_e2poly_T2(M, a):
    h = 1e-5 # numerical differencing
    ih = 1./h
    h = 1./ih

    L = Lebedev(9)
    y = L.x[int(rand.random()*L.N)] #*(rand.random()*2.0+1.0)

    L2 = Lebedev(5)
    x2 = L2.x*h
    xp = zeros((len(x2),) + L.x.shape)
    xp = L.x[:,newaxis,:] + x2

    g = e2poly(xp, y, M+2, a)
    g.next(); g.next();
    g2 = e2poly_T2(L.x, y, M+2, a)
    g2.next(); g2.next();

    for n, p1, (p2, dp2, tp2) in zip(range(M), g, g2):
        #p2 = (0.25/pi)*dot(p1, L2.wt) # numerical average
        #dp2 = sum(L2.x*p1[...,newaxis]*L2.wt[newaxis,:,newaxis], 1)*3*0.25/pi*ih # numerical derivative
        d0 = p2[:,newaxis] - p1
        e0 = abs(d0).max()
        d0 += sum(x2*dp2[:,newaxis,:], -1)

        e1 = abs(d0).max()*ih
        d0 += 0.5*sum(x2*sum(x2[:,newaxis,:]*tp2[:,newaxis,:,:], -1), -1)
        e2 = abs(d0).max()*ih*ih
        print "Derivative %d Error = %e %e %e"%(n, e0,e1,e2)

# Numerical differentiation test of spoly_T2
def test_spoly_T2(M):
    h = 1e-5 # numerical differencing
    ih = 1./h
    h = 1./ih

    L = Lebedev(9)
    y = L.x[int(rand.random()*L.N)] #*(rand.random()*2.0+1.0)

    L2 = Lebedev(5)
    x2 = L2.x*h
    xp = zeros((len(x2),) + L.x.shape)
    xp = L.x[:,newaxis,:] + x2

    g = spoly(xp, y, M+2)
    g.next(); g.next();
    g2 = spoly_T2(L.x, y, M)
    g2.next(); g2.next();

    T = zeros((3,3,L.N))
    for n, p1, (p2, dp2, tp2) in zip(range(M), g, g2):
        #p2 = (0.25/pi)*dot(p1, L2.wt) # numerical average
        #dp2 = sum(L2.x*p1[...,newaxis]*L2.wt[newaxis,:,newaxis], 1)*3*0.25/pi*ih # numerical derivative
        d0 = p2[:,newaxis] - p1
        e0 = abs(d0).max()
        d0 += sum(x2*dp2[:,newaxis,:], -1)

        e1 = abs(d0).max()*ih
        d0 += 0.5*sum(x2*sum(x2[:,newaxis,:]*tp2[:,newaxis,:,:], -1), -1)
        e2 = abs(d0).max()*ih*ih
        print "Derivative %d Error = %e %e %e"%(n, e0,e1,e2)

def test():
    def integrals(L):
        err = test_rule(L)
        if err > 1e-8:
            print "  FAIL: monomial err = %g"%(err)
            return 1
        else:
            print "  monomial err = %g"%(err)
            return 0

    rules = [ 3, 9, 15, 21, 27, 35, 53, 71, 89, 107, 125, 131 ]
    #rules = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
    tests = [integrals]
    F = 0
    for k in rules:
      print "--- Rule %d ---"%k
      L = Lebedev(k)
      for t in tests:
        F += t(L)
      print

    print "%d/(%d*%d) failed tests."%(F,len(rules),len(tests))

def test_spoly(n):
    L = Lebedev(17)

    g = spoly(L.x, L.x, n)
    g2 = spoly2(L.x, L.x, n)
    for a, b in zip(g, g2):
        print abs(a-b).max()

def test_sfit(K):
    L = Lebedev(2*K+1)
    x = rand.random((100,3))-0.5
    q = rand.random((100,3))-0.5
    w = zeros(L.N)
    v = zeros((L.N,3))
 
    dests = L.x*4.0
    ir = sum((dests[:,newaxis,:]-x[newaxis,:,:])**2,2)**(-0.5)
    qr = tensordot(q, ( \
                  (dests[:,newaxis,:,newaxis]-x[newaxis,:,:,newaxis]) \
                * (dests[:,newaxis,newaxis,:]-x[newaxis,:,newaxis,:]) ) \
                * ir[...,newaxis,newaxis]**3, axes=[(0,1), (1,3)])

    pot = dot(ir, q) + qr
    pe = zeros(pot.shape)
 
    for n, P in enumerate(e2poly(x, L.x, K+1, -1)):
        v += (2*n+1)*tensordot(P, q, axes=[0,0])
        w += (2*n+1)*tensordot(P, sum(x*q,1), axes=[0,0])
    v *= L.wt[:,newaxis]*0.25/pi
    w *= L.wt*0.25/pi

    for n,(P,dP) in enumerate(e2poly_T1(L.x, dests, K+1, -1)):
        pe += tensordot(P, v, axes=[0,0]) + sum(tensordot(dP, v, axes=[0,0])*dests[:,newaxis,:], 2) \
                - tensordot(dP, w, axes=[0,0])
        print "%d: G error: %g / %g (%g)"%(n, sqrt(dot(sum((pe - pot)**2,1), L.wt)*0.25/pi), \
                                sqrt(dot(sum(pot**2,1), L.wt)*0.25/pi), abs(pe-pot).max())

    print pot[:10]
    print pe[:10]

#print "Legendre error:"
#print test_legendre(50)
#test_e2poly_T2(40, -1)
#test_e2poly_T2(40, 1)

#test_sfit(8)
#test_spoly_T2(40)

#test()
#test_spoly(200)
