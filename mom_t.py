#!/usr/bin/env python

import sys
from integral import *
from poisson import *

Npts = 1000

rules = [ 2, 5, 8 ] #, 21, 27, 35, 53 ] #, 71, 89, 107, 125]#, 131 ]

T = arange(5)/5.0 # translation along x-axis
SC = (sqrt(3-2*T*T) - T)/3.0 # scale factor for ea. translation -- 1 / box radius
#print 2*SC**2 + (SC+T)**2 # 1 as expected.

def rand_pts(Npts, x, d=1):
    c = rand.random((Npts,3))*2 - 1.0
    q = rand.random(Npts)*2 - 1.0

    if d == -1:
        rad2 = sum(c*c,1)
        q[where(rad2 < 1e-16)] = 0.0
        rad2[where(rad2 < 1e-16)] = 0.0001
        c /= rad2[:,newaxis] # invert in sphere

    # potential at the destination points
    if d == -1:
      pot = array([sum(q*sum((r*x[...,newaxis,:]+array([t,0,0]) \
                    - c)**2,-1)**(-0.5),-1)*0.25/pi \
                 for t,r in zip(T,SC)])
    else:
      pot = array([sum(q*sum((x[...,newaxis,:] \
                    - c*r - array([t,0,0]))**2,-1)**(-0.5),-1)*0.25/pi \
                 for t,r in zip(T,SC)])
    return q,c, pot

def main(argv):
    global T, SC
    assert len(argv) == 3, "Usage: %s <d> <out-pre>"%argv[0]

    dct = int(argv[1])
    rad = 2.0
    if dct == -1:
        rad = SC[0]/rad # radius of eval. sphere
        T *= rad # scale translations within sphere
        SC = rad - T # absolute scale of eval. pts
        rad = 1.0 # fix sinks, x, on unit sphere

    L = map(lambda n: Poisson(n), rules)
    x = Lebedev(53).x*rad
    l = [(xi[0],xi[1],xi[2]) for xi in x]
    l.sort() # sort by cosine with x-axis
    x = array(l)
    def mpot(s, r):
        return (0.25/pi)*sum((s(x)-r)**2, -1)**(-0.5)
    Id = lambda z: z

    R = zeros((len(T), len(x), 2*len(rules)+1))
    R[:,:,0] = x[:,0]/rad
    S = 100
    for i in xrange(S):
        q,c, pot = rand_pts(Npts, x, dct)
        print "Iteration %d:"%i,
        for j,u in enumerate(L):
          if dct == -1:
            R1 = 1./sqrt(3.0) # radius bounding outer charges
            w0 = u.solve_moments(q,c,R1,dct)
          print "%d"%u.K,
          for k,t,r in zip(range(len(T)),T,SC):
            if dct == -1:
                R2 = R1-t # radius of translated inner exp.
                w = u.ishift(w0, R1, array([t,0,0]), R2)
                #w = u.ginv(w0, array([t,0,0]))
                if k == 0:
                    print abs(w - w0).max(),
                #w = u.solve_moments(q,c-array([t,0,0]),dct) # close?
                R[k,:,2*j+1] += (sum([w[i]*mpot(lambda x: r*x, R2*u.L.x[i])\
                            for i in range(len(w))], 0) \
                                    - pot[k])**2
                R[k,:,2*j+2] += (u.ig_quad(w, R2, r*x) - pot[k])**2
            else:
                w = u.solve_moments(q,c*r,r,dct)
                w = u.oshift(w, r, array([-t,0,0]), 1.0)
                #w = u.solve_moments(q,c*r+array([t,0,0])) # identical!
                R[k,:,2*j+1] += (sum([w[i]*mpot(Id,u.L.x[i])\
                            for i in range(len(w))], 0) - pot[k])**2
                R[k,:,2*j+2] += (u.g_quad(w, 1.0, x) - pot[k])**2
        print

    R[:,:,1:] = sqrt(R[:,:,1:]/float(S))
    for k,t in enumerate(T):
        write_matrix("%s_%.2f.dat"%(argv[2],t), R[k])

if __name__=="__main__":
    main(sys.argv)


