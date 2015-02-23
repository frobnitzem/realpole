#!/usr/bin/env python
# Tests moment representation error.

import sys
from integral import *
from poisson import *

Npts = 1000

rules = [ 2, 5, 8 ] #, 21, 27, 35, 53 ] #, 71, 89, 107, 125]#, 131 ]

SC = sqrt(3.0)/3.0 # 1 / box radius

def rand_pts(Npts, x, d=1):
    c = (rand.random((Npts,3))*2 - 1.0)*SC
    q = rand.random(Npts)*2 - 1.0

    if d == -1:
        rad2 = sum(c*c,1)
        q[where(rad2 < 1e-16)] = 0.0
        rad2[where(rad2 < 1e-16)] = 0.0001
        c /= rad2[:,newaxis] # invert in sphere

    # potential at the destination points
    pot = sum(q*sum((x[...,newaxis,:] \
                    - c)**2,-1)**(-0.5),-1)*0.25/pi
    return q,c, pot

def main(argv):
    assert len(argv) == 3, "Usage: %s <d> <out.dat>"%argv[0]

    dct = int(argv[1])
    rad = arange(100)*0.1+1.00001
    if dct == -1:
        rad = 1./rad
    L = map(lambda n: Poisson(n), rules)
    x = L[-1].L.x*rad[:,newaxis,newaxis]
    def mpot(r):
        return (0.25/pi)*sum((x-r)**2, -1)**(-0.5)

    print x.shape
    R = zeros((len(rad), 2*len(rules)+1))
    R[:,0] = rad
    S = 10
    for i in xrange(S):
        q,c, pot = rand_pts(Npts, x, dct)
        print "Iteration %d:"%i,
        for j,u in enumerate(L):
            print "%d"%u.K,
            w = u.solve_moments(q,c,1.0,dct)
            R[:,2*j+1] += dot((sum([w[i]*mpot(u.L.x[i]) \
                    for i in range(len(w))], 0) - pot)**2, L[-1].L.wt)
            if dct == -1:
                R[:,2*j+2] += dot((u.ig_quad(w, 1.0, x) - pot)**2, L[-1].L.wt)
            else:
                R[:,2*j+2] += dot((u.g_quad(w, 1.0, x) - pot)**2, L[-1].L.wt)
        print

    R[:,1:] = sqrt(R[:,1:]/(float(S)*4*pi))
    write_matrix(argv[2], R)

if __name__=="__main__":
    main(sys.argv)


