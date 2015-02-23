# Poisson flow, multi-body interactions

from poisson import *
from gmres import *
from vis import *
import pylab as plt

d1 = array([1./12, -2/3.0, 0.0, 2/3.0, -1/12.0])
off = -2
# One-sided difference schemes:
#d1 = array([-25/12., 4., -3., 4/3., -1/4.])
#off1 = 0

# Directional derivative of f by numerical differencing.
def diff(f, x, r, h=1e-7):
    h = max(h, 1e-7)
    ih = 1./h
    h = 1./ih
    d = None
    for i in range(len(d1)):
        if d1[i] == 0.0:
            continue
        if d == None:
            d = d1[i]*f(x + (i+off1)*h*r)
        else:
            d += d1[i]*f(x + (i+off1)*h*r)

    return d*ih

# Represent the objects using spherical boundaries.
def solve_fluid(x, u, R, K):
    i4pi = 0.25/pi
    P = Poisson(K)
    L = P.L

    # weights for velocities
    u = dot(u, transpose(L.x))*L.wt*(R*R)[:,newaxis]
    # Solve transformed problem, where
    # s = R[:,newaxis] * L.wt * phi

    def flow(s, dst): # calculates phi(dst) via BEM
        z = zeros(dst.shape[:-1])
        for i in range(len(x)):
            z += P.f_quad(s[i], R[i], dst-x[i]) \
               + P.g_quad(u[i], R[i], dst-x[i])
        return z

    def vel(s, dst): # calculates phi, v(dst) via BEM
        print "Still returns bad data!"
        q = zeros(dst.shape[:-1])
        z = zeros(dst.shape)
        # vel() needs derivative wrt 2nd arg (dst), so flip
        y = dst - x[:,newaxis,:]
        imd = 1./sqrt(sum(y**2, -1))
        y *= imd[...,newaxis]
        for i in range(len(x)):
            for n,(p,dP) in enumerate(e2poly_T1(y[i], L.x, P.K, -1)):
                p *= ((R[i]*imd[i])**n * imd[i])[:,newaxis]
                dP *= ((R[i]*imd[i])**n * imd[i])[:,newaxis,newaxis]
                q += tensordot(n*s[i] + u[i],  p, axes=[0,1])
                z += tensordot(n*s[i] + u[i], dP, axes=[0,1])
        return i4pi*q, i4pi*z

    dst = x[:,newaxis,:] + L.x*R[:,newaxis,newaxis]
    def pflow(s): # calculates LHS for solution of BEM
        s = s.reshape((len(x), L.N))
        phi = s/(R[:,newaxis]*L.wt)
        for i in range(len(x)):
            phi -= P.f_quad(s[i], R[i], dst-x[i])
        return reshape(phi, -1)

    # Compute the matrix giving LHS for solution of BEM
    def pflow_mat():
        M = zeros((len(x), L.N, len(x), L.N))
        for i in range(len(x)):
            M[i,:,i] += identity(L.N)/R[i]
            #M[i,:,i] += diag(1./(L.wt*R[i]))
            for n,A in enumerate(e2poly(R[i]*P.L.x, dst-x[i], P.K, -1)):
                M[i] -= A*(n*0.25/pi)*L.wt[:,newaxis,newaxis]
                #M[i] -= A*(n*0.25/pi)
        return reshape(transpose(M, (2,3,0,1)), \
                                 (len(x)*L.N, len(x)*L.N))

    pf = FakeMatrix(pflow, (len(x)*L.N, len(x)*L.N))
    spot = zeros((len(x), L.N))
    #u = dot(u, transpose(L.x))*L.wt*(R*R)[:,newaxis]
    for i in range(len(x)): # compute RHS for solution
        spot += P.g_quad(u[i], R[i], dst-x[i])
    spot = spot.reshape(-1)

    #print "Solving sz: ", len(spot)
    # Figure out phi (s)
    #s = gmres(pf, spot, tol=1e-12, stall_iterations=100).solution.reshape((len(x),L.N))
    M = pflow_mat()
    #print len(M), la.cond(M)
    s = la.solve(M, spot).reshape((len(x), L.N))*L.wt
    #s = la.solve(M, spot).reshape((len(x), L.N))

    return lambda d: flow(s, d), lambda d: vel(s, d)

K = 8
x = array([[-1.0, 1.5, 0.0],[-1.0, -1.5, 0.0],[4.0, 0.0, 0.0]])
u = array([[1.0, 0.0, 0.0],[1.0, 0.0, 0.0],[-1.0, 0.0, 0.0]])
R = array([1.0, 1.0, 3.0])
#x = (arange(10)*3)[:,newaxis]*identity(3)[1]
#u = ones(10)[:,newaxis]*identity(3)[0]
#R = ones(len(x))

def calc_err_plot():
    # Error norm quantities
    LP = Lebedev(59)
    #LP = Lebedev(13)
    nvel = dot(u, transpose(LP.x)) # req'd b.c.
    print "BC Norm: ", dot(nvel**2, LP.wt)/sum(LP.wt)
    #ih = 1./1e-7
    #h = 1./ih
    npts = x[:,newaxis,:] + LP.x*R[:,newaxis,newaxis]
    #npts = x[:,newaxis,:] + LP.x*(R[:,newaxis,newaxis] - h)
    #npts2 = x[:,newaxis,:] + LP.x*(R[:,newaxis,newaxis] + h)
    def err(f, vel):
        #p, v = vel(npts)
        #f = lambda r: vel(r)[0]
        #v = -(f(npts2) - f(npts))*0.5*ih
        v = -diff(f, npts, LP.x)
        #v2 = sum(vel(npts)[1]*LP.x, -1) # actual b.c.
        #print v
        #print v2
        #print v
        #print nvel
        return sqrt( dot((v - nvel)**2, LP.wt)/(4*pi) )

    for i in range(3, 15):
        f, vel = solve_fluid(x, u, R, i)
        print i, err(f, vel)

def vis_flow():
    f, vel = solve_fluid(x, u, R, K)
    # slice flow field in the xy-plane
    def g(x1, y1):
        dx = x1[-1,0] - x1[-2,0]
        dy = y1[0,-1] - y1[0,-2]
        z = zeros((x1.shape[0]+1, y1.shape[1]+1, 3))
        z[:-1,:, 0] = x1[:,0,newaxis]
        z[-1, :, 0]  = x1[-1,0,newaxis] + dx
        z[:, :-1,1] = y1[0,:]
        z[:,  -1,1]  = y1[0,-1] + dy

        #def if_quad(w, R, x):
        #    for u,du in e2poly_T1(x, R*P.L.x, P.K, -1):
        #        w += tensordot(w, du, axes=[0,1])
        #    return 0.25/pi*w
        #def calc(q, x, y):
        #    iR3 = pow(sum(x-y[...,:,newaxis,:], -1)**2, -1.5)
        #    return tensordot(q, (x-y[...,:,newaxis,:])*iR3[...,newaxis], \
        #                     axes=[0,len(y.shape)-2])

        #f = FMM(G, if_quad, calc)

        #pot, v = vel(z)
        pot = f(z)
        v = zeros((x1.shape[0], y1.shape[1], 2))
        v[:,:,0] = (pot[1:,:-1]-pot[:-1,:-1])/dx
        v[:,:,1] = (pot[:-1,1:]-pot[:-1,:-1])/dy
        #for i in range(len(x)):
        #    m = where(sum((x[i]-z)**2,-1) < R[i]**2)
        #    print len(m[0])
        #    v[m[0],m[1],:] = u[i,:2]
        return v, pot[:-1,:-1]

    #plot_f(g, (-12.0, 12.0), (-2.5, 29.5), "sph_stack", 400)
    x0 = -5.0
    x1 = 10.0
    y0 = -7.5
    y1 = 7.5
    def draw_circ():
        fig = plt.gcf()
        for i in range(len(x)):
            circle = plt.Circle(( (x[i,0]-x0)/(x1-x0), \
                                  (x[i,1]-y0)/(y1-y0) ), \
                            R[i]/(x1-x0), fill=False,  transform=fig.transFigure)
            plt.gca().add_patch(circle)

    #plot_f(g, (x0, x1), (y0, y1), "sph_coll", dpi=300, code=draw_circ)
    plot_f(g, (x0, x1), (y0, y1), "sph_coll", dpi=100, code=draw_circ)
    #plot_f(g, (-5.0, 10.0), (-7.5, 7.5), "sph_stack", dpi=100, code=draw_circ)

vis_flow()
#calc_err_plot()

