from gmres import *
from poisson import *

cell = 1

# Distance hierarchy-aware oct-tree suitable for calculating FMM integrals.
class otree:
    def __init__(self, x, levels=4, Nmin=20, box=None):
        global cell
        if box == None: # new tree
            box = array([x.min(0), x.max(0)])
        self.R = sqrt(sum((box[1]-box[0])**2))/sqrt(3.0)
        #print "box = " + str(box) + " R = %f"%self.R
        self.cell = cell
        cell += 1

        self.levels = levels # Max number of levels below self.
        self.ctr = 0.5*sum(box, 0) # center
        if self.levels == 0 or len(x) < Nmin:
            self.x = x - self.ctr
            self.child = None
            #print " c%d [label=\"%d\"];"%(self.cell, len(self.x))
            return
        self.x = x - self.ctr
        #self.x = None # may be required by nearby low density region!

        # split in 8 pieces
        self.child = []
        for i in range(2):
          if i == 0:
            li = array(where(x[:,0] < self.ctr[0]), dtype=np.int)
          else:
            li = array(where(x[:,0] >= self.ctr[0]), dtype=np.int)
          for j in range(2):
            if j == 0:
              lj = li[where(x[li,1] < self.ctr[1])]
            else:
              lj = li[where(x[li,1] >= self.ctr[1])]
            for k in range(2):
              if k == 0:
                lk = lj[where(x[lj,2] < self.ctr[2])]
              else:
                lk = lj[where(x[lj,2] >= self.ctr[2])]
              cbox = box.copy()
              cbox[1-i,0] = self.ctr[0]
              cbox[1-j,1] = self.ctr[1]
              cbox[1-k,2] = self.ctr[2]
              o = otree(x[lk], levels-1, Nmin, cbox)
              #print "  c%d -> c%d;"%(self.cell, o.cell)
              # translation to child ctr, list of child indices, child object
              self.child.append((self.ctr - 0.5*sum(cbox,0), lk, o))

    # Set interaction list through a recursive call.
    # s is the set of all cells neighboring the parent, (including the parent)
    # in the form: [(dist, otree)], where dist is the integer distance (-1,0,1)^3
    def set_ilist(self, crd, s):
        adj = []
        self.interact = []

        num = lambda k: array([k/4, (k/2)%2, k%2])
        # Every child (or self) of s is classified as either adj. or interact.
        for d,p in s:
            if p.child != None:
                for k,c in enumerate(p.child):
                    dist = 2*d + num(k) - crd # simplified centroid sep.
                    if all(abs(dist) <= 1):
                        adj.append((dist, c[2]))
                    else:
                        self.interact.append((self.ctr - c[2].ctr, c[2]))
            elif p.x != None:
                dist = 4*d - 2*crd + np.ones(3, np.int) # centroid sep.
                if all(abs(dist) <= 3): # convex hull addition r=1 + r=2
                    adj.append((d, p)) # act as neighbor in 'd' direction
                else:
                    self.interact.append((self.ctr - p.ctr, p))
        #print "testing interact:"
        #for t,c in self.interact:
        #    if sqrt(sum(t**2)) < 1.5*self.R:
        #        print "I: %f (R = %f)"%(sqrt(sum(t**2)), self.R)

        # Recurse through child lists, which work with my adj list.
        if self.child != None:
            #print "     Node, %d interact, %d adj"%(len(self.interact), len(adj))
            for k,c in enumerate(self.child):
                    #print "  Child: " + str(num(k))
                    c[2].set_ilist(num(k), adj)
        else:
            # Fix up adj list to use actual translation vectors (to self)
            # and remove 'self' point.
            for i in range(len(adj)-1, -1, -1):
                if all(adj[i][0] == 0):
                    del adj[i]
                    continue
                adj[i] = (self.ctr - adj[i][1].ctr,) + adj[i][1:]
            #print "     Leaf, %d items, %d interact, %d adj"%(len(self.x), len(self.interact), len(adj))
            self.adj = adj
            #print "testing adj:"
            #for t,c in self.interact:
            #    if sqrt(sum(t**2)) < 0.5*sqrt(3.0)*self.R:
            #        print "A: %f (R = %f)"%(sqrt(sum(t**2)), self.R)

    # Do an upward sweep, collecting moments from all children.
    def gather(self, G, q):
        if self.child != None:
            self.M = 0.0
            for (t, l,c) in self.child:
                self.M += G.oshift(c.gather(G, q[l]), c.R, t, self.R)
                #err = abs(G.g_quad(c.M, c.R, t + self.R*G.L.x)
                #        - G.g_quad(G.oshift(c.M, c.R, t, self.R),
                #                    self.R, self.R*G.L.x)).max()
                #if err > 1e-3:
                #    print " Large M -> M error: %g"%err
                #    print "  self = %d R = %f"%(self.cell, self.R)
                #    print "  C = %d R = %f"%(c.cell, c.R)
                #    print "  t = %f %f %f (%f)"%(t[0], t[1], t[2], sqrt(sum(t*t)))
        else:
            self.M = G.solve_moments(q, self.x, self.R)
        self.q = q

        return self.M

    # Do a downward sweep, distributing the L-expansion
    # for the far field to all children.
    def distribute(self, G, L, u, ig_quad, calc):
        for (t,c) in self.interact:
            L += G.ishift(c.M, c.R, t, self.R) # M -> L[t]
            #print c.q.shape, c.x.shape
            #err = abs(G.calc(c.q, c.x, t + 0.5*self.R*G.L.x)
            #        - G.ig_quad(G.ishift(c.M, c.R, t, self.R),
            #                    self.R, 0.5*self.R*G.L.x)).max()
            #if err > 1e-6:
            #    print " Large M -> L error: %g"%err
            #    print "  self = %d R = %f"%(self.cell, self.R)
            #    print "  I = %d R = %f"%(c.cell, c.R)
            #    print "  t = %f %f %f (%f)"%(t[0], t[1], t[2], sqrt(sum(t*t)))
            #L += G.solve_moments(c.M, c.R*G.L.x-t, self.R, -1) # M -> L[t]
            #L += G.solve_moments(c.q, c.x-t, self.R, -1) # M -> L[t]

        if self.child != None:
            if isinstance(L, np.ndarray):
              for (t,l,c) in self.child:
                u[l] = c.distribute(G, G.ishift(L, self.R, -t, c.R), \
                                    u[l], ig_quad, calc) # L[t] -> L
                #err = abs(G.ig_quad(L, self.R, 0.5*G.L.x*c.R-t)
                #          - G.ig_quad(G.ishift(L, self.R, -t, c.R),
                #                      c.R, 0.5*G.L.x*c.R)).max()
                #if err > 1e-6:
                #    print " Large L -> L error: %g"%err
                #    print "  self = %d R = %f"%(self.cell, self.R)
                #    print "  C = %d R = %f"%(c.cell, c.R)
                #    print "  t = %f %f %f (%f)"%(t[0], t[1], t[2], sqrt(sum(t*t)))
            else:
              for (t,l,c) in self.child:
                u[l] = c.distribute(G, L, u[l], ig_quad, calc)
        else:
            if isinstance(L, np.ndarray):
                u += ig_quad(L, self.R, self.x)
            for (t,c) in self.adj:
                if len(c.q) > 0:
                  u += calc(c.q, c.x-t, self.x)
            for i in range(len(self.x)):
                if i > 0:
                    u[i:i+1] += calc(self.q[:i], self.x[:i], self.x[i:i+1])
                if i < len(self.x)-1:
                    u[i:i+1] += calc(self.q[i+1:], self.x[i+1:], self.x[i:i+1])
        return u

    # Top-level sweep routine
    # f : source strengths
    # u : dest. strengths
    # ig_quad : moments -> R -> dests (N x 3) -> N x ...
    # calc : q -> src (M x 3) -> dests (N x 3) -> N x ...
    def sweep(self, G, f, u, ig_quad, calc):
        self.gather(G, f)
        return self.distribute(G, 0.0, u, ig_quad, calc)

    # Top-level neighbor-list routine
    def nlist(self):
        num = lambda k: array([k/4, (k/2)%2, k%2])
        self.interact = []
        if self.child != None:
            z = np.zeros(3, dtype=np.int)
            for k,c in enumerate(self.child):
                #print "  Child: " + str(num(k))
                c[2].set_ilist(num(k), [(z, self)])
        else:
            self.adj = []

# FMM kernel and solution routines for G f = u using Gmres
# G : FMM-able kernel (contains calc, solve_moments, shift, ginv)
# dof : number of degrees of freedom per src/dst point (= 0 for 1 with no extra dim.)
# ig_quad and calc: moment-based and direct calculation of kernel
#   for Poisson, use G.ig_quad, G.calc or G.f_quad and G.calc2
# tree_params : passed (in grid()) as kwargs to otree(x, **tree_params)
# fix : func. to run on result of tree.sweep(G, f, u, ig_quad, calc)
class FMM:
    def __init__(self, G, ig_quad, calc, dof=0, fix=None, **tree_params):
        self.G = G
        self.dof = dof
        self.tree_params = tree_params
        self.tree = None
        self.shape = (0,0)
        self.N = 0
        self.ig_quad = ig_quad
        self.calc = calc
        if fix == None:
            self.fix = lambda x: x
        else:
            self.fix = fix

    # initialize a new particle position oct-tree
    def grid(self, x):
        self.tree = otree(x, **self.tree_params)
        self.tree.nlist()
        self.N = len(x)
        if self.dof == 0:
            n = self.N
        else:
            n = self.N*self.dof
        self.shape = (n, n)

    # Calculate sum_j G(x_i, y_j; f_j) using FMM
    # requires self.tree (made by Solver.grid())
    def __call__(self, f):
        if self.dof > 0:
            u = zeros((self.N, self.dof))
            self.tree.sweep(self.G, reshape(f, (self.N, self.dof)), \
                            u, self.ig_quad, self.calc)
        else:
            u = zeros(self.N)
            self.tree.sweep(self.G, f, \
                            u, self.ig_quad, self.calc)
        self.fix(u)
        return reshape(u, self.shape[0])

    # requires self.tree (made by Solver.grid())
    def solve(self, u, **kwargs):
        return gmres(self, u, **kwargs)

def test():
    N = 10
    rand = np.random
    G = Poisson(12)
    f = FMM(G, G.ig_quad, G.calc, levels=4)

    x = rand.random((N,N,N,3))
    x[...,0] += arange(N)[:,newaxis,newaxis]
    x[...,1] += arange(N)[newaxis,:,newaxis]
    x[...,2] += arange(N)[newaxis,newaxis,:]
    N = N**3
    x = reshape(x, (N,3))*10.0

    f.grid(x) # create oct-tree

    q = rand.random(N)-0.5
    # use FMM to compute potentials
    phi = f(q)
    phi2 = zeros(phi.shape)
    for i in range(N):
        if i > 0:
            phi2[i:i+1] += G.calc(q[:i], x[:i], x[i:i+1])
        if i < N-1:
            phi2[i:i+1] += G.calc(q[i+1:], x[i+1:], x[i:i+1])
    print sqrt(sum((phi - phi2)**2)/sum(phi2**2))
    exit(0)

    #d = sqrt(sum((x[:,newaxis,:]-x)**2, 2))
    #for i in range(N):
    #    d[i,i] = 10.0
    #
    #M = 0.25/pi/d
    #for i in range(N):
    #    M[i,i] = 0.0
    #print M.max()

    #print np.linalg.eigh(M)[0]
    #q3 = np.linalg.solve(M, phi)
    #print q3 - q

    #phi = rand.random(100) # 100 random potential points
    q2 = f.solve(phi, tol=1e-8, no_progress_factor=1.0) # potential that generates 'em

    print q
    print q2 # statistics of sol'n, etc.

