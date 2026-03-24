# implicit scheme
# stokes - omega
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import sparse
from scipy import linalg
import pyamg as amg

mesh = Mesh("3.xml")
#print mesh
plot(mesh, interactive=True)

nCell = 0
for cell in cells(mesh):
    nCell = nCell + 1
print "Number of cells = ", nCell

nEdge = 0
for edge in edges(mesh):
    nEdge = nEdge + 1
print "Number of edges = ", nEdge

nVertex = 0
for vertex in vertices(mesh):
    nVertex = nVertex + 1
print "Number of vertices = ", nVertex

#center point for cell
xc = []
yc = []
for cell in cells(mesh):
    xv = []
    yv = []
    for v in vertices(cell):
        xv.append(v.point().x())
        yv.append(v.point().y())
	    # cell.
    x1 = xv[0]
    x2 = xv[1]
    x3 = xv[2]
    y1 = yv[0]
    y2 = yv[1]
    y3 = yv[2]
    d = 2*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
    xx = ((x1*x1+y1*y1)*(y2-y3)+(x2*x2+y2*y2)*(y3-y1)+(x3*x3+y3*y3)*(y1-y2))/d
    xc.append(xx)
    yy = ((x1*x1+y1*y1)*(x3-x2)+(x2*x2+y2*y2)*(x1-x3)+(x3*x3+y3*y3)*(x2-x1))/d
    yc.append(yy)
    pnt = Point(xx,yy)
    if cell.contains(pnt) == False:
        sys.exit("Bag cell of grid!!!")

#distance ortogonal for edge
ortLenth = []
for edge in edges(mesh):
    dd = []
    for cell in cells(mesh):
        for edge1 in edges(cell):
            if edge1.index() == edge.index():
                dd.append(cell.index())
    if len(dd) == 2:
        ortLenth.append(np.sqrt((xc[dd[1]]-xc[dd[0]])**2+ (yc[dd[1]]-yc[dd[0]])**2))
    elif len(dd) == 1:
        ortLenth.append(np.sqrt((edge.midpoint().x()-xc[dd[0]])**2+ (edge.midpoint().y()-yc[dd[0]])**2))
    else:
        sys.exit("Bag edge of grid!!!")

#voronoi volume
vol = np.zeros(nVertex,'float')
f = np.zeros(nVertex,'float')
diag = np.zeros(nVertex,'float')

#matrix
for cell in cells(mesh):
    for edge in edges(cell):
        for v in vertices(edge):
            dn = np.sqrt((edge.midpoint().x()-xc[cell.index()])**2+ (edge.midpoint().y()-yc[cell.index()])**2)
            vol[v.index()] = vol[v.index()] + 0.25*edge.length()*dn
            diag[v.index()] = diag[v.index()] + dn/edge.length()

#boundary
bmesh = BoundaryMesh(mesh, "exterior")
print bmesh

#map nodes
imap = np.zeros(nVertex,'int')
mapping = bmesh.entity_map(0)
#boundary nodes
for vi in vertices(bmesh):
    i = mapping[vi.index()]
    imap[i] = 2
#close boundary nodes
for vi in vertices(mesh):
    #non boundary node
    i = vi.index()
    if imap[i] == 2:
        continue
    bFound = False
    for edge in edges(mesh):
        bEdge = False
        for vj in vertices(edge):
            if vj.index() == i:
                bEdge = True
        if bEdge == True:
            for vj in vertices(edge):
                if imap[vj.index()] == 2:
                    bFound = True
    if bFound == True:
        imap[i] = 1

dD = 100000.0
A = []
I = []
J = []
for vi in vertices(mesh):
    #diagonal
    i = vi.index()
    sd = diag[i]
    if imap[i] == 2:
        sd = sd + dD
    A.append(sd)
    I.append(i)
    J.append(i)
    #nondiagonal
    for edge in edges(mesh):
        bFound = False
        for vj in vertices(edge):
            if vj.index() == i:
                bFound = True
        if bFound == True:
            for vj in vertices(edge):
                if vj.index() != i:
                    j = vj.index()
                    A.append(-ortLenth[edge.index()]/edge.length())
                    I.append(i)
                    J.append(j)

A = sparse.coo_matrix((A, (I,J)))

R = []
I = []
J = []
for vi in vertices(mesh):
    #non boundary node
    i = vi.index()
    if imap[i] != 1:
        R.append(0.0)
        I.append(i)
        J.append(i)
        continue
    dd = 0.0
    for edge in edges(mesh):
        bEdge = False
        for vk in vertices(edge):
            if vk.index() == i:
                bEdge = True
        if bEdge == True:
            for vk in vertices(edge):
                if imap[vk.index()] == 2:
                    dd = dd + (ortLenth[edge.index()]/edge.length())**2/vol[vk.index()]
    R.append(dd)
    I.append(i)
    J.append(i)
    for edge in edges(mesh):
        bEdge = False
        for vk in vertices(edge):
            if vk.index() == i:
                bEdge = True
        if bEdge == True:
            for vk in vertices(edge):
                if imap[vk.index()] == 1 and vk.index() != i:
                    for cell in cells(mesh):
                        for edge1 in edges(cell):
                            if edge1.index() == edge.index():
                                for vc in vertices(cell):
                                    if imap[vc.index()] == 2:
                                        j = vk.index()
                                        dd = 1.0
                                        for edge2 in edges(cell):
                                            if edge2.index() != edge.index():
                                                dd = dd*ortLenth[edge2.index()]/edge2.length()
                                        R.append(dd/vol[vc.index()])
                                        I.append(i)
                                        J.append(j)

R = sparse.coo_matrix((R, (I,J)))

T = 1.0
M = 200
tau = T / M

#boundary condition
fw = np.zeros(nVertex,'float')
for vi in vertices(mesh):
    #non boundary node
    i = vi.index()
    if imap[i] == 2 and (vi.point().y() > 0.999*np.pi or vi.point().y() < 0.001):
        dd = 0.0
        for edge in edges(mesh):
            bEdge = False
            for vk in vertices(edge):
                if vk.index() == i:
                    bEdge = True
            if bEdge == True:
                for vk in vertices(edge):
                    if imap[vk.index()] == 2 and vk.index() !=  i and (vi.point().y() > 0.999*np.pi or vi.point().y() < 0.001):
                        dd = dd + 0.5*np.sin(vi.point().x())*edge.length()/ vol[i]
        fw[i] = dd*dD
    if imap[i] == 2 and (vi.point().x() > 0.999*np.pi or vi.point().x() < 0.001):
        dd = 0.0
        for edge in edges(mesh):
            bEdge = False
            for vk in vertices(edge):
                if vk.index() == i:
                    bEdge = True
            if bEdge == True:
                for vk in vertices(edge):
                    if imap[vk.index()] == 2 and vk.index() !=  i and (vi.point().x() > 0.999*np.pi or vi.point().x() < 0.001):
                        dd = dd + 0.5*np.sin(vi.point().y())*edge.length()/ vol[i]
        fw[i] = dd*dD
D = []
D1 = []
I = []
J = []
for vi in vertices(mesh):
    i = vi.index()
    D.append(vol[i])
    D1.append(1.0/vol[i])
    I.append(i)
    J.append(i)
D = sparse.coo_matrix((D, (I,J)))
D1 = sparse.coo_matrix((D1, (I,J)))

#full
# initial and boundary value
w = np.zeros(nVertex, 'float')
wT = np.zeros(nVertex, 'float')
psi = np.zeros(nVertex, 'float')
psiT = np.zeros(nVertex, 'float')

psiN = np.zeros((M+1), 'float')
psiN[0] = 0.
wN = np.zeros((M+1), 'float')
wN[0] = 0.

# initial
for vi in vertices(mesh):
    i = vi.index()
    psi[i] = np.sin(vi.point().x())*np.sin(vi.point().y())

AA = A + tau*(A*D1*A + R)
# time
for m in range(1,M+1):
    t = m*tau
    pt = np.exp(-2*t)
    print t
    # psi
    b = A*psi + tau*fw*pt

    ml = amg.ruge_stuben_solver(AA)
    psi = ml.solve(b, tol=1.0e-8, cycle='W', accel='cg')
    print "residual norm is", linalg.norm(b - AA*psi)

    # exact solition
    for vi in vertices(mesh):
        i = vi.index()
        psiT[i] = np.sin(vi.point().x())*np.sin(vi.point().y())*pt
        wT[i] = 2.0*np.sin(vi.point().x())*np.sin(vi.point().y())*pt

    w = D1*A*psi
    for vi in vertices(mesh):
        i = vi.index()
        if imap[i] == 2:
            dd = 0.0
            for edge in edges(mesh):
                bFound = False
                for vj in vertices(edge):
                    if vj.index() == i:
                        bFound = True
                if bFound == True:
                    for vj in vertices(edge):
                        if vj.index() != i and imap[i] != 2:
                            j = vj.index()
                            dd = dd - ortLenth[edge.index()]/edge.length() * psi[j] / vol[i]
            w[i] = dd

    s1 = 0.0
    s2 = 0.0
    for vi in vertices(mesh):
        i = vi.index()
#        if imap[i] == 2:
#            continue
        s1 = s1 + (psi[i]-psiT[i])**2*vol[i]
        s2 = s2 + (w[i]-wT[i])**2*vol[i]
    psiN[m] = np.sqrt(s1)
    wN[m] = np.sqrt(s2)
    print "error = ", s1, s2

# result
plt.figure(2)
tt = np.linspace(0., T, M+1)
plt.xlabel("$t$")
plt.ylabel("$\\varepsilon_1$")
plt.plot(tt, psiN)
plt.figure(3)
plt.xlabel("$t$")
plt.ylabel("$\\varepsilon_2$")
plt.plot(tt, wN)

f1=open("t.npy","wb")
f2=open("p.npy","wb")
f3=open("w.npy","wb")
np.save(f1,tt)
np.save(f2,psiN)
np.save(f3,wN)

plt.show()

V = FunctionSpace(mesh, "Lagrange", 1)
u = Function(V)
u.vector()[:] = psi[V.dofmap().vertex_to_dof_map(mesh)]
plot(u, interactive=True)









