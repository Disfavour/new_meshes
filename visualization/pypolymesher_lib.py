import pyPolyMesher
from pyPolyMesher.exampleDomains import MichellDomain
MichellDomain.Plot()
Node, Element, Supp, Load, P = pyPolyMesher.PolyMesher(MichellDomain, 50, 100, anim=True)