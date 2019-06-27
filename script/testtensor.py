#!/usr/bin/env python
################################################################################
# File: testtensor.py
# Author: Sergei Ossokine
# This contains some examples of usage of the various GRPy classes.
# The example investigated is a simple FLRW solution.
# Last Modified: Auhust 28th, 2009
# This file is part of GRPy, a small GR-oriented package based on sympy.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see:
# http://www.gnu.org/licenses/gpl.html.
################################################################################


from GRpy.all import *

t = Symbol('t')

r = Symbol('r')
theta = Symbol('theta')
phi = Symbol('phi')
k = Symbol('k')
R = Function('R')(t)


g = Metric((t,r,theta,phi))
g[0,0] = -1.0
g[-1,-1] = R**2/(1-k*r**2)
g[-2,-2] = r**2 * R**2
g[-3,-3] = R**2*r**2*(sin(theta))**2

g.invert()

C = Christoffel(g)
R = Riemann(C)
Ric = Ricci(R)
Rs = RicciScalar(g,Ric)
print "The components of the Ricci tensor for a FLRW universe are: "
for i in range(len(Ric.nonzero)):
    print(str(Ric.nonzero[i][0])+":")
    print(cancel(Ric.nonzero[i][1]))
for i in range(len(C.nonzero)):
    print(str(C.nonzero[i][0])+":")
    print (C.nonzero[i][1])
G = EnMt(g,Ric,Rs)
print "The energy density is given by"
print cancel(ratsimp(G[0,0]))
