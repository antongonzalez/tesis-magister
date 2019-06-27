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


from Tensor import *
from Christoffel import *
from sympy import *
from Riemann import *


R = Symbol('R')
theta = Symbol('theta')
phi = Symbol('phi')


g = Metric((theta,phi))
g[-0,-0] = 1
g[-1,-1] = R**2*(sin(theta))**2

print(g.components.keys())

g.invert()

C = Christoffel(g)
R = Riemann(C)
Ric = Ricci(R)
Rs = RicciScalar(g,Ric)
print "Christoffel: "
for i in range(len(C.nonzero)):
    print(str(C.nonzero[i][0])+":")
    print(cancel(C.nonzero[i][1]))
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
