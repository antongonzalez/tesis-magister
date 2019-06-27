#!/usr/bin/env python
################################################################################
# File: Christoffel.py
# Author: Sergei Ossokine
# This contains the implementation of the Christoffel symbols of the 2nd kind
# The Christoffel symbols are represented as Tensor objects.
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
import sympy as sp
from GRpy.Tensor import *
import numpy as np

class Christoffel(Tensor):
	'''The class to represent Christoffel Symbols of the second kind.
	Please note that while it inherits from Tensor, Christoffel symbols 
	are NOT tensors'''
	def __init__(self,metr):
		# The metric
		self.g_down = metr
		# Since we have a metric we do indeed have a coordinate system
		self.rep  = self.g_down.rep
		self.g_up = metr.inverse
		# Please note that this call will trigger a call to allocate in
		# the Tensor class, but the allocate will actually be the allocate
		# defined below
		super(Christoffel,self).__init__('\Gamma^{a}_{\ \ bc}',(1,2),(1,-1,-1),coords=metr.coords)
		
	def allocate(self,rank):
		Tensor.allocate(self,rank)
		# Now that we have allocated things, time to actually compute things
		for i in np.arange(self.dim):
			for k in np.arange(self.dim):
				for l in np.arange(self.dim):
					suma = 0
					for m in np.arange(self.dim):
						term1 = sp.diff(self.g_down[-m,-k],self.g_down.coords[l])
						term2 = sp.diff(self.g_down[-m,-l],self.g_down.coords[k])
						term3 = sp.diff(self.g_down[-k,-l],self.g_down.coords[m])
						suma += self.g_up[i,m] * (term1+term2-term3)
					self.components[i,-k,-l] = sp.Rational(1,2)*suma
		self.getNonZero()
