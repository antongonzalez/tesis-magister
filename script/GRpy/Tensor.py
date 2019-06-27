#!/usr/bin/env python
################################################################################
# File: Tensor.py
# Author: Sergei Ossokine
# This contains the basic definition fo the Tensor class which can be used to
# store arbitrary-dimensional and arbitrary-rank tensors. It also defines a
# Metric class to represent the rank(0,2) non-degenerate symmetric tensor.
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

# TODO: Either expand or phase out the formalTensor class. Implement the sym
# attribute

# The required imports
import sympy as sp
import numpy as np
import itertools
from copy import deepcopy

class formalTensor(object):
	def __init__(self,rank,symbol):
		self.symbol = sp.Symbol(symbol)
		self.rank = rank
    
class Tensor(object):
	'''A class to represent a tensor in a particular basis'''
	def __init__(self,symbol,rank,shape,sym=None,coords=None,**args):
		self.symbol = sp.Symbol(symbol) # The symbol to represent this Tensor
		self.coords = coords # The coordinate system we are using for the representation
		self.shape = shape # The shape of the tensor, for example (-1,1) for k_{a}^{b}
		self.rank = rank # Rank
		self.contr = rank[0]
		self.cov = rank[1]
# We need to know the dimensionality of our spacetime. For the moment
# we will deduce it from the coordinates provide, or if none are given, then
# the assumption will be made that it's stored in the optional arguments
		if coords is not None:
			self.dim = len(self.coords)
		else:
			self.dim = args['dim']
		
		if coords is not None:
			self.allocate(rank)
			self.rep = True
			self.symbolic = formalTensor(rank,symbol)
			
	def __setitem__(self,idx,val):
		self.components[idx] = val
		
	def __getitem__(self,idx):
		return self.components[idx]
	
	def allocate(self,rank):
		'''Allocate the dictionary(hash table) necessary to store the components
		Note that covariant indices are negative! (except for 0 of course)'''
		n = rank[0] + rank[1]
		indc = list(itertools.product(range(self.dim),repeat=n))
		mastr = []
		for i in range(len(indc)):
			temp = []
			for k in range(len(indc[i])):
				if self.shape[k] == -1:
					temp.append(-indc[i][k])
				else:
					temp.append(indc[i][k])
		mastr.append(tuple(temp))
		self.components = dict(zip(mastr,[0 for i in range(len(indc))]))
	
	def _dictkeycopy(self, hay):
		keys = hay.keys()
		return dict(zip(keys,[0]*len(keys)))
		
	def getNonZero(self):
		'''Returns only non-zero components of the tensor, if the coordinate system is provided'''
		if self.rep:
			nonzerok = []
			nonzerov = []
			for key in self.components.keys():
				if self.components[key] !=0:
					nonzerok.append(key)
					nonzerov.append(self.components[key])
			d = dict(zip(nonzerok,nonzerov))
			keys = d.keys()
			#keys.sort()
			self.nonzero = [(key,d[key]) for key in keys]
			return self.nonzero
		else:
			print("Attempted to get components that have not been initialized!")
	
	def __str__(self):
		'''Print a "nice" human - readable representation of the tensor'''
		self.getNonZero()  
# We will print only non-zero components unless all the components are zero
		ttl=""
		if self.nonzero:
			print(70*'=')
			print('The non-zero components of '+str(self.symbol)+' are:')
			for i in range(len(self.nonzero)):
				ttl = (str(self.nonzero[i][0]) + " : "
				+str(sp.cancel(self.nonzero[i][1])))
				print(ttl)
			print(70*'=')
		else:
			print('All the components of '+str(self.symbol)+' are 0!')
  
class Metric(Tensor):
	'''Represents a metric. Note that coordinates now MUST be provided'''
	def __init__(self,coords,rank=(0,2),sh=(-1,-1),symbol='g_{ab}'):
		self.coords = coords
		super(Metric,self).__init__(symbol,rank,sh,coords=coords)
		self.getNonZero()
		
	def metric(self):
		temp = sp.eye(self.dim)
		for key in self.components.keys():
#			print('key = ', key)
			id = tuple(np.abs(key))
#			print('id =',id)
			temp[id] = self.components[key]
#			print('compo = ',self.components[key])
#		print('temp = ', temp)
#		print('self = ', self)
#		for i in range(self.dim):
#			for j in range(self.dim):
#				temp[i,j] = self.components[-i,-j]
#				print('compo = ',self.components[-i,-j])
		self.metric = temp		
#	def determinant(self):
#		self.metric()
		self.determinant = sp.cancel(self.metric.berkowitz_det())
#		self.determinant = sp.cancel(self.det())
#		return self.determinant
		
	def invert(self):
		'''Find the inverse of the metric and store the result in a Metric object self.inverse'''
# Store the data in a matrix, invert it using sympy than switch back
		temp = sp.eye(self.dim)
		for key in self.components.keys():
			id = tuple(np.abs(key))
			temp[id] = self.components[key]
		inv = temp.inv()
		self.inv = inv
		inverse = self._dictkeycopy(self.components)
		for i in range(self.dim):
			for j in range(self.dim):
				inverse[i,j] = inv[i,j]
		self.inverse = Metric(self.coords,rank=(2,0),sh=(1,1),symbol='g_inv')
		self.inverse.components = inverse
		return self.inverse
 		
class XI(Tensor):
	def __init__(self,coords,rank=(1,0),shape=(1,),symbol='xi^{a}'):
		self.coords = coords
		super(XI,self).__init__(symbol,rank,shape,coords=coords)
		
class Killing_Equation(Tensor):
	def __init__(self,xi,metr):
		self.xi_up = xi
		self.g_down = metr
#		self.xi_up = Tensor(symbol='xi^{a}',rank=(1,0),shape=(1,),coords=self.g_down.coords)
#		for i in range(self.g_down.dim):
#			self.xi_up.components[i] = sp.Function()
		self.rep  = self.g_down.rep
		self.g_up = metr.inverse
		super(Killing_Equation,self).__init__('K_{ab}',rank=(0,2),shape=(-1,-1),coords=metr.coords)

	def allocate(self,rank):
		Tensor.allocate(self,rank)
		for i in range(self.dim):
			for j in range(self.dim):
				suma = 0.0
				for k in range(self.dim):
					term1 = self.g_down.components[-i,-k]*sp.diff(self.xi_up.components[k],self.g_down.coords[j])
					term2 = self.g_down.components[-k,-j]*sp.diff(self.xi_up.components[k],self.g_down.coords[i])
					term3 = self.xi_up[k]*sp.diff(self.g_down.components[-i,-j], self.g_down.coords[k])
					suma += term1+term2+term3
				self.components[-i,-j]= sp.simplify(suma)
				
class Covariant_Derivative(Tensor):
	def __init__(self,xi,Chris):
		self.xi = xi
		self.Chris = Chris
		self.g_down = Chris.g_down
		self.rep  = self.g_down.rep
		self.g_up = self.g_down.inverse
		super(Covariant_Derivative,self).__init__('nabla_{a}',rank=(1,1),shape=(-1,1),coords=Chris.g_down.coords)

	def allocate(self,rank):
		Tensor.allocate(self,rank)
		for i in range(self.dim):
			for j in range(self.dim):
				suma = sp.diff(self.xi.components[j],self.Chris.coords[i])
				for k in range(self.dim):
					suma += self.Chris.components[j,-i,-k]*self.xi.components[k]			
				self.components[-i,j]= sp.simplify(suma)
			
class Geodesic_Equation(Tensor):
	def __init__(self,Chris):
		self.Chris = Chris
		self.g_down = Chris.g_down
		self.rep  = self.g_down.rep
		self.g_up = self.g_down.inverse
		x_up = Tensor(symbol='x^{a}',rank=(1,0),shape=(1,),coords=self.g_down.coords)
		self.l = sp.Symbol('l')
		for i in range(self.Chris.dim):
			x_up.components[i] = sp.Function(str(Chris.coords[i]))(self.l)
		self.x_up = x_up
		super(Geodesic_Equation,self).__init__('(Geoeq)^{a}',rank=(1,0),shape=(1,),coords=self.g_down.coords)
		ite = list(range(self.dim))
		for i in range(self.dim):
			ite[i] = (self.coords[i],self.x_up.components[i])
		x_down = Tensor(symbol='x_{a}',rank=(0,1),shape=(-1,),coords=self.g_down.coords)
		for i in range(self.dim):
			suma = 0
			for k in range(self.dim):
				suma += self.g_down[-i,-k]*self.x_up.components[k]
			x_down.components[-i] = suma.subs(ite,simultaneous=True)
		self.x_down = x_down

	def allocate(self,rank):
		Tensor.allocate(self,rank)
		for i in range(self.dim):
			suma = sp.diff(sp.diff(self.x_up.components[i],self.l),self.l)
			for j in range(self.dim):
				for k in range(self.dim):
					suma += self.Chris.components[i,-j,-k]*sp.diff(self.x_up.components[j],self.l)*sp.diff(self.x_up.components[k],self.l)		
			self.components[i]= sp.simplify(suma)
	
	def check(self,x):
		check = Tensor('Check_Geo_eq^{a}',rank=(1,0),shape=(1,),coords=self.g_down.coords)
		check.allocate(self.rank)
		self.x_up = x
		for i in range(self.dim):
			suma = sp.diff(sp.diff(self.x_up.components[i],self.l),self.l)
			for j in range(self.dim):
				for k in range(self.dim):
					suma += self.Chris.components[i,-j,-k]*sp.diff(self.x_up.components[j],self.l)*sp.diff(self.x_up.components[k],self.l)		
			check.components[i]= sp.simplify(suma)
		return check

class Desvio_Equation(Tensor):
	def __init__(self,x,Riemann):
		self.x_up = x
		self.Riemann = Riemann
		self.Chris = Riemann.Chris
		self.g_down = self.Chris.g_down
		self.rep  = self.g_down.rep
		self.g_up = self.g_down.inverse
#
		xi_up = Tensor(symbol='xi^{a}',rank=(1,0),shape=(1,),coords=self.g_down.coords)
		self.l = sp.Symbol('l')
		for i in range(self.g_down.dim):
			xi_up.components[i] = sp.Function('xi%d'%i)(self.l)
		self.xi_up = xi_up
#		
		xi_down = Tensor(symbol='xi_{a}',rank=(0,1),shape=(-1,),coords=self.g_down.coords)
		for i in range(self.g_down.dim):
			suma = 0
			for k in range(self.g_down.dim):
				suma += self.g_down.components[-i,-k]*self.xi_up.components[k]
			xi_down.components[-i] = suma
		self.xi_down = xi_down
#		
		u_up = XI(coords=self.Chris.coords)
		for i in range(self.g_down.dim):
			u_up.components[i] = sp.diff(x.components[i], self.l)
		self.u_up = u_up
#
		u_down = Tensor(symbol='u_{a}',rank=(0,1),shape=(-1,),coords=self.g_down.coords)
		for i in range(self.g_down.dim):
			for k in range(self.g_down.dim):
				suma += self.g_down.components[-i,-k]*self.u_up.components[k]
			u_down.components[-i] = suma
		self.u_down = u_down
		ite = list(range(self.g_down.dim))
		for i in range(self.g_down.dim):
			ite[i] = (self.g_down.coords[i],self.x_up.components[i])
		self.ite = ite
		super(Desvio_Equation,self).__init__('Des_eq^{a}',rank=(1,0),shape=(1,),coords=self.g_down.coords)

	def allocate(self,rank):
		Tensor.allocate(self,rank)
		u_up = self.u_up
		Chris = self.Chris
		xi_up = self.xi_up
		coords = self.coords
		R_tensor = self.Riemann.R_tensor
		for i in range(self.dim):
			suma = sp.diff(sp.diff(xi_up.components[i],self.l),self.l)
			for j in range(self.dim):
				for k in range(self.dim):
					t1 = 2*Chris.components[i,-j,-k]*xi_up.components[j]*u_up.components[k] 
					for m in range(self.dim):
						t2 = sp.diff(Chris.components[i,-j,-k], coords[m])*xi_up.components[j]*u_up.components[k]*u_up.components[m] 
						t5 = -R_tensor.components[i,-j,-k,-m]*u_up.components[j]*xi_up.components[k]*u_up.components[m] 
						for n in range(self.dim):
							t3 = (Chris.components[i,-m,-n]*Chris.components[m,-j,-k]
							*xi_up.components[j]*u_up.components[k]*u_up.components[n])
							t4 = -(Chris.components[i,-j,-k]*Chris.components[k,-m,-n]
							*xi_up.components[j]*u_up.components[m]*u_up.components[n])
							suma += t1+t2+t3+t4-t5
			self.components[i] = sp.simplify(suma.subs(self.ite,simultaneous=True))

	def check(self,xi):
		check = Tensor('Check_Des_eq^{a}',rank=(1,0),shape=(1,),coords=self.g_down.coords)
		check.allocate(self.rank)
		xi_up = xi
		u_up = self.u_up
		Chris = self.Chris
		coords = self.coords
		R_tensor = self.Riemann.R_tensor
		for i in range(self.dim):
			suma = sp.diff(sp.diff(xi_up.components[i],self.l),self.l)
			for j in range(self.dim):
				for k in range(self.dim):
					t1 = 2*Chris.components[i,-j,-k]*xi_up.components[j]*u_up.components[k] 
					for m in range(self.dim):
						t2 = sp.diff(Chris.components[i,-j,-k], coords[m])*xi_up.components[j]*u_up.components[k]*u_up.components[m] 
						t5 = -R_tensor.components[i,-j,-k,-m]*u_up.components[j]*xi_up.components[k]*u_up.components[m] 
						for n in range(self.dim):
							t3 = (Chris.components[i,-m,-n]*Chris.components[m,-j,-k]
							*xi_up.components[j]*u_up.components[k]*u_up.components[n])
							t4 = -(Chris.components[i,-j,-k]*Chris.components[k,-m,-n]
							*xi_up.components[j]*u_up.components[m]*u_up.components[n])
							suma += t1+t2+t3+t4-t5
			check.components[i] = sp.simplify(suma.subs(self.ite,simultaneous=True))
		return check 
