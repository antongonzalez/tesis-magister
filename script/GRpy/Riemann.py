#!/usr/bin/env python
################################################################################
# File: Riemann.py
# Author: Sergei Ossokine
# This contains the implementation of the following GR-related quantities:
# Riemann cruvature tensor, Ricci tensor, Ricci scalar, Einstein tensor
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
from GRpy.Tensor import Tensor,Metric
from numpy import arange


class Riemann(Tensor):
	'''This is a class to represent the completely covariant 
	Riemann curvature tensor in a basis. We first compute the tensor
	with the last indexed raised, then we lower it'''
	def __init__(self,Chris):
		self.Chris = Chris
		super(Riemann,self).__init__('R_{abcd}',(0,4),(-1,-1,-1,-1),
		coords = Chris.g_down.coords)
		self.g_down = Chris.g_down
		self.g_up = Chris.g_up
		R_tensor = Tensor('R^{a}_{bcd}',(1,3),(1,-1,-1,-1),
		coords=Chris.g_down.coords)
		for a in arange(self.dim):
			for b in arange(self.dim):
				for c in arange(self.dim):
					for d in arange(self.dim):
						suma = 0
						for al in range(self.dim):
							term = (Chris[a,-al,-c]*Chris[al,-b,-d]
							-Chris[a,-al,-d]*Chris[al,-b,-c])
							suma += term
						t1 = sp.diff(Chris[a,-b,-d],self.coords[c])
						t2 = sp.diff(Chris[a,-b,-c],self.coords[d])
						res = t1 - t2 + suma
						R_tensor.components[a,-b,-c,-d] = res
		self.R_tensor = R_tensor
		Riemann_cont = Tensor('R^{abcd}',(4,0),(1,1,1,1),
		coords=Chris.g_down.coords)
		for a in range(self.dim):
			for b in range(self.dim):
				for c in range(self.dim):
					for d in range(self.dim):
						suma = 0
						for j in range(self.dim):
							for k in range(self.dim):
								for l in range(self.dim):
									suma += self.g_up[b,j]*self.g_up[c,k]*self.g_up[d,l]*R_tensor[a,-j,-k,-l]
						Riemann_cont.components[a,b,c,d] = suma
		self.Riemann_cont = Riemann_cont
		Riemann_cov = Tensor('R_{abcd}',(0,4),(-1,-1,-1,-1),
		coords=Chris.g_down.coords)		
		for a in arange(self.dim):
			for b in arange(self.dim):
				for c in arange(self.dim):
					for d in range(self.dim):
						suma = 0
						for f in arange(self.dim):
							suma +=self.g_down[-a,-f]*R_tensor[f,-b,-c,-d]
						self.components[-a,-b,-c,-d] = suma
		self.Riemann_cov = self
		Riemann_par = Tensor('R^{ab}_{\ \ \ cd}',(2,2),(-1,-1,-1,-1),
		coords=Chris.g_down.coords)
		for a in arange(self.dim):
			for b in arange(self.dim):
				for c in arange(self.dim):
					for d in range(self.dim):
						suma = 0
						for e in range(self.dim):
							for f in range(self.dim):
								suma += self.g_down[-c,-e]*self.g_down[-d,-f]*Riemann_cont[a,b,e,f]
						Riemann_par.components[a,b,-c,-d] = suma
		self.Riemann_par = Riemann_par
		self.getNonZero()

class Ricci(Tensor):
	''' This class represents the Ricci curvature tensor'''
	def __init__(self,Riem):
		super(Ricci,self).__init__('R_{ab}',(0,2),(-1,-1),coords =Riem.coords)
		self.g_up = Riem.g_up
		self.Riemann_cov = Riem.Riemann_cov
		for a in arange(self.dim):
			for b in arange(self.dim):
				suma = 0
				for l in range(self.dim):
					for k in range(self.dim):
#						suma += self.g_up[l,k]*self.Riemann_cov[-a,-l,-b,-k] # 04/05/2016
						suma += self.g_up[l,k]*self.Riemann_cov[-k,-a,-l,-b]
				self.components[-a,-b] = suma
		self.getNonZero()

class Scalar(object):
	""" Computing scalars of curvature"""
	def __init__(self,metric,Ricci, Riemann):
		self.g_up = metric.inverse
		self.Ricci = Ricci
		self.Riemann_cov = Riemann.Riemann_cov
		self.Riemann_cont = Riemann.Riemann_cont
		self.Riemann_par = Riemann.Riemann_par
		suma = 0
		for i in arange(metric.dim):
			for j in arange(metric.dim):
				suma += self.g_up[i,j]*self.Ricci[-i,-j]
		self.Scr_Ricci = suma
		suma = 0		
		for i in arange(metric.dim):
			for j in arange(metric.dim):
				for k in arange(metric.dim):
					for l in arange(metric.dim):
						suma += self.Riemann_cov[-i,-j,-k,-l]*self.Riemann_cont[i,j,k,l]
		self.Scr_Kretsch = suma
		suma = 0
		for a in arange(metric.dim):
			for b in arange(metric.dim):
				for c in arange(metric.dim):
					for d in arange(metric.dim):
						for e in arange(metric.dim):
							for f in arange(metric.dim):
								for g in arange(metric.dim):
									suma += self.Riemann_cov[-a,-b,-c,-d]*self.Riemann_cont[c,d,e,f]*self.Riemann_par[a,b,-e,-f]
		self.Scr_Other = suma
		
	def __str__(self):
		print(70*'=')
		print("The Ricci scalar is:")
		print(str(sp.cancel(self.Scr_Ricci)))
		print(70*'=')
		print("The Kretschmann scalar is:")
		print(str(sp.cancel(self.Scr_Kretsch)))
		print(70*'=')
		print("The Other scalar is:")
		print(str(sp.cancel(self.Scr_Other)))
		print(70*'=')

	def __repr__(self):
		return self

def Scurv(metric,Ricci):
	'''Computes the Curvature scalar R = g^{ab}R_{ab}'''
	suma = 0
	for i in arange(metric.dim):
		for j in arange(metric.dim):
			suma += metric.inverse[i,j]*Ricci[-i,-j]
	return suma

class Einstein(Tensor):
	'''Computes the Einstein Tensor'''
	def __init__(self,metric,Ricci):
		super(Einstein,self).__init__('G_{ab}',(0,2),(-1,-1), coords=metric.coords)
		for a in arange(self.dim):
			for b in arange(self.dim):
				self.components[-a,-b] = Ricci[-a,-b] - sp.Rational(1,2)*Scurv(metric,Ricci)*metric[-a,-b]
		self.getNonZero()
				
