#!/usr/bin/env python
##############################################################################################
#######Gravitational Waves!!!!
##############################################################################################
from Tensor import *
from sympy import *
from numpy import arange
#En 4 dimensiones
x0 = Symbol('x0')
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
eta_ij = Metric((x0,x,y,z))
eta_ij[-0,-0] = 1
eta_ij[-0,-1] = 0
eta_ij[-0,-2] = 0
eta_ij[-0,-3] = 0
eta_ij[-1,-0] = 0
eta_ij[-1,-1] = -1
eta_ij[-1,-2] = 0
eta_ij[-1,-3] = 0
eta_ij[-2,-0] = 0
eta_ij[-2,-1] = 0
eta_ij[-2,-2] = -1
eta_ij[-2,-3] = 0
eta_ij[-3,-0] = 0
eta_ij[-3,-1] = 0
eta_ij[-3,-2] = 0
eta_ij[-3,-3] = -1
eta_ij.metric()
etaij_down = eta_ij
eta_ij.invert()
etaij_up = eta_ij.inverse
etaij_up.metric()
#etaij_up = eta_ij
#
class Gw(Tensor):
	'''Represents a gravitational wave. Note that coordinates now MUST be provided'''
	def __init__(self,coords,rank=(0,2),sh=(-1,-1),symbol='h_{ab}'):
		self.coords = coords
		super(Gw,self).__init__(symbol,rank,sh,coords=coords)
		self.getNonZero()
		
	def metric(self):
		temp = sp.eye(self.dim)
		for key in self.components.keys():
#			print 'key = ', key
			id = tuple(np.abs(key))
#			print 'id =',id
			temp[id] = self.components[key]
#			print 'compo = ',self.components[key]
#		print 'temp = ', temp
#		print 'self = ', self
#		for i in range(self.dim):
#			for j in range(self.dim):
#				temp[i,j] = self.components[-i,-j]
#				print 'compo = ',self.components[-i,-j]
		self.metric = temp
		self.determinant = sp.cancel(self.metric.berkowitz_det())
		
#	def determinant(self):
#		self.metric()
#		self.determinant = sp.cancel(self.metric.berkowitz_det())
#		self.determinant = sp.cancel(self.det())
#		return self.determinant
		
#		temp = sp.eye(self.dim)
#		self.gw1_down = self
#		for key in self.components.keys():
#			print 'key = ', key
#			id = tuple(np.abs(key))
#			temp[id] = self.components[key]
#		print 'temp = ', temp
#		self.metric = temp
#		self.determinant = sp.cancel(self.metric.berkowitz_det())
#		self.determinant = sp.cancel(self.det())
	def gw1_down(self):
#		self.metric()
		gw1_down = Metric(self.coords,rank=(0,2),sh=(-1,-1),symbol='h_{ab}')
		for i in range(self.dim):
			for j in range(self.dim):
#				print '[%d,%d]'%(i,j),self.metric[i,j]
				gw1_down.components[-i,-j] = self.metric[i,j]
		self.gw1_down = gw1_down
		return self.gw1_down
	def gw1_up(self):
#		self.gw1_down()
		gw1_up = Metric(self.coords,rank=(2,0),sh=(1,1),symbol='h^{ab}')
		for i in range(self.dim):
			for j in range(self.dim):
#				print self.metric[-i,-j]
				for l in range(self.dim):
					for m in range(self.dim):
						t = etaij_up[i,l]*etaij_up[j,m]*self.metric[-l,-m]
				gw1_up.components[i,j] = t
		self.gw1_up = gw1_up
		return self.gw1_up
#	def invert(self):
#		'''Find the inverse of the metric and store the result in a Metric object self.inverse'''
# Store the data in a matrix, invert it using sympy than switch back
#		temp = sp.eye(self.dim)
#		for key in self.components.keys():
#			id = tuple(np.abs(key))
#			temp[id] = self.components[key]
#		inv = temp.inv()
#		inverse = self._dictkeycopy(self.components)
#		for i in range(self.dim):
#			for j in range(self.dim):
#				inverse[i,j] = inv[i,j]
#		self.inverse = Metric(self.coords,rank=(2,0),sh=(1,1),symbol='g_inv')
#		self.inverse.components = inverse

class Christoffel_gw1(Tensor):
	'''The class to represent Christoffel Symbols of the second kind.
	Please note that while it inherits from Tensor, Christoffel symbols 
	are NOT tensors'''
	def __init__(self,gw1):
		# The metric
		self.gw1_down = gw1.gw1_down
		self.gw1_up = gw1.gw1_up
		# Since we have a metric we do indeed have a coordinate system
		self.rep  = self.gw1_down.rep
		# Please note that this call will trigger a call to allocate in
		# the Tensor class, but the allocate will actually be the allocate
		# defined below
		super(Christoffel_gw1,self).__init__('Gamma^{a}_{bc}',(1,2),(1,-1,-1),coords=gw1.coords)
		
	def allocate(self,rank):
		Tensor.allocate(self,rank)
		# Now that we have allocated things, time to actually compute things
		for i in np.arange(self.dim):
			for k in np.arange(self.dim):
				for l in np.arange(self.dim):
					suma = 0
					for m in np.arange(self.dim):
						term1 = sp.diff(self.gw1_down[-m,-k],self.gw1_down.coords[l])
						term2 = sp.diff(self.gw1_down[-m,-l],self.gw1_down.coords[k])
						term3 = sp.diff(self.gw1_down[-k,-l],self.gw1_down.coords[m])
						suma += etaij_up[i,m] * (term1+term2-term3)
					self.components[i,-k,-l] = sp.Rational(1,2)*suma
		self.getNonZero()

class Riemann_gw1(Tensor):
	'''This is a class to represent the completely covariant 
	Riemann curvature tensor in a basis. We first compute the tensor
	with the last indexed raised, then we lower it'''
	def __init__(self,Chris):
		self.Chris = Chris
		super(Riemann_gw1,self).__init__('R_{abcd}',(0,4),(-1,-1,-1,-1),
		coords = Chris.gw1_down.coords)
		self.gw1_down = Chris.gw1_down
		self.gw1_up = Chris.gw1_up
		R_tensor = Tensor('R^{a}_{bcd}',(1,3),(1,-1,-1,-1),
		coords=Chris.gw1_down.coords)
		for a in arange(self.dim):
			for b in arange(self.dim):
				for c in arange(self.dim):
					for d in arange(self.dim):
#						suma = 0
#						for al in range(self.dim):
#							term = (Chris[a,-al,-c]*Chris[al,-b,-d]
#							-Chris[a,-al,-d]*Chris[al,-b,-c])
#							suma += term
						t1 = sp.diff(Chris[a,-b,-d],self.coords[c])
						t2 = sp.diff(Chris[a,-b,-c],self.coords[d])
						R_tensor.components[a,-b,-c,-d] = t1-t2
		self.R_tensor = R_tensor
		Riemann_cont = Tensor('R^{abcd}',(4,0),(1,1,1,1),
		coords=Chris.gw1_down.coords)
		for a in range(self.dim):
			for b in range(self.dim):
				for c in range(self.dim):
					for d in range(self.dim):
						suma = 0
						for j in range(self.dim):
							for k in range(self.dim):
								for l in range(self.dim):
									suma += etaij_up[b,j]*etaij_up[c,k]*etaij_up[d,l]*R_tensor[a,-j,-k,-l]
						Riemann_cont.components[a,b,c,d] = suma
		self.Riemann_cont = Riemann_cont
		Riemann_cov = Tensor('R_{abcd}',(0,4),(-1,-1,-1,-1),
		coords=Chris.gw1_down.coords)		
		for a in arange(self.dim):
			for b in arange(self.dim):
				for c in arange(self.dim):
					for d in range(self.dim):
						suma = 0
						for f in arange(self.dim):
							suma += etaij_down[-a,-f]*R_tensor[f,-b,-c,-d]
						self.components[-a,-b,-c,-d] = suma
		self.Riemann_cov = self
		Riemann_par = Tensor('R^{ab}_{cd}',(2,2),(-1,-1,-1,-1),
		coords=Chris.gw1_down.coords)
		for a in arange(self.dim):
			for b in arange(self.dim):
				for c in arange(self.dim):
					for d in range(self.dim):
						suma = 0
						for e in range(self.dim):
							for f in range(self.dim):
								suma += etaij_down[-c,-e]*etaij_down[-d,-f]*Riemann_cont[a,b,e,f]
						Riemann_par.components[a,b,-c,-d] = suma
		self.Riemann_par = Riemann_par
		self.getNonZero()
class Ricci_gw1(Tensor):
	''' This class represents the Ricci curvature tensor'''
	def __init__(self,Riem):
		super(Ricci_gw1,self).__init__('R_{ab}',(0,2),(-1,-1),coords =Riem.coords)
		self.gw1_down = Riem.gw1_down
		self.gw1_up = Riem.gw1_up
		self.Riemann_cov = Riem.Riemann_cov
		for a in arange(self.dim):
			for b in arange(self.dim):
				suma = 0
				for l in range(self.dim):
					for k in range(self.dim):
						suma += etaij_up[l,k]*self.Riemann_cov[-k,-a,-l,-b]
				self.components[-a,-b] = suma
		self.getNonZero()
		
class Scalar_gw1(object):
	""" Computing scalars of curvature"""
	def __init__(self,Gw1,Ricci_gw1, Riemann_gw1):
		self.gw1_down = Gw1.gw1_down
		self.gw1_up = Gw1.gw1_up
		self.Ricci = Ricci_gw1
		self.Riemann_cov = Riemann_gw1.Riemann_cov
		self.Riemann_cont = Riemann_gw1.Riemann_cont
		self.Riemann_par = Riemann_gw1.Riemann_par
		suma = 0
		for i in arange(Gw1.dim):
			for j in arange(Gw1.dim):
				suma += etaij_up[i,j]*self.Ricci[-i,-j]
		self.Scr_Ricci = suma
		suma = 0		
		for i in arange(Gw1.dim):
			for j in arange(Gw1.dim):
				for k in arange(Gw1.dim):
					for l in arange(Gw1.dim):
						suma += self.Riemann_cov[-i,-j,-k,-l]*self.Riemann_cont[i,j,k,l]
		self.Scr_Kretsch = suma
		suma = 0
		for a in arange(Gw1.dim):
			for b in arange(Gw1.dim):
				for c in arange(Gw1.dim):
					for d in arange(Gw1.dim):
						for e in arange(Gw1.dim):
							for f in arange(Gw1.dim):
								for g in arange(Gw1.dim):
									suma += self.Riemann_cov[-a,-b,-c,-d]*self.Riemann_cont[c,d,e,f]*self.Riemann_par[a,b,-e,-f]
		self.Scr_Other = suma
		
	def __str__(self):
		print 70*'='
		print "The Ricci scalar is:"
		print str(sp.cancel(self.Scr_Ricci))
		print 70*'='
		print "The Kretschmann scalar is:"
		print str(sp.cancel(self.Scr_Kretsch))
		print 70*'='
		print "The Other scalar is:"
		print str(sp.cancel(self.Scr_Other))
		print 70*'='

	def __repr__(self):
		return self
		
class Einstein_gw1(Tensor):
	'''Computes the Einstein Tensor'''
	def __init__(self,Gw1,Ric_gw1,RS_gw1):
		super(Einstein_gw1,self).__init__('G_{ab}',(0,2),(-1,-1),coords =Ric_gw1.coords)
#		g_down = metric
		for a in arange(self.dim):
			for b in arange(self.dim):
				self.components[-a,-b] = Ric_gw1[-a,-b] - 0.5*RS_gw1*etaij_down[-a,-b]
		self.getNonZero()
