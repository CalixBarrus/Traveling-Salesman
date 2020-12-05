#!/usr/bin/python3
from typing import List

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import heapq


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()

		while not foundTour and time.time()-start_time < time_allowance and count < ncities:
			route = [cities[count]]
			for _ in range( ncities-1):
				current_city = route[-1]
				valid_cities = np.setdiff1d(cities, route, assume_unique=True)
				cheapest_city = valid_cities[np.argmin([current_city.costTo(city) for city in valid_cities])]
				route.append(cheapest_city)

			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		start_time = time.time()

		# Helper class that will be in the queue
		class State:
			"""
			State class represents partial solution to the TSP problem

			route: List of cities representing the route constructed so far
			reduced_cost_matrix: matrix of costs between graph vertices. Used to
			calculate and update the lower_bound
			lower_bound: Pessimistic lower bound on solutions to the TSP along
			this partial route.
			"""
			def	__init__(self, route: List, reduced_cost_matrix: np.ndarray, lower_bound: int):
				"""
				This has O(1) temporal complexity and O(n^2) spatial complexity.
				:param route: List of cities
				:param reduced_cost_matrix: Reduced cost matrix updated with
				respect to the cities currently in the route
				:param lower_bound: Lower bound for the TSP updated with respect
				to the given reduced_cost_matrix
				"""
				self.route = route
				self.reduced_cost_matrix = reduced_cost_matrix
				self.lower_bound = lower_bound
			# def __lt__(self, other):
			# 	return self.lower_bound < other.lower_bound

			def update_route(self, new_city: City) -> None:
				"""
				Add the city to the route and update the reduced_cost_matrix and
				lower_bound accordingly

				O(n^2) temporal complexity and O(1) spaitial complexity (given
				that that cost has already been paid in the constructor).

				:param new_city: City to be added to the route.
				"""
				city_index = cities.index(new_city)
				prev_index = cities.index(self.route[-1])

				# Account for the cost of traveling to the given city
				self.lower_bound += self.reduced_cost_matrix[prev_index, city_index]
				# Cancel the column from which we travel
				self.reduced_cost_matrix[prev_index, :] = np.infty
				# And the column to which we now travel
				self.reduced_cost_matrix[:, city_index] = np.infty

				self.route.append(new_city)

				# Make sure there is a 0 in each column and row. Subtract from
				# the entire column the least element to make it so, if necessary.
				n = len(self.reduced_cost_matrix)
				for i in range(n):
					min_index = np.argmin(self.reduced_cost_matrix[i, :])
					if self.reduced_cost_matrix[i, min_index] == np.infty:
						# If a row or column not on the route has only infinities, there
						# is no valid tour given the route so far.
						if i not in [city._index for city in self.route[:len(self.route)-1]]:
							self.lower_bound = np.infty
					elif self.reduced_cost_matrix[i, min_index] > 0:
						self.lower_bound += self.reduced_cost_matrix[i, min_index]
						self.reduced_cost_matrix[i, :] -= self.reduced_cost_matrix[i, min_index]

				for i in range(n):
					min_index = np.argmin(self.reduced_cost_matrix[:, i])
					if self.reduced_cost_matrix[min_index, i] == np.infty:
						if i not in [city._index for city in self.route[1:]]:
							self.lower_bound = np.infty
					elif self.reduced_cost_matrix[min_index, i] > 0:
						self.lower_bound += self.reduced_cost_matrix[min_index, i]
						self.reduced_cost_matrix[:, i] -= self.reduced_cost_matrix[min_index, i]


			def child_state(self, new_city):
				"""
				Spin of a state similar to the current problem state but with
				an additional city added to the route.
				O(n^2) temporal complexity and O(n^2) spatial complexity due to
				the creation of a whole new reduced_cost_matrix.
				:param new_city: City to be added to the route
				:return: Child problem state of the current State
				"""
				result = State(list(self.route), np.array(self.reduced_cost_matrix), self.lower_bound)
				result.update_route(new_city)
				return result

		cost_matrix, lower_bound = self._create_cost_matrix(cities, 0)  # O(n^2) temporally and spatially
		initial_problem = State([cities[0]], cost_matrix, lower_bound)  # O(n^2) temporally and spatially
		total = 0
		self.cost_queue = [ (initial_problem.lower_bound, total, initial_problem) ]
		deepest_state = None # Keep track of the best, deepest state seen so far
		total += 1

		# best solution so far
		self.bssf = self.greedy()['soln'] # Worst case O(n^2) temporally, avg. case O(n) temporally. O(n) spatially in any case.

		cost_or_depth = True
		max_queue_length = 0
		states_trimmed = 0
		while not len(self.cost_queue) == 0 and time.time()-start_time < time_allowance:
			# Alternate prioritizing lower bound or tree depth
			if 0 != np.random.randint(0, 10): # About 1 in 10 times look at the deepest state
				current = heapq.heappop(self.cost_queue)[2] # O(log(n)) temporally
				if deepest_state is not None and current == deepest_state[2]:
					deepest_state = None
			else:
				if deepest_state == None:
					current = heapq.heappop(self.cost_queue)[2] # O(log(n)) temporally
				else:
					current = deepest_state[2]
					self.cost_queue.remove(deepest_state) # O(n) temporally
					deepest_state = None
					heapq.heapify(self.cost_queue) # O(n) temporally

			if current.lower_bound > self.bssf.cost:
				states_trimmed += 1
				continue

			# Complete tours will skip this for loop
			# At worst, this loop (combined with the outer loop) will go through all n! different partial tour possibilites
			for city in np.setdiff1d(cities, current.route, assume_unique=True):  # Look at cities not in the route
				if current.route[-1].costTo(city) == np.infty: # No edge to city
					continue

				child = current.child_state(city) # O(n^2) temporally and spatially
				if child.lower_bound < self.bssf.cost:
					new_state = (child.lower_bound, total, child)
					heapq.heappush(self.cost_queue, new_state) # O(log n)

					# Remember the best, deepest state
					if deepest_state is None:
						deepest_state = new_state
					elif len(new_state[2].route) >= len(deepest_state[2].route) and new_state[0] > deepest_state[0]:
						deepest_state = new_state
					total += 1
				else:
					states_trimmed += 1

			# Complete tours
			if len(current.route) == ncities:
				deepest_state = None
				solution = TSPSolution(current.route)  # O(n)
				count += 1
				if solution.cost < self.bssf.cost:
					self.bssf = solution
					if len(self.cost_queue) > max_queue_length:
						max_queue_length = len(self.cost_queue)


		if len(self.cost_queue) > max_queue_length:
			max_queue_length = len(self.cost_queue)

		states_trimmed += len(self.cost_queue)

		end_time = time.time()
		results['cost'] = self.bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = self.bssf
		results['max'] = max_queue_length
		results['total'] = total
		results['pruned'] = states_trimmed
		return results

	def _create_cost_matrix(self, cities, starting_city_index) -> (np.ndarray, int):
		"""
		Temporal and spatial complexity O(n^2)
		"""
		n = len(cities)
		lower_bound = 0
		result = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				result[i, j] = cities[i].costTo(cities[j])

		for row in result:
			min_index = np.argmin(row)
			if row[min_index] != np.infty and row[min_index] > 0:
				lower_bound += row[min_index]
				row -= row[min_index]

		for col in np.rollaxis(result, 1):
			min_index = np.argmin(col)
			if col[min_index] != np.infty and col[min_index] > 0:
				lower_bound += col[min_index]
				col -= col[min_index]

		return result, lower_bound



''' <summary>
	This is the entry point for the algorithm you'll write for your group project.
	</summary>
	<returns>results dictionary for GUI that contains three ints: cost of best solution, 
	time spent to find best solution, total number of solutions found during search, the 
	best solution found.  You may use the other three field however you like.
	algorithm</returns> 
'''
	# def fancy(self, time_allowance=60.0):
	# 	pass
