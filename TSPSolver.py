#!/usr/bin/python3
import copy
import logging.config
import queue
from typing import List

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

# Set up logger
logging.config.fileConfig("./logging.conf")
logger = logging.getLogger(__name__)


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
		random.shuffle(cities)
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

	def branchAndBound(self, time_allowance=60.0):
		# Time and Space O(n!): At worst case it is as bad as brute force,
		#   however in practice it is much faster
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		pruned = 0
		total = 0
		max_size = 1
		start_time = time.time()
		# To get the initial BSSF I used a random tour
		# Time and space O(n!): worst case search all of the n! possibilities
		self._bssf = self.greedy()['soln']
		rcm = [[from_city.costTo(to_city) for to_city in cities] for from_city in cities]
		first_node = TSPNode(0, [0], rcm)
		first_node.calculateRCM()
		total += 1

		# The priority queue I used is queue.PriorityQueue(). The way I use it, I put
		#   PrioritizedItem's into it that have a priority(taken from the TSPNode) and
		#   an item which is not ordered and contains the actual TSPNode.
		#   queue.PriorityQueue() uses a heap, so operations are log(n)
		priority_queue = queue.PriorityQueue()
		# Time and Space O(log(n)): priority queue uses a heap, so operations are log(n)
		priority_queue.put(PrioritizedItem(first_node.priority, first_node))

		# Time and Space O(n!): at worst explore each of the n! possibilities
		while (not priority_queue.empty()) and time.time() - start_time < time_allowance:
			# Time and Space O(log(n)): priority queue uses a heap, so operations are log(n)
			current_node = priority_queue.get().item
			current_city_index = current_node.path[-1]
			if current_node.bound >= self._bssf.cost:
				continue
			# Time and Space O(n): At worst this loop expands to n more possibilities
			for j in range(ncities):
				from_current = current_node.rcm[current_city_index][j]
				if np.isinf(from_current):
					continue
				temp_path = copy.deepcopy(current_node.path)
				temp_path.append(j)
				temp_rcm = copy.deepcopy(current_node.rcm)
				temp_node = TSPNode(current_node.bound + from_current,
									temp_path,
									temp_rcm)
				temp_node.setInfinities(current_city_index, j)
				# Time O(n^2): See calculateRCM method...
				# Space O(1): See calculateRCM method...
				temp_node.calculateRCM()
				total += 1
				if temp_node.bound >= self._bssf.cost:
					# prune the node
					pruned += 1
					continue
				if len(temp_node.path) == ncities:
					# make node new best solution so far
					temp_route = []
					for i in temp_node.path:
						temp_route.append(cities[i])
					self._bssf = TSPSolution(temp_route)
					count += 1
				else:
					# add node to the queue
					# Time and Space O(log(n)): priority queue uses a heap, so operations are log(n)
					priority_queue.put(PrioritizedItem(temp_node.priority, temp_node))
					if priority_queue.qsize() > max_size:
						max_size = priority_queue.qsize()

		# Time and Space O(1): time's up prune the rest, just add size of queue
		pruned += priority_queue.qsize()

		end_time = time.time()
		results['cost'] = self._bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = self._bssf
		results['max'] = max_size
		results['total'] = total
		results['pruned'] = pruned
		return results

	def fancy(self, time_allowance: int=60.0):
		"""
		Genetic Algorithm solution to the Traveling Salesman Problem. This
		method creates a number of routes, chooses the best routes, creates
		children by crossing over successful routes and mutating them, and then
		selects the next generation so the process can repeat. The algorithm runs
		until the time runs out, and then returns the best route found.

		:param time_allowance:
		:return: Dictionary of values to for the GUI
		"""
		# Calix

		########################################################################
		# Parameters
		population_size = 21
		num_parents = 7
		# With num_parents = n, we will produce C(2, n) children per iteration
		# where C(k, n) are the number of combinations 'n choose k'.
		crossover = True
		mutation = True
		########################################################################

		results = {}
		count = 0 # Number of times the algorithm found an improved bssf
		total = 0 # Total number of solutions generated and considered

		start_time = time.time()

		logger.info("Initializing population with {} individuals.".format(population_size))
		population = self.create_initial_population(population_size)
		total += len(population)
		bssf = population[0] # Let the bssf be a random solution
		rounds_since_change = 0

		while time.time()-start_time < time_allowance:
			parents = self.select_parents(population, num_parents)

			# Have every pair of parents produce a child.
			children = []
			if crossover:
				for pairing in itertools.combinations(parents, 2):
					children.append(self.crossover(*pairing))
			else:
				children = parents.copy()

			if mutation:
				for i in range(len(children)):
					children[i] = self.mutation(children[i])

			total += len(children)

			population = self.select_next_generation(population, children)

			if bssf.cost > population[0].cost:
				rounds_since_change = 0
				bssf = population[0]
				count += 1
				logger.debug("Found improved route")
			else:
				rounds_since_change += 1
			if rounds_since_change >= 30:
				break

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = total
		results['pruned'] = None
		return results


	def create_initial_population(self, population_size: int) -> List[TSPSolution]:
		# Olya
		number_initial_greedy = math.ceil(population_size / 2)

		population = []

		i = 0
		for i in range(0, number_initial_greedy):
			population.append(self.greedy()['soln'])
		for i in range(i, population_size):
			population.append(self.defaultRandomTour()['soln'])

		return population

	def select_parents(self, population: List[TSPSolution], num_parents: int) -> List[TSPSolution]:
		"""
		Select the best parents from the population.
		:param population:
		:param num_parents:
		:return: Sublist of the chosen individuals to be parents
		"""
		# Calix
		population = sorted(population, key=lambda solution: solution.cost)

		return population[:num_parents]

	def crossover(self, solution_one: TSPSolution, solution_two: TSPSolution) -> TSPSolution:
		# Alex
		# Get size of sublist
		for child_attempt_number in range(45):
			route_size = len(solution_one.route)
			sublist_size = random.randint(1, min(5, route_size))

			# Get index of sublist for solution_one and solution_two
			index_one = random.randrange(0, route_size - sublist_size)
			index_two = random.randrange(0, route_size - sublist_size)

			# Get sublist from solution_one and solution_two
			sublist_one = solution_one.route[index_one:index_one + sublist_size]
			sublist_two = solution_two.route[index_two:index_two + sublist_size]

			child_one_route = copy.deepcopy(solution_one.route)
			child_two_route = copy.deepcopy(solution_two.route)

			# child_one_route = solution_one.route
			# child_two_route = solution_two.route

			for i in range(sublist_size):
				self.swap(
					child_one_route,
					solution_one.route.index(sublist_one[i]),
					solution_one.route.index(sublist_two[i]))
				self.swap(
					child_two_route,
					solution_two.route.index(sublist_one[i]),
					solution_two.route.index(sublist_two[i]))

			child_one = TSPSolution(child_one_route)
			child_two = TSPSolution(child_two_route)

			min_child = None
			if child_one.cost <= child_two.cost:
				min_child = child_one
			else:
				min_child = child_two

			if not np.isinf(min_child.cost):
				return min_child
		min_parent = None
		if solution_one.cost <= solution_two.cost:
			min_parent = solution_one
		else:
			min_parent = solution_two
		return min_parent

	# Helper function to swap values in given list at given indexes
	def swap(self, swap_list, index_one, index_two):
		temp = swap_list[index_one]
		swap_list[index_one] = swap_list[index_two]
		swap_list[index_two] = temp

	def mutation(self, solution: TSPSolution) -> TSPSolution:
		# Alex
		for mutation_attempt_number in range(45):
			route_size = len(solution.route)

			index_one = random.randrange(0, route_size)
			index_two = random.randrange(0, route_size)
			while index_two is index_one:
				index_two = random.randrange(0, route_size)

			mutation_route = copy.deepcopy(solution.route)

			self.swap(mutation_route, index_one, index_two)

			mutation = TSPSolution(mutation_route)
			if not np.isinf(mutation.cost):
				return mutation
		return solution

	def select_next_generation(self, population: List[TSPSolution], children: List[TSPSolution]) -> List[TSPSolution]:
		# Olya
		next_generation = sorted(population + children, key=lambda soln: soln.cost)

		return next_generation[:len(population)]
