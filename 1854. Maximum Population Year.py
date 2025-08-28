from collections import defaultdict
from typing import List


class Solution:
    def maximumPopulation(self, logs: List[List[int]]) -> int:
        min_year, max_year = logs[0][0], logs[0][1]
        sweeper = defaultdict(int)
        for log in logs:
            min_year = min(min_year, log[0])
            max_year = max(max_year, log[1])
            sweeper[log[0]] += 1
            sweeper[log[1]] -= 1

        max_population, current_population = 0, 0
        for i in range(min_year, max_year + 1):
            if i in sweeper:
                current_population += sweeper[i]
            if current_population > max_population:
                max_population = current_population
                min_year = i
        return min_year