import math
from collections import defaultdict


class Solution:

    def calcEquation(self, equations, values, queries):

        def dfs(start, end, visited):
            if start == end:
                return 1.0
            visited.add(start)
            for v in adjacency_list[start]:
                if v in visited:
                    continue
                res = operations[(start, v)] * dfs(v, end, visited)
                if res != max_f:
                    return res
            return max_f

        result, operations, adjacency_list, max_f = [], defaultdict(float), defaultdict(set), math.inf
        for (u, v), value in zip(equations, values):
            operations[(u, v)] = value
            operations[(v, u)] = 1 / value
            adjacency_list[u].add(v)
            adjacency_list[v].add(u)

        for start, end in queries:
            if start not in adjacency_list or end not in adjacency_list:  # query variable does not exist
                result.append(-1.0)
            elif start == end:
                result.append(1.0)
            else:
                res = dfs(start, end, set())
                result.append(res if res != max_f else -1.0)
        return result


if __name__ == '__main__':
    equations, values, queries = [["a", "b"], ["b", "c"], ["bc", "cd"]], [1.5, 2.5, 5.0], [["a", "c"], ["c", "b"],
                                                                                           ["bc", "cd"], ["cd", "bc"]]
    print(Solution().calcEquation(equations, values, queries))
