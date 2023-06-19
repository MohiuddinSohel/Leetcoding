from functools import lru_cache
from typing import List


class Solution:
    def countPaths(self, grid: List[List[int]]) -> int:
        # because of stricly increasing path property, start of a path can not be at the end of path
        # so iterative standard dp is not possible, do dfs with memoization
        def dfs(r, c):
            if not dp[r][c]:
                dp[r][c] = 1  # cell itself is a stricly increasing path
                for x, y in direction:
                    new_r, new_c = r + x, c + y
                    if 0 <= new_r < len_r and 0 <= new_c < len_c and grid[r][c] < grid[new_r][new_c]:
                        dp[r][c] += (dfs(new_r, new_c) % mod)
                dp[r][c] %= mod

            return dp[r][c]

        len_r, len_c, direction, mod = len(grid), len(grid[0]), [(0, 1), (1, 0), (0, -1), (-1, 0)], 10 ** 9 + 7
        dp = [[0 for _ in range(len_c)] for _ in range(len_r)]

        # call can stuck at single cell, so we have to call dfs every cell
        return sum(dfs(i, j) for i in range(len_r) for j in range(len_c)) % mod


if __name__ == '__main__':
    cost = [[1,1],[3,4]]
    print(Solution().countPaths(cost))


"""
You are given an m x n integer matrix grid, where you can move from a cell to any adjacent cell in all 4 directions.

Return the number of strictly increasing paths in the grid such that you can start from any cell and end at any cell. Since the answer may be very large, return it modulo 109 + 7.

Two paths are considered different if they do not have exactly the same sequence of visited cells.
"""