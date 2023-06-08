from typing import List


class Solution:
    @staticmethod
    def numDistinctIslands(grid: List[List[int]]) -> int:

        def dfs_path(r, c, movement):
            grid[r][c] = 2
            island.append(movement)
            for x, y, z in direction_with_move:
                xR, yC = x + r, y + c
                if 0 <= xR < r_len and 0 <= yC < c_len and grid[xR][yC] == 1:
                    dfs_path(xR, yC, z)
            island.append(0)

        def dfs(r, c, start_r, start_c):
            grid[r][c] = 2
            island.append((r - start_r, c - start_c))
            for x, y in direction:
                xR, yC = x + r, y + c
                if 0 <= xR < r_len and 0 <= yC < c_len and grid[xR][yC] == 1:
                    dfs(xR, yC, start_r, start_c)

        if not grid:
            return 0
        direction_with_move = [(0, 1, 1), (1, 0, 2), (0, -1, 3), (-1, 0, 4)]
        r_len, c_len, islands, direction = len(grid), len(grid[0]), set(), [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for i in range(r_len):
            for j in range(c_len):
                island = []
                if grid[i][j] == 1:
                    # dfs(i, j, i, j)
                    dfs_path(i, j, 0)
                    islands.add(tuple(island))

        return len(islands)


if __name__ == '__main__':
    grid = [[1,1,0,1,1],[1,0,0,0,0],[0,0,0,0,1],[1,1,0,1,1]]
    print(Solution.numDistinctIslands(grid))

"""
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

An island is considered to be the same as another if and only if one island can be translated (and not rotated or reflected) to equal the other.

Return the number of distinct islands.

 

Example 1:


Input: grid = [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
Output: 1
Example 2:


Input: grid = [[1,1,0,1,1],[1,0,0,0,0],[0,0,0,0,1],[1,1,0,1,1]]
Output: 3
 

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 50
grid[i][j] is either 0 or 1.
"""