from typing import List
from collections import deque


class Solution:
    @staticmethod
    def shortestPathBinaryMatrix(grid: List[List[int]]) -> int:
        if not grid or grid[0][0] or grid[-1][-1]:
            return -1

        queue, grid[0][0], rLen, cLen  = deque([(1,0,0)]), 1, len(grid), len(grid[0])
        direction = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

        while queue:
            step, r, c = queue.popleft()
            if r == rLen-1 and c == cLen-1:
                return step
            for x, y in direction:
                nR, nC = x + r, y + c
                if 0 <= nR < rLen and 0 <= nC < cLen and not grid[nR][nC]:
                    queue.append((step+1, nR, nC))
                    grid[nR][nC] = 1
        return -1


if __name__ == '__main__':
    grid = [[0, 0, 0], [1, 1, 0], [1, 1, 0]]
    print(Solution.shortestPathBinaryMatrix(grid))


"""
Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. If there is no clear path, return -1.

A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n - 1, n - 1)) such that:

All the visited cells of the path are 0.
All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).
The length of a clear path is the number of visited cells of this path.

 

Example 1:


Input: grid = [[0,1],[1,0]]
Output: 2
Example 2:


Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4
Example 3:

Input: grid = [[1,0,0],[1,1,0],[1,1,0]]
Output: -1
 

Constraints:

n == grid.length
n == grid[i].length
1 <= n <= 100
grid[i][j] is 0 or 1
"""