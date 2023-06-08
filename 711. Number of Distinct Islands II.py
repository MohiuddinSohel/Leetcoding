from typing import List


class Solution:
    @staticmethod
    def numDistinctIslands2(grid: List[List[int]]) -> int:
        def dfs(r, c):
            grid[r][c] = 2
            island.append((r, c))
            for x, y in direction:
                xR, yC = x + r, y + c
                if 0 <= xR < r_len and 0 <= yC < c_len and grid[xR][yC] == 1:
                    dfs(xR, yC)

        def normalizeGridPoint():
            shape = [[] for _ in range(8)]  # 8 shape for one island

            for r, c in island:
                shape[0].append([r, c])  # same or 360 rotation
                shape[1].append([c, -r])  # 90 degree rotation
                shape[2].append([-r, -c])  # 180 degree rotation
                shape[3].append([-c, r])  # 270 degree rotation
                shape[6].append([c, r])  # reflection by line y = x
                shape[4].append([-r, c])  # reflection by y axis or reflection by line y = x and rotate 90 degree
                shape[7].append([-c, -r])  # reflection by line y = -x or reflection by line y = x and rotate 180 degree
                shape[5].append([r, -c])  # reflection by x axis or reflection by line y = x and rotate 270 degree

            for i in range(len(shape)):
                # transform each shape to (0,0) coordinate and ...
                # sort each shape
                shape[i].sort()
                for j in range(1, len(shape[i])):
                    shape[i][j][0] -= shape[i][0][0]
                    shape[i][j][1] -= shape[i][0][1]
                shape[i][0][0] = 0
                shape[i][0][1] = 0

            # among all possible shapes, save only the smallest one as representative of all shapes
            shape.sort()
            representative = [tuple(shape[0][i]) for i in range(len(shape[0]))]
            return tuple(representative)

        if not grid:
            return 0
        r_len, c_len, islands, direction = len(grid), len(grid[0]), set(), [(0, 1), (1, 0), (-1, 0), (0, -1)]
        for i in range(r_len):
            for j in range(c_len):
                island = []
                if grid[i][j] == 1:
                    dfs(i, j)
                    shape = normalizeGridPoint()
                    islands.add(shape)

        return len(islands)


if __name__ == '__main__':
    grid = [[1,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],[0,0,0,1,1]]
    print(Solution.numDistinctIslands2(grid))

"""
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

An island is considered to be the same as another if they have the same shape, or have the same shape after rotation (90, 180, or 270 degrees only) or reflection (left/right direction or up/down direction).

Return the number of distinct islands.

 

Example 1:


Input: grid = [[1,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],[0,0,0,1,1]]
Output: 1
Explanation: The two islands are considered the same because if we make a 180 degrees clockwise rotation on the first island, then two islands will have the same shapes.
Example 2:


Input: grid = [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
Output: 1
 

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 50
grid[i][j] is either 0 or 1.
"""