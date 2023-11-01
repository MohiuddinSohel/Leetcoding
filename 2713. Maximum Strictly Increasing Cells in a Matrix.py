from collections import defaultdict
from typing import List


class Solution:
    def maxIncreasingCells(self, mat: List[List[int]]) -> int:
        val_map, n, m = defaultdict(list), len(mat), len(mat[0])
        for i in range(n):
            for j in range(m):
                val_map[mat[i][j]].append((i, j))

        row_max, col_max, max_visited = [0] * n, [0] * m, 0
        dp = [[0] * m for _ in range(n)]

        # all equal vals ending chain are calculated first
        for val in sorted(val_map):
            for r, c in val_map[val]:
                dp[r][c] = max(row_max[r], col_max[c]) + 1
                max_visited = max(max_visited, dp[r][c])

            for r, c in val_map[val]:
                row_max[r] = max(dp[r][c], row_max[r])
                col_max[c] = max(dp[r][c], col_max[c])

        return max_visited
"""
Given a 1-indexed m x n integer matrix mat, you can select any cell in the matrix as your starting cell.

From the starting cell, you can move to any other cell in the same row or column, but only if the value of the destination cell is strictly greater than the value of the current cell. You can repeat this process as many times as possible, moving from cell to cell until you can no longer make any moves.

Your task is to find the maximum number of cells that you can visit in the matrix by starting from some cell.

Return an integer denoting the maximum number of cells that can be visited.



Example 1:



Input: mat = [[3,1],[3,4]]
Output: 2
Explanation: The image shows how we can visit 2 cells starting from row 1, column 2. It can be shown that we cannot visit more than 2 cells no matter where we start from, so the answer is 2.
Example 2:



Input: mat = [[1,1],[1,1]]
Output: 1
Explanation: Since the cells must be strictly increasing, we can only visit one cell in this example.
Example 3:



Input: mat = [[3,1,6],[-9,5,7]]
Output: 4
Explanation: The image above shows how we can visit 4 cells starting from row 2, column 1. It can be shown that we cannot visit more than 4 cells no matter where we start from, so the answer is 4.


Constraints:

m == mat.length
n == mat[i].length
1 <= m, n <= 105
1 <= m * n <= 105
-105 <= mat[i][j] <= 105
"""