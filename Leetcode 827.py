class Solution:

    @staticmethod
    def largestIsland(A) -> int:
        if not A:
            return 0

        def find(parent, x):
            while x in parent and x != parent[x]:
                if parent[x] in parent:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
            return x

        def unionFind(parent, rank, x, y):
            xP, yP = find(parent, x), find(parent, y)
            if xP == yP:
                return
            if rank[xP] < rank[yP]:
                xP, yP = yP, xP
            rank[xP] += rank[yP]
            parent[yP] = xP

        parent, rank, direction, r_len, c_len = {}, {}, [(0, -1), (-1, 0), (1, 0), (0, 1)], len(A), len(A[0])
        for r in range(r_len):
            for c in range(c_len):
                if A[r][c]:
                    if (r, c) not in parent:
                        parent[(r, c)], rank[(r, c)] = (r, c), 1
                    for dx, dy in direction:
                        n_r, n_c = dx + r, dy + c
                        if 0 <= n_r < r_len and 0 <= n_c < c_len and A[n_r][n_c]:
                            if (n_r, n_c) not in parent:
                                parent[(n_r, n_c)], rank[(n_r, n_c)] = (n_r, n_c), 1
                            unionFind(parent, rank, (r, c), (n_r, n_c))

        max_size, all_one = 0, True
        for r in range(r_len):
            for c in range(c_len):
                count, islandList = 1, set()  # count = 1 for current water cell
                if not A[r][c]:
                    all_one = False
                    for dx, dy in direction:
                        n_r, n_c = dx + r, dy + c
                        if 0 <= n_r < r_len and 0 <= n_c < c_len and A[n_r][n_c]:
                            p = find(parent, (n_r, n_c))
                            if p not in islandList:
                                count += rank[p]
                                islandList.add(p)
                    max_size = max(max_size, count)

        return max_size if not all_one else r_len * c_len


if __name__ == '__main__':
    grid = [[1,1],[1,0]]
    print(Solution.largestIsland(grid))

'''
You are given an n x n binary matrix grid. You are allowed to change at most one 0 to be 1.

Return the size of the largest island in grid after applying this operation.

An island is a 4-directionally connected group of 1s.

 

Example 1:

Input: grid = [[1,0],[0,1]]
Output: 3
Explanation: Change one 0 to 1 and connect two 1s, then we get an island with area = 3.
Example 2:

Input: grid = [[1,1],[1,0]]
Output: 4
Explanation: Change the 0 to 1 and make the island bigger, only one island with area = 4.
Example 3:

Input: grid = [[1,1],[1,1]]
Output: 4
Explanation: Can't change any 0 to 1, only one island with area = 4.
 

Constraints:

n == grid.length
n == grid[i].length
1 <= n <= 500
grid[i][j] is either 0 or 1.
    '''