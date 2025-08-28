class Solution:
    @staticmethod
    def largestIsland(A) -> int:
        if not A:
            return 0

        class DisjointSet:
            def __init__(self):
                self.parent = {}
                self.rank = {}

            def find(self, x):
                if x not in self.parent:
                    self.parent[x] = x
                    self.rank[x] = 1
                    return x

                # Path compression
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x, y):
                root_x = self.find(x)
                root_y = self.find(y)

                if root_x == root_y:
                    return

                # Union by rank
                if self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                    self.rank[root_y] += self.rank[root_x]
                else:
                    self.parent[root_y] = root_x
                    self.rank[root_x] += self.rank[root_y]

            def get_size(self, x):
                return self.rank[self.find(x)]

        r_len, c_len = len(A), len(A[0])
        ds = DisjointSet()
        direction = [(0, -1), (-1, 0), (1, 0), (0, 1)]

        # Connect islands
        for r in range(r_len):
            for c in range(c_len):
                if A[r][c]:
                    curr = (r, c)
                    # Ensure the current cell is in the disjoint set
                    ds.find(curr)

                    for dx, dy in direction:
                        n_r, n_c = r + dx, c + dy
                        neighbor = (n_r, n_c)

                        if 0 <= n_r < r_len and 0 <= n_c < c_len and A[n_r][n_c]:
                            ds.union(curr, neighbor)

        # Find the largest island after changing one 0 to 1
        max_size, all_one = 0, True
        for r in range(r_len):
            for c in range(c_len):
                if not A[r][c]:
                    all_one = False
                    connected_islands = set()
                    size = 1  # Start with 1 for the current cell

                    for dx, dy in direction:
                        n_r, n_c = r + dx, c + dy
                        neighbor = (n_r, n_c)

                        if 0 <= n_r < r_len and 0 <= n_c < c_len and A[n_r][n_c]:
                            root = ds.find(neighbor)
                            if root not in connected_islands:
                                connected_islands.add(root)
                                size += ds.get_size(root)

                    max_size = max(max_size, size)

        return max_size if not all_one else r_len * c_len


if __name__ == '__main__':
    grid = [[1, 1], [1, 0]]
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

