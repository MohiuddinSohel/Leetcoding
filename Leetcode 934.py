from collections import deque


class Solution:
    def shortestBridge(A) -> int:

        def dfs(r, c):
            island2.add((r, c))
            for x, y in direction:
                n_r, n_c = x + r, y + c
                if 0 <= n_r < r_len and 0 <= n_c < c_len and (n_r, n_c) not in island2 and A[n_r][n_c]:
                    dfs(n_r, n_c)

        def bidirectional_bfs(island1, island2):
            # visited1 and visited2 are island1 and island2
            queue1, queue2 = deque(list(island1)), deque(list(island2))
            flip1 = flip2 = 0
            len1, len2 = len(queue1), len(queue2)
            while queue1 or queue2:

                for _ in range(len1):
                    r, c = queue1.popleft()
                    for x, y in direction:
                        n_r, n_c = x + r, y + c
                        if (n_r, n_c) in island2:
                            return flip1 + flip2

                        if 0 <= n_r < r_len and 0 <= n_c < c_len and (n_r, n_c) not in island1:
                            island1.add((n_r, n_c))
                            queue1.append((n_r, n_c))
                flip1 += 1
                len1 = len(queue1)

                for _ in range(len2):
                    r, c = queue2.popleft()
                    for x, y in direction:
                        n_r, n_c = x + r, y + c
                        if (n_r, n_c) in island1:
                            return flip1 + flip2

                        if 0 <= n_r < r_len and 0 <= n_c < c_len and (n_r, n_c) not in island2:
                            island2.add((n_r, n_c))
                            queue2.append((n_r, n_c))
                flip2 += 1
                len2 = len(queue2)

        if not A:
            return 0
        r_len, c_len = len(A), len(A[0])
        island1 = set()
        for i in range(r_len):
            for j in range(c_len):
                if A[i][j]:
                    island1.add((i, j))

        island2, direction = set(), [(0, 1), (1, 0), (0, -1), (-1, 0)]
        r, c = island1.pop()
        dfs(r, c)
        island1.add((r, c))
        island1 -= island2

        return bidirectional_bfs(island1, island2)


if __name__ == '__main__':
    A = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
    print(Solution.shortestBridge(A))


'''
You are given an n x n binary matrix grid where 1 represents land and 0 represents water.

An island is a 4-directionally connected group of 1's not connected to any other 1's. There are exactly two islands in grid.

You may change 0's to 1's to connect the two islands to form one island.

Return the smallest number of 0's you must flip to connect the two islands.

 

Example 1:

Input: grid = [[0,1],[1,0]]
Output: 1
Example 2:

Input: grid = [[0,1,0],[0,0,0],[0,0,1]]
Output: 2
Example 3:

Input: grid = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
Output: 1
 

Constraints:

n == grid.length == grid[i].length
2 <= n <= 100
grid[i][j] is either 0 or 1.
There are exactly two islands in grid.
'''