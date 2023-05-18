from collections import deque


class Solution:
    def orangesRotting(self, grid) -> int:
        return self.orangesRottingHelper(grid)

    def orangesRottingHelper(self, grid) -> int:
        rLen = len(grid)
        if rLen < 0:
            return 0
        cLen = len(grid[0])
        fresh = 0
        queue = deque()
        for i in range(rLen):
            for j in range(cLen):
                if grid[i][j] == 1:
                    fresh += 1
                elif grid[i][j] == 2:
                    queue.append((0, i, j))
        if fresh == 0:
            return 0

        minutes = 0
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        while queue:
            length = len(queue)
            for _ in range(length):
                m, r, c = queue.popleft()
                for x, y in directions:
                    n_r, n_c = r + x, c + y
                    if 0 <= n_r < rLen and 0 <= n_c < cLen and grid[n_r][n_c] == 1:
                        fresh -= 1
                        queue.append((m + 1, n_r, n_c))
                        grid[n_r][n_c] = 2
                        minutes = m + 1
                if fresh == 0:
                    return minutes
        return -1


if __name__ == '__main__':
    sol = Solution()
    grid = [[2, 1, 1], [1, 1, 0], [0, 1, 1]]
    print(sol.orangesRotting(grid))
