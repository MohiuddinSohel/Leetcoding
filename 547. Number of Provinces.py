from typing import List


class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        return self.findCircleNumUnionPathRank(M)
        # return self.findCircleNumHelper(M)

    def findCircleNumHelper(self, M: List[List[int]]) -> int:
        def dfs(M, row, visited):
            for j in range(len(M)):
                if j not in visited and M[row][j] == 1:
                    visited.add(j)
                    dfs(M, j, visited)

        visited = set()
        count = 0
        for r in range(len(M)):
            if r in visited:
                continue
            count += 1
            dfs(M, r, visited)
        return count

    def findCircleNumUnionPathRank(self, M: List[List[int]]) -> int:
        rank = [0 for _ in range(len(M))]
        parent = [i for i in range(len(M))]
        count = len(M)

        def find(parent, x):
            while x != parent[x]:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(parent, rank, x, y):
            xP = find(parent, x)
            yP = find(parent, y)

            if xP == yP:
                return 0
            if rank[xP] < rank[yP]:
                xP, yP = yP, xP

            parent[yP] = xP
            if rank[xP] == rank[yP]:
                rank[xP] += 1
            return 1

        for i in range(len(M)):
            for j in range(i + 1, len(M[0])):
                if M[i][j] == 1:
                    count -= union(parent, rank, i, j)
        return count


if __name__ == '__main__':
    isConnected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    sol = Solution()
    print(sol.findCircleNumUnionPathRank(isConnected))
    print(sol.findCircleNumHelper(isConnected))

"""
There are n cities. Some of them are connected, while some are not. If city a is connected directly with city b, and city b is connected directly with city c, then city a is connected indirectly with city c.

A province is a group of directly or indirectly connected cities and no other cities outside of the group.

You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

Return the total number of provinces.

 

Example 1:


Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2
Example 2:


Input: isConnected = [[1,0,0],[0,1,0],[0,0,1]]
Output: 3
 

Constraints:

1 <= n <= 200
n == isConnected.length
n == isConnected[i].length
isConnected[i][j] is 1 or 0.
isConnected[i][i] == 1
isConnected[i][j] == isConnected[j][i]
"""