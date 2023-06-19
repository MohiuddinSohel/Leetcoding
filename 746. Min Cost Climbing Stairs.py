from functools import lru_cache
from typing import List


class Solution:
    @staticmethod
    def minCostClimbingStairs(cost: List[int]) -> int:
        # 2 <= cost.length <= 1000, so no edge case check
        # jump start from outside the cost beginning and end outside of cost ending
        dp = [0] * (len(cost) + 2)
        dp[1] = cost[0]  # only one jump, if 2 jump is made, it will be outside dp[i]
        cost.append(0)
        for i in range(2, len(dp)):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 1])
        return dp[-1]

    def minCostClimbingStairs1(self, cost: List[int]) -> int:
        @lru_cache
        def mc(i):
            if i <= 1:
                return 0
            return min(mc(i - 1) + cost[i - 1], mc(i - 2) + cost[i - 2])

        def mcIterative():
            i0 = i1 = 0
            for i in range(2, len(cost) + 1):
                i0, i1 = i1, min(i0 + cost[i - 2], i1 + cost[i - 1])
            return i1

        # return mc(len(cost))
        return mcIterative()


if __name__ == '__main__':
    cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
    print(Solution.minCostClimbingStairs(cost))