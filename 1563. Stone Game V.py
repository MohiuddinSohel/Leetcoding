from typing import List

class Solution:
    def stoneGameV(self, stoneValue: List[int]) -> int:  # TLE, n^2logn or n^2 solution possible
        def dfs(s, e):
            if s >= e:
                return 0
            elif dp[s][e] == 0:
                best = 0
                for k in range(s, e):
                    # left subarray (s, k), right subarray (k+1, e)
                    lSum, rSum = cSum[k + 1] - cSum[s], cSum[e + 1] - cSum[k + 1]
                    # bob throws away subarray with max(lSum, rSum)
                    if lSum == rSum:
                        best = max(best, max(lSum + dfs(s, k), rSum + dfs(k + 1, e)))
                    elif lSum > rSum:
                        best = max(best, rSum + dfs(k + 1, e))
                    else:
                        best = max(best, lSum + dfs(s, k))
                dp[s][e] = best
            return dp[s][e]

        def iterative():
            c_sum = [0] * (len(stoneValue) + 1)
            for i, v in enumerate(stoneValue):
                c_sum[i + 1] = c_sum[i] + v

            s_len = len(stoneValue)
            dp = [[0 for _ in range(s_len)] for _ in range(s_len)]

            for s in range(s_len - 2, -1, -1):
                for e in range(s + 1, s_len):
                    for k in range(s, e):
                        # left subarray (s, k), right subarray (k+1, e)
                        l_sum, r_sum = c_sum[k + 1] - c_sum[s], c_sum[e + 1] - c_sum[k + 1]
                        # bob throws away subarray with max(lSum, rSum)
                        if l_sum == r_sum:
                            dp[s][e] = max(dp[s][e], l_sum + dp[s][k], r_sum + dp[k + 1][e])
                        elif l_sum < r_sum:
                            dp[s][e] = max(dp[s][e], l_sum + dp[s][k])
                        else:
                            dp[s][e] = max(dp[s][e], r_sum + dp[k + 1][e])
            return dp[0][s_len - 1]

        return iterative()

        cSum = [0] * (len(stoneValue) + 1)
        for i, v in enumerate(stoneValue):
            cSum[i + 1] = cSum[i] + v

        sLen = len(stoneValue)
        dp = [[0 for _ in range(sLen)] for _ in range(sLen)]
        return dfs(0, sLen - 1)


if __name__ == '__main__':
    stoneValue = [6, 2, 3, 4, 5, 5]
    print(Solution().stoneGameV(stoneValue))