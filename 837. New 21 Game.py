class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        def linear_time():
            dp = [0] * (n+1)
            dp[k : min(k + maxPts, n+1)] = [1]* (min(k + maxPts - 1, n) - k + 1)
            window_sum = (min(k + maxPts - 1, n) - k + 1)
            for current in range(k-1, -1, -1):
                dp[current] = window_sum * (1 / maxPts)
                if current + maxPts <= n:
                    window_sum -= dp[current + maxPts]
                window_sum += dp[current]
            return dp[0]
        return linear_time()

        def repetitive():
            dp = [0] * (n+1)
            dp[k : min(k + maxPts, n+1)] = [1]* (min(k + maxPts - 1, n) - k + 1)
            for current in range(k-1, -1, -1):
                for next in range(current + 1, min(current + maxPts + 1, n+1)):
                    dp[current] += (dp[next] * (1 /maxPts))
            return dp[0]
        return repetitive()

        def dfs(c_point):
            if c_point >= k:
                return 1
            elif dp[c_point] == -1:
                dp[c_point] = 0
                for current in range(c_point + 1, min(c_point + maxPts + 1, n+1)):
                    dp[c_point] += (dfs(current) * (1 /maxPts))

            return dp[c_point]
        dp = [-1] * (n+1)
        return dfs(0)


if __name__ == '__main__':
    n, k, maxPts = 21, 17, 100
    print(Solution().new21Game(n, k, maxPts))