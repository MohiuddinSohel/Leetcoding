class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        def dfs(current_len):
            # this approach can not be iterative since in iterativer version we will have to count
            # string with length < low
            if current_len > high:
                return 0
            elif current_len == high:
                return 1  # current_len >= low
            elif dp[current_len] == -1:
                dp[current_len] = (low <= current_len <= high)
                dp[current_len] += dfs(current_len + zero) + dfs(current_len + one)
                dp[current_len] %= mod
            return dp[current_len]

        def iterative():
            dp = [0] * (high + 1)
            mod = 10 ** 9 + 7
            dp[0] = 1  # 1 way to form empty string
            for current_len in range(1, high + 1):
                if current_len - zero >= 0:
                    dp[current_len] += dp[current_len - zero]
                if current_len - one >= 0:
                    dp[current_len] += dp[current_len - one]
                dp[current_len] %= mod
            return sum(dp[low:high + 1]) % mod

        # return iterative()

        dp = [-1] * high
        mod = 10 ** 9 + 7
        return dfs(0)
