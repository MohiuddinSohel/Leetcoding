class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        def dfs(current_len):
            # this approach can not be iterative since in iterativer version we will have to count string with length < low
            if current_len > high:
                return 0
            elif current_len == high:
                return 1  # current_len >= low, because of the constraint we do not need to check it against low
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
'''
Given the integers zero, one, low, and high, we can construct a string by starting with an empty string, and then at each step perform either of the following:

Append the character '0' zero times.
Append the character '1' one times.
This can be performed any number of times.

A good string is a string constructed by the above process having a length between low and high (inclusive).

Return the number of different good strings that can be constructed satisfying these properties. Since the answer can be large, return it modulo 109 + 7.

 

Example 1:

Input: low = 3, high = 3, zero = 1, one = 1
Output: 8
Explanation: 
One possible valid good string is "011". 
It can be constructed as follows: "" -> "0" -> "01" -> "011". 
All binary strings from "000" to "111" are good strings in this example.
Example 2:

Input: low = 2, high = 3, zero = 1, one = 2
Output: 5
Explanation: The good strings are "00", "11", "000", "110", and "011".
 

Constraints:

1 <= low <= high <= 105
1 <= zero, one <= low
'''