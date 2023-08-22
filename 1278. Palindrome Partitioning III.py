class Solution:
    def palindromePartition(self, s: str, k: int) -> int:
        def cost(i, j):
            if dp[i][j] != -1:
                return dp[i][j]
            else:
                if s[i] == s[j]:
                    dp[i][j] = 0 if j - i <= 1 else cost(i + 1, j - 1)
                else:
                    dp[i][j] = 1 + (0 if j - i <= 1 else cost(i + 1, j - 1))
            return dp[i][j]

        @lru_cache(None)
        def recur(i, k):
            if k == len(s) - i:
                return 0
            elif k > len(s) - i:
                return inf
            # elif i >= len(s): # redundant condition
            #     return inf
            elif k == 1:
                return cost(i, len(s) - 1)
            else:
                cos = inf
                for j in range(i, len(s) - 1):
                    cos = min(cos, cost(i, j) + recur(j + 1, k - 1))
                return cos

        dp = [[-1 for _ in range(len(s))] for _ in range(len(s))]
        return recur(0, k)

"""
You are given a string s containing lowercase letters and an integer k. You need to :

First, change some characters of s to other lowercase English letters.
Then divide s into k non-empty disjoint substrings such that each substring is a palindrome.
Return the minimal number of characters that you need to change to divide the string.

 

Example 1:

Input: s = "abc", k = 2
Output: 1
Explanation: You can split the string into "ab" and "c", and change 1 character in "ab" to make it palindrome.
Example 2:

Input: s = "aabbc", k = 3
Output: 0
Explanation: You can split the string into "aa", "bb" and "c", all of them are palindrome.
Example 3:

Input: s = "leetcode", k = 8
Output: 0
 

Constraints:

1 <= k <= s.length <= 100.
s only contains lowercase English letters.
"""