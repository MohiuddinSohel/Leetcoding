class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        def iterative1D(text1, text2):
            len_t1, len_t2 = len(text1), len(text2)
            dp = [[0 for _ in range(len_t2 + 1)] for _ in range(2)]
            for t1 in range(1, len_t1 + 1):
                for t2 in range(1, len_t2 + 1):
                    if text1[t1 - 1] == text2[t2 - 1]:
                        dp[t1 & 1][t2] = 1 + dp[1 - (t1 & 1)][t2 - 1]
                    else:
                        dp[t1 & 1][t2] = max(dp[1 - (t1 & 1)][t2], dp[t1 & 1][t2 - 1])
            return dp[t1 & 1][t2]

        return iterative1D(text1, text2)

        def iterative(text1, text2):
            len_t1, len_t2 = len(text1), len(text2)
            dp = [[0 for _ in range(len_t2 + 1)] for _ in range(len_t1 + 1)]
            for t1 in range(1, len_t1 + 1):
                for t2 in range(1, len_t2 + 1):
                    if text1[t1 - 1] == text2[t2 - 1]:
                        dp[t1][t2] = 1 + dp[t1 - 1][t2 - 1]
                    else:
                        dp[t1][t2] = max(dp[t1 - 1][t2], dp[t1][t2 - 1])
            return dp[-1][-1]

        return iterative(text1, text2)

        def dfs(t1_Index, t2_index):
            if t1_Index == len_t1 or t2_index == len_t2:
                return 0
            elif dp[t1_Index][t2_index] == -1:
                dp[t1_Index][t2_index] = 0

                if text1[t1_Index] == text2[t2_index]:
                    dp[t1_Index][t2_index] = 1 + dfs(t1_Index + 1, t2_index + 1)

                dp[t1_Index][t2_index] = max(dfs(t1_Index, t2_index + 1), dfs(t1_Index + 1, t2_index))
            return dp[t1_Index][t2_index]

        len_t1, len_t2 = len(text1), len(text2)
        dp = [[-1 for _ in range(len_t2)] for _ in range(len_t1)]
        return dfs(0, 0)



if __name__ == '__main__':
    print(Solution().longestCommonSubsequence("abcde", "ace"))


"""
Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.
"""