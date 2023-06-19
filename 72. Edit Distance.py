class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        def minDistance1D(word1, word2):
            len_w1, len_w2 = len(word1), len(word2)
            dp = [[0 for _ in range(len_w2 + 1)] for _ in range(2)]
            for i in range(len_w1 + 1):
                for j in range(len_w2 + 1):
                    if i == 0: #insert j item
                        dp[i & 1][j] = j
                    elif j == 0: #delete i item
                        dp[i & 1][j] = i
                    else:
                        if word1[i - 1] == word2[j - 1]:
                            dp[i & 1][j] = dp[1 - (i & 1)][j - 1]
                        else:
                            # delete, insert, replace
                            dp[i & 1][j] = 1 + min(dp[1 - (i & 1)][j], dp[i & 1][j - 1], dp[1 - (i & 1)][j - 1])
            return dp[i & 1][-1]
        return minDistance1D(word1, word2)

        def minDistance2D(word1, word2):
            len_w1, len_w2 = len(word1), len(word2)
            dp = [[0 for _ in range(len_w2 + 1)] for _ in range(len_w1 + 1)]
            for i in range(len_w1 + 1):
                for j in range(len_w2 + 1):
                    if i == 0: #insert j item
                        dp[i][j] = j
                    elif j == 0:#delete i item
                        dp[i][j] = i
                    else:
                        if word1[i-1] == word2[j-1]:
                            dp[i][j] = dp[i-1][j-1]
                        else:
                            # delete, insert, replace
                            dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            return dp[-1][-1]
        return minDistance2D(word1, word2)

        def dfs(w1Index, w2Index):
            if w1Index == len(word1): # insert remaining char
                return len(word2) - w2Index
            elif w2Index == len(word2): # delete remaining char
                return len(word1) - w1Index
            elif dp[w1Index][w2Index] == -1:
                if word1[w1Index] == word2[w2Index]:
                    dp[w1Index][w2Index] = dfs(w1Index + 1, w2Index + 1)
                else:
                    # min of insert, delete and replace
                    dp[w1Index][w2Index] = 1 + min(dfs(w1Index, w2Index + 1), dfs(w1Index + 1, w2Index), dfs(w1Index + 1, w2Index + 1))
            return dp[w1Index][w2Index]
        dp = [[-1 for _ in range(len(word2))]for _ in range(len(word1))]
        return dfs(0, 0)


if __name__ == '__main__':
    print(Solution().minDistance("intention", "execution"))
