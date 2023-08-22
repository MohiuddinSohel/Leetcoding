class Solution:
    def maxPalindromes(self, s: str, k: int) -> int:
        # is arr[i:j]
        def isPalindrome(i, j):
            if j >= len(s) or i < 0:
                return False
            if palindrome[i][j] == -1:
                if i == j:
                    palindrome[i][j] = True
                else:
                    palindrome[i][j] = (s[i] == s[j]) and (i + 1 == j or isPalindrome(i + 1, j - 1))
            return palindrome[i][j]

        def dfsWithMemo(i):
            if i + k - 1 >= len(s):
                return 0
            if dp[i] != -1:
                return dp[i]

            if isPalindrome(i, i + k - 1):
                dp[i] = max(dp[i], 1 + dfsWithMemo(i + k))
            if isPalindrome(i, i + k):
                dp[i] = max(dp[i], 1 + dfsWithMemo(i + k + 1))
            dp[i] = max(dp[i], dfsWithMemo(i + 1))
            return dp[i]

        # dp[i]= number of palindrome until s index i-1
        # if there is a palindrom of length k+n, there must be a palindrom of k or k+1
        # considering the minimum one (at least k) will maximize total number of palindrom
        def iterative():
            dp = [0] * (len(s) + 1)
            for i in range(k, len(s) + 1):
                dp[i] = dp[i - 1]  # if s[i-1] is not part of palindrom staring at s[i-k], not including s[i]
                if isPalindrome(i - k, i - 1):  # k length palindrome
                    dp[i] = max(dp[i], 1 + dp[i - k])
                if isPalindrome(i - k - 1, i - 1):  # k+1 length palindrom staring at s[i-k-1], not including s[i]
                    dp[i] = max(dp[i], 1 + dp[i - k - 1])
            return dp[-1]

        palindrome = [[-1 for _ in range(len(s) + 1)] for _ in range(len(s))]
        return iterative()

        dp = [-1] * (len(s))
        count = dfsWithMemo(0)
        return count if count >= 0 else 0

    """
    You are given a string s and a positive integer k.

Select a set of non-overlapping substrings from the string s that satisfy the following conditions:

The length of each substring is at least k.
Each substring is a palindrome.
Return the maximum number of substrings in an optimal selection.

A substring is a contiguous sequence of characters within a string.

 

Example 1:

Input: s = "abaccdbbd", k = 3
Output: 2
Explanation: We can select the substrings underlined in s = "abaccdbbd". Both "aba" and "dbbd" are palindromes and have a length of at least k = 3.
It can be shown that we cannot find a selection with more than two valid substrings.
Example 2:

Input: s = "adbcda", k = 2
Output: 0
Explanation: There is no palindrome substring of length at least 2 in the string.
 

Constraints:

1 <= k <= s.length <= 2000
s consists of lowercase English letters.
    """