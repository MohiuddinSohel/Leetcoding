class Solution:
    def minCut(self, s: str) -> int:
        def isPalindrom(start, end):
            if palindrom[start][end] is None:
                if start == end:
                    palindrom[start][end] = True
                elif start + 1 == end and s[start] == s[end]:
                    palindrom[start][end] = True
                elif s[start] == s[end]:
                    palindrom[start][end] = isPalindrom(start + 1, end - 1)
                else:
                    palindrom[start][end] = False
            return palindrom[start][end]

        def brute_force(start):
            if start == len(s):
                return 0
            elif dp[start] is None:
                count = len(s) - start
                if isPalindrom(start, len(s) - 1):  # whole string is palindrom, no need to add cut
                    count = 0
                else:
                    for i in range(start, len(s) - 1):
                        if isPalindrom(start, i):
                            count = min(count, 1 + brute_force(i + 1))
                dp[start] = count
            return dp[start]

        def iterative():
            # pDp(i) = minimum number of cut for string i to end
            pDp = [inf] * len(s)
            for i in range(len(s) - 1, -1, -1):
                if isPalindrom(i, len(s) - 1):
                    pDp[i] = 0
                else:
                    for cut in range(i, len(s) - 1):
                        if isPalindrom(i, cut):
                            pDp[i] = min(pDp[i], 1 + pDp[cut + 1])
            return pDp[0] if pDp else 0

        palindrom = [[None for _ in range(len(s))] for _ in range(len(s))]
        # dp = [None]*len(s)
        # return brute_force(0)

        return iterative()
"""
Given a string s, partition s such that every 
substring
 of the partition is a 
palindrome
.

Return the minimum cuts needed for a palindrome partitioning of s.

 

Example 1:

Input: s = "aab"
Output: 1
Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.
Example 2:

Input: s = "a"
Output: 0
Example 3:

Input: s = "ab"
Output: 1
 

Constraints:

1 <= s.length <= 2000
s consists of lowercase English letters only.
"""