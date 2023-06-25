from typing import List


class Solution:
    def partition(self, s: str) -> List[List[str]]:

        def can_partition(start, current):
            if start == len(s):
                result.append(current[:])
                return result
            for p_index in range(start, len(s)):
                # partition at p_index, new s starts at p_index + 1
                # dp[start+1][p_index-1] is calculted in previous call
                if s[start] == s[p_index] and (p_index - start < 2 or dp[start + 1][p_index - 1]):
                    dp[start][p_index] = True
                    current.append(s[start:p_index + 1])
                    can_partition(p_index + 1, current)
                    current.pop()
            return result

        dp = [[False for _ in range(len(s))] for _ in range(len(s))]
        result = []
        return can_partition(0, [])

"""
Given a string s, partition s such that every
substring
 of the partition is a
palindrome
. Return all possible palindrome partitioning of s.



Example 1:

Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
Example 2:

Input: s = "a"
Output: [["a"]]


Constraints:

1 <= s.length <= 16
s contains only lowercase English letters.
"""