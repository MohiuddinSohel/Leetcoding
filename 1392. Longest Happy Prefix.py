class Solution:
    def longestPrefix(self, s: str) -> str:
        lps, previous, right = [0] * len(s), 0, 1
        while right < len(s):
            if s[right] == s[previous]:
                previous += 1
                lps[right] = previous
                right += 1
            elif previous:
                previous = lps[previous - 1]
            else:
                lps[right] = previous  # previous = 0 here
                right += 1
        # print(lps)
        return ''.join(s[len(s) - lps[-1]:]) if lps[-1] else ""
"""
A string is called a happy prefix if is a non-empty prefix which is also a suffix (excluding itself).

Given a string s, return the longest happy prefix of s. Return an empty string "" if no such prefix exists.

 

Example 1:

Input: s = "level"
Output: "l"
Explanation: s contains 4 prefix excluding itself ("l", "le", "lev", "leve"), and suffix ("l", "el", "vel", "evel"). The largest prefix which is also suffix is given by "l".
Example 2:

Input: s = "ababab"
Output: "abab"
Explanation: "abab" is the largest prefix which is also suffix. They can overlap in the original string.
 

Constraints:

1 <= s.length <= 105
s contains only lowercase English letters.
"""