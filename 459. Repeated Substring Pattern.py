class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return self.KMP(s)
        return self.greedy(s)

    def KMP(self, s):
        # if a string can be constructed by repeated substring addtion,
        # smallest block = len(s) - lps(s)
        lps = [0] * len(s)
        previous_lps, right = 0, 1

        while right < len(s):
            if s[right] == s[previous_lps]:
                previous_lps += 1
                lps[right] = previous_lps
                right += 1
            elif previous_lps == 0:
                lps[right] = 0
                right += 1
            else:
                previous_lps = lps[previous_lps - 1]

        if lps[-1] != 0 and (len(s) % (len(s) - lps[-1])) == 0:
            return True
        return False

    def greedy(self, s):
        ss = (s + s)[1:-1]
        if ss.find(s) == -1:
            return False
        return True


if __name__ == '__main__':
    s = "abcabcabcabc"
    sol = Solution()
    print(sol.repeatedSubstringPattern(s))

"""
Given a string s, check if it can be constructed by taking a substring of it and appending multiple copies of the substring together.

 

Example 1:

Input: s = "abab"
Output: true
Explanation: It is the substring "ab" twice.
Example 2:

Input: s = "aba"
Output: false
Example 3:

Input: s = "abcabcabcabc"
Output: true
Explanation: It is the substring "abc" four times or the substring "abcabc" twice.
"""