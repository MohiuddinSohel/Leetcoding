class Solution:
    @staticmethod
    def findGoodStrings(n: int, s1: str, s2: str, evil: str) -> int:
        def KMP(needle, lps):
            previous_lps, right = 0, 1
            while right < len(needle):
                if needle[previous_lps] == needle[right]:
                    lps[right] = previous_lps + 1
                    previous_lps += 1
                    right += 1
                elif previous_lps == 0:
                    lps[right] = 0
                    right += 1
                else:
                    previous_lps = lps[previous_lps - 1]

        def digit_dp_with_KMP(index, evil_index, left_tight, right_tight):
            if evil_index == len(evil):
                return 0

            elif index == n:
                return 1

            elif dp[index][evil_index][left_tight][right_tight] == -1:
                l = s1[index] if left_tight else 'a'
                r = s2[index] if right_tight else 'z'
                count = 0

                for current in range(ord(l), ord(r) + 1):
                    # KMP
                    new_evil_index = evil_index
                    while new_evil_index > 0 and evil[new_evil_index] != chr(current):
                        new_evil_index = lps[new_evil_index - 1]
                    if evil[new_evil_index] == chr(current):
                        new_evil_index += 1

                    new_left_tight = left_tight and chr(current) == s1[index]
                    new_right_tight = right_tight and chr(current) == s2[index]
                    count += digit_dp_with_KMP(index + 1, new_evil_index, new_left_tight, new_right_tight)
                    count %= mod

                dp[index][evil_index][left_tight][right_tight] = count

            return dp[index][evil_index][left_tight][right_tight]

        mod = 10 ** 9 + 7
        lps = [0] * len(evil)
        dp = [[[[-1 for _ in range(2)] for _ in range(2)] for _ in range(len(evil))] for _ in range(n)]
        KMP(evil, lps)
        return digit_dp_with_KMP(0, 0, True, True)


if __name__ == '__main__':
    n, s1, s2, evil = 2, "aa", "da", "b"
    print(Solution.findGoodStrings(n, s1, s2, evil))

"""
Given the strings s1 and s2 of size n and the string evil, return the number of good strings.

A good string has size n, it is alphabetically greater than or equal to s1, it is alphabetically smaller than or equal to s2, and it does not contain the string evil as a substring. Since the answer can be a huge number, return this modulo 109 + 7.

 

Example 1:

Input: n = 2, s1 = "aa", s2 = "da", evil = "b"
Output: 51 
Explanation: There are 25 good strings starting with 'a': "aa","ac","ad",...,"az". Then there are 25 good strings starting with 'c': "ca","cc","cd",...,"cz" and finally there is one good string starting with 'd': "da". 
Example 2:

Input: n = 8, s1 = "leetcode", s2 = "leetgoes", evil = "leet"
Output: 0 
Explanation: All strings greater than or equal to s1 and smaller than or equal to s2 start with the prefix "leet", therefore, there is not any good string.
Example 3:

Input: n = 2, s1 = "gx", s2 = "gz", evil = "x"
Output: 2
 

Constraints:

s1.length == n
s2.length == n
s1 <= s2
1 <= n <= 500
1 <= evil.length <= 50
All strings consist of lowercase English letters.
"""