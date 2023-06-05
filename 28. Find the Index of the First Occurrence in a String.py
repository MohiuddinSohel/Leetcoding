class Solution:
    @staticmethod
    def strStr(haystack: str, needle: str) -> int:
        def KMP():
            if len(needle) == 0:
                return 0
            #pre-process needle
            lps = [0] * len(needle)
            previous_lps, right = 0, 1
            while right < len(needle):
                if needle[right] == needle[previous_lps] or not previous_lps:
                    if needle[right] == needle[previous_lps]:
                        previous_lps += 1
                    lps[right] = previous_lps
                    right += 1
                else:
                    previous_lps = lps[previous_lps - 1]
            #KMP
            index_haystack, index_needle = 0, 0
            while index_haystack < len(haystack):
                if haystack[index_haystack] == needle[index_needle]:
                    index_needle += 1
                    index_haystack += 1
                else:
                    if index_needle == 0:
                        index_haystack += 1
                    else:
                        index_needle = lps[index_needle - 1]
                if index_needle == len(needle):
                    return index_haystack - index_needle
            return -1
        return KMP()

if __name__ == '__main__':
    haystack = "sadbutsad"
    needle = "sad"
    print(Solution.strStr(haystack, needle))

    """
    Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

 

Example 1:

Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.
Example 2:

Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.
    """