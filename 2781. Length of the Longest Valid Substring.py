from typing import List


class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # wondow[l: r]: if forbidden word exists from k:r, k >= l <=r,
        # we can say no substring from l:r containing k:r is valid
        # so forward l to k + 1
        # search forbidden word in reverse order, so build trie also in reverse order

        def isExist(l, r, word, trie):
            while l <= r:
                if word[r] in trie:
                    trie = trie[word[r]]
                    if '#' in trie:
                        return r + 1
                else:
                    return l
                r -= 1
            return l

        trie = {}
        for w in forbidden:
            current = trie
            for i in range(len(w) - 1, -1, -1):
                if w[i] not in current:
                    current[w[i]] = {}
                current = current[w[i]]
            current['#'] = True

        l = max_len = 0
        for r, c in enumerate(word):
            l = isExist(l, r, word, trie)
            max_len = max(max_len, r-l+1)
        return max_len



if __name__ == '__main__':
    word = "cbaaaabc"
    forbidden = ["aaa","cb"]
    print(Solution().longestValidSubstring(word, forbidden))
    a = ['a', 'b', 'c', 'd']
    print(a[-1::-1], a[0::-1],a[1::-1], a[2::-1], a[3::-1], a[4::-1])
    print(a[:-1:-1], a[:0:-1], a[:1:-1], a[:2:-1], a[:3:-1], a[:4:-1])

"""
You are given a string word and an array of strings forbidden.

A string is called valid if none of its substrings are present in forbidden.

Return the length of the longest valid substring of the string word.

A substring is a contiguous sequence of characters in a string, possibly empty.

 

Example 1:

Input: word = "cbaaaabc", forbidden = ["aaa","cb"]
Output: 4
Explanation: There are 9 valid substrings in word: "c", "b", "a", "ba", "aa", "bc", "baa", "aab", and "aabc". The length of the longest valid substring is 4. 
It can be shown that all other substrings contain either "aaa" or "cb" as a substring. 
Example 2:

Input: word = "leetcode", forbidden = ["de","le","e"]
Output: 4
Explanation: There are 11 valid substrings in word: "l", "t", "c", "o", "d", "tc", "co", "od", "tco", "cod", and "tcod". The length of the longest valid substring is 4.
It can be shown that all other substrings contain either "de", "le", or "e" as a substring. 
 

Constraints:

1 <= word.length <= 105
word consists only of lowercase English letters.
1 <= forbidden.length <= 105
1 <= forbidden[i].length <= 10
forbidden[i] consists only of lowercase English letters.


["MyLinkedList","addAtHead","addAtIndex","get","addAtHead","addAtTail","get","addAtTail","get","addAtHead","get","addAtHead"]
[[],               [5],       [1,2],      [1],     [6],       [2],     [3],     [1],      [5],    [2],      [2],    [6]]

"""