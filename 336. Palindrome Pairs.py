from collections import defaultdict
from typing import List


class Node:
    def __init__(self):
        self.trie = defaultdict(Node)
        self.end = -1
        self.palindrome_suffix = []


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word, index):
        current = self.root
        for i, c in enumerate(word):
            if word[i:] == word[i:][::-1]:
                current.palindrome_suffix.append(index)
            current = current.trie[c]
        current.end = index


class Solution:
    # feasible for online algorithm, have to maintain two trie, one with reversed, another with original
    # to solve finding pair where current word is second
    # In single trie, current word is always first of the pair
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        def trie_solution():
            trie_root = Trie()
            for i, word in enumerate(words):
                trie_root.insert(word[::-1], i)

            result = []
            for index, word in enumerate(words):
                current = trie_root.root
                for i, c in enumerate(word):
                    # case 3: [w + p, w_reverse] current word is w + p
                    if current.end != -1 and word[i:] == word[i:][::-1]:
                        result.append([index, current.end])
                    if c not in current.trie:
                        break
                    current = current.trie[c]

                else:
                    # case 1: [w, w_reverse], current word is w
                    if current.end != -1 and current.end != index:
                        result.append([index, current.end])

                    # case 2: [w_reverse, p + w], current word is p + w
                    for second in current.palindrome_suffix:
                        result.append([index, second])

            return result

        return trie_solution()

        # not feasible for online algorithm,
        # if current word is the shorter of the two, we can not determine the otherone
        # from a shorter word, we do not know which longer word to form by adding prefix / suffix
        def haspMap_solution():
            def get_all_valid_sufix(word):
                # slicing: [i::-1], reverse [0:i+1] list element, exclude i+1
                # slicing: [:i:-1], reverse [i+1:] list element
                suffix = []
                for i, c in enumerate(word):
                    if word[:i + 1] == word[i::-1]:
                        suffix.append(word[i + 1:])
                return suffix

            def get_all_valid_prefix(word):
                prefix = []
                for i, c in enumerate(word):
                    if word[i:] == word[i:][::-1]:
                        prefix.append(word[:i])
                return prefix

            word_index, result = {word: i for i, word in enumerate(words)}, []

            for index, word in enumerate(words):
                # case 1: [w, w_reverse], current word is w
                reversed_word = word[::-1]
                if reversed_word in word_index and index != word_index[reversed_word]:
                    result.append([index, word_index[reversed_word]])

                # case 2: [w_reverse, p + w], current word is p + w
                for suffix in get_all_valid_sufix(word):
                    reversed_suffix = suffix[::-1]
                    if reversed_suffix in word_index:
                        result.append([word_index[reversed_suffix], index])

                # case 3: [w + p, w_reverse] current word is w + p
                for prefix in get_all_valid_prefix(word):
                    reversed_prefix = prefix[::-1]
                    if reversed_prefix in word_index:
                        result.append([index, word_index[reversed_prefix]])
            return result

        return haspMap_solution()

if __name__ == '__main__':
    words = ["abcd","dcba","lls","s","sssll"]
    print(Solution().palindromePairs(words))

"""
You are given a 0-indexed array of unique strings words.

A palindrome pair is a pair of integers (i, j) such that:

0 <= i, j < words.length,
i != j, and
words[i] + words[j] (the concatenation of the two strings) is a 
palindrome
.
Return an array of all the palindrome pairs of words.

 

Example 1:

Input: words = ["abcd","dcba","lls","s","sssll"]
Output: [[0,1],[1,0],[3,2],[2,4]]
Explanation: The palindromes are ["abcddcba","dcbaabcd","slls","llssssll"]
Example 2:

Input: words = ["bat","tab","cat"]
Output: [[0,1],[1,0]]
Explanation: The palindromes are ["battab","tabbat"]
Example 3:

Input: words = ["a",""]
Output: [[0,1],[1,0]]
Explanation: The palindromes are ["a","a"]
 

Constraints:

1 <= words.length <= 5000
0 <= words[i].length <= 300
words[i] consists of lowercase English letters.
"""

