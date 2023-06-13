from typing import List

class TrieNode:
    def __init__(self):
        self.count = 0
        self.children = {}


class Trie:
    def __init__(self):
        self.trie = TrieNode()

    def insert(self, row, grid):
        my_trie = self.trie
        for c in range(len(grid)):
            if grid[row][c] not in my_trie.children:
                my_trie.children[grid[row][c]] = TrieNode()
            my_trie = my_trie.children[grid[row][c]]
        my_trie.count += 1

    def search(self, col, grid):
        my_trie = self.trie
        for r in range(len(grid)):
            if grid[r][col] in my_trie.children:
                my_trie = my_trie.children[grid[r][col]]
            else:
                return 0
        return my_trie.count


class Solution:
    @staticmethod
    def equalPairs(grid: List[List[int]]) -> int:
        my_trie = Trie()
        count = 0
        n = len(grid)

        for r in range(n):
            my_trie.insert(r, grid)

        for c in range(n):
            count += my_trie.search(c, grid)

        return count


if __name__ == '__main__':
    grid = [[3,1,2,2],[1,4,4,5],[2,4,2,2],[2,4,2,2]]
    print(Solution().equalPairs(grid))