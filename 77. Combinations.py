from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def dfs(current, count, c_chosen):
            if count == k:
                result.append(c_chosen[:])
                return result

            c_chosen.append(current)
            dfs(current + 1, count + 1, c_chosen) # choose
            c_chosen.pop()

            # we not choose current, only when there is enough left to add
            if current + (k - count) <= n:
                dfs(current + 1, count, c_chosen) # not choose
            return result

        result = []
        return dfs(1, 0, [])




"""
Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].

You may return the answer in any order.



Example 1:

Input: n = 4, k = 2
Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
Explanation: There are 4 choose 2 = 6 total combinations.
Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to be the same combination.
Example 2:

Input: n = 1, k = 1
Output: [[1]]
Explanation: There is 1 choose 1 = 1 total combination.


Constraints:

1 <= n <= 20
1 <= k <= n
"""