from typing import List


class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # result = []
        nums.sort()
        return self.dfs(nums, set(), [], [])
        # return result
        return self.permutation(0, nums, [])

    def dfs(self, nums, visited, current, result):
        if len(nums) == len(current):
            result.append(current[:])
            return result
        for i in range(len(nums)):
            if i in visited:
                continue
            if i > 0 and nums[i] == nums[i - 1] and i - 1 not in visited:
                continue
            visited.add(i)
            current.append(nums[i])
            self.dfs(nums, visited, current, result)
            visited.remove(i)
            current.pop()
        return result

"""
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.



Example 1:

Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]
Example 2:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]


Constraints:

1 <= nums.length <= 8
-10 <= nums[i] <= 10
"""