from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(slot_index, current):
            if slot_index == len(nums):
                result.append(current[:])

            for nei in nums:
                if nei in visited:
                    continue
                visited.add(nei)
                current.append(nei)
                dfs(slot_index + 1, current)
                current.pop()
                visited.remove(nei)
            return result

        result, visited = [], set()
        return dfs(0, [])

        # return self.permuteHelper(nums, len(nums), 0, [], set(), [])
        # return self.permuteStack(nums)

    def permuteHelper(self, nums, n, length, current, visited, result):
        if n == length:
            result.append(current)
            return result

        for i in range(len(nums)):
            if i in visited:
                continue

            visited.add(i)
            self.permuteHelper(nums, n, length + 1, current + [nums[i]], visited, result)
            visited.remove(i)
        return result

"""
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.



Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:

Input: nums = [1]
Output: [[1]]


Constraints:

1 <= nums.length <= 6
-10 <= nums[i] <= 10
All the integers of nums are unique.
"""