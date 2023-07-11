import math
from typing import List


class Solution:
    def minAbsDifference(self, nums: List[int], goal: int) -> int:
        def meet_in_the_middle():
            def find_equal_or_greater(arr, target):
                l, r = 0, len(arr) - 1
                while l <= r:
                    m = l + (r-l)//2
                    if arr[m] == target:
                        return m
                    elif arr[m] > target:
                        r = m - 1
                    else:
                        l = m + 1
                return l

            def find_equal_or_smaller(arr, target):
                l, r = 0, len(arr) - 1
                while l <= r:
                    m = l + (r-l)//2
                    if arr[m] == target:
                        return m
                    elif arr[m] > target:
                        r = m - 1
                    else:
                        l = m + 1
                return r

            def find_all_combination(half, index, c_sum, storage):
                if index == len(half):
                    storage.add(c_sum)
                    return storage
                find_all_combination(half, index + 1, c_sum + half[index], storage)
                find_all_combination(half, index + 1, c_sum, storage)
                return storage

            n = len(nums)
            first_half, second_half = find_all_combination(nums[:n//2], 0, 0, set()), find_all_combination(nums[n//2:], 0, 0, set())
            second_half = list(second_half)
            second_half.sort()
            min_diff = math.inf
            for s in first_half:
                target = goal - s

                index = find_equal_or_greater(second_half, target)
                if index < len(second_half):
                    min_diff = min(min_diff, abs(s + second_half[index] - goal))

                index = find_equal_or_smaller(second_half, target)
                if index >= 0:
                    min_diff = min(min_diff, abs(s + second_half[index] - goal))
            return min_diff
        return meet_in_the_middle()





        def dp_with_range(): # MLE
            def dfs(index, target):
                if target == 0:
                    return True
                elif index == n:
                    return False
                elif (index, target) not in dp:
                    dp[(index, target)] = dfs(index +1, target - nums[index]) or dfs(index + 1, target)
                return dp[(index, target)]

            n, dp = len(nums), {}
            smallest_sum = largest_sum = 0
            for num in nums:
                if num < 0:
                    smallest_sum += num
                else:
                    largest_sum += num

            smallest_sum, largest_sum = min(smallest_sum, smallest_sum + largest_sum), min(largest_sum, smallest_sum + largest_sum)
            smallest = largest = goal
            while smallest >= smallest_sum or largest <= largest_sum:
                if smallest >= smallest_sum:
                    if dfs(0, smallest):
                        return goal - smallest
                    smallest -= 1
                if largest <= largest_sum:
                    if dfs(0, largest):
                        return largest - goal
                    largest += 1
            return abs(smallest_sum + largest_sum - goal)

if __name__ == '__main__':
    nums = [7,-9,15,-2]
    goal = -5
    print(Solution().minAbsDifference(nums, goal))


"""
You are given an integer array nums and an integer goal.

You want to choose a subsequence of nums such that the sum of its elements is the closest possible to goal. That is, if the sum of the subsequence's elements is sum, then you want to minimize the absolute difference abs(sum - goal).

Return the minimum possible value of abs(sum - goal).

Note that a subsequence of an array is an array formed by removing some elements (possibly all or none) of the original array.

 

Example 1:

Input: nums = [5,-7,3,5], goal = 6
Output: 0
Explanation: Choose the whole array as a subsequence, with a sum of 6.
This is equal to the goal, so the absolute difference is 0.
Example 2:

Input: nums = [7,-9,15,-2], goal = -5
Output: 1
Explanation: Choose the subsequence [7,-9,-2], with a sum of -4.
The absolute difference is abs(-4 - (-5)) = abs(1) = 1, which is the minimum.
Example 3:

Input: nums = [1,2,3], goal = -7
Output: 7
"""