import math
from typing import List


class Solution:
    def minimumDifference(self, nums: List[int]) -> int:
        def findMinDiff(index, count, cSum):
            nonlocal gMin
            if gMin == 0:
                return
            elif count == n / 2:
                gMin = min(gMin, abs(2 * cSum - total))
                return
            elif index >= n:
                return
            else:
                findMinDiff(index + 1, count + 1, cSum + nums[index])
                findMinDiff(index + 1, count, cSum)

        # Brute Force, TC = 2^(2n)
        # total = sum(nums)
        # n = len(nums)
        # gMin = float('inf')
        # findMinDiff(0, 0, 0)
        # return gMin

        def getPossibleSum(index, count, cSum, result, nums):
            if count != 0:
                result[count].add(cSum)
            if index == n:
                return result
            getPossibleSum(index + 1, count + 1, cSum + nums[index], result, nums)
            getPossibleSum(index + 1, count, cSum, result, nums)
            return result

        def binarySearch(array, target):
            l, r = 0, len(array) - 1
            while l <= r:
                m = l + (r - l) // 2
                if array[m] > target:
                    r = m - 1
                else:
                    l = m + 1
            return array[r]

        # meet in the middle, tc = 2^(n/2) + 2^(n/2)+ (n/2)(n/2 * log(n/2) + log(n/2))
        total = sum(nums)
        n = len(nums) // 2
        l_sum = [set() for _ in range(n + 1)]
        # calculate possible sum containing 1,2,3,..., n element
        l_sum = getPossibleSum(0, 0, 0, l_sum, nums[:n])

        r_sum = [set() for _ in range(n + 1)]
        # calculate possible sum containing 1,2,3,..., n element
        r_sum = getPossibleSum(0, 0, 0, r_sum, nums[n:])

        gMin = float('inf')
        # minimize abs((lS+rS) - (total - (lS+rS)))
        for i in range(1, len(l_sum)):
            array = list(r_sum[n - i])
            array.sort()
            for lS in list(l_sum[i]):
                target = total / 2 - lS
                if n - i == 0:
                    rS = 0
                else:
                    rS = binarySearch(array, target)
                gMin = min(gMin, abs((lS + rS) - (total - (lS + rS))))
                if gMin == 0:
                    return 0
        return gMin




if __name__ == '__main__':
    nums = [3,9,7,3]
    goal = -5
    print(Solution().minimumDifference(nums))
"""
You are given an integer array nums of 2 * n integers. You need to partition nums into two arrays of length n to minimize the absolute difference of the sums of the arrays. To partition nums, put each element of nums into one of the two arrays.

Return the minimum possible absolute difference.

 

Example 1:

example-1
Input: nums = [3,9,7,3]
Output: 2
Explanation: One optimal partition is: [3,9] and [7,3].
The absolute difference between the sums of the arrays is abs((3 + 9) - (7 + 3)) = 2.
Example 2:

Input: nums = [-36,36]
Output: 72
Explanation: One optimal partition is: [-36] and [36].
The absolute difference between the sums of the arrays is abs((-36) - (36)) = 72.
Example 3:

example-3
Input: nums = [2,-1,0,4,-2,-9]
Output: 0
Explanation: One optimal partition is: [2,4,-9] and [-1,0,-2].
The absolute difference between the sums of the arrays is abs((2 + 4 + -9) - (-1 + 0 + -2)) = 0.
 

Constraints:

1 <= n <= 15
nums.length == 2 * n
-107 <= nums[i] <= 107
"""
