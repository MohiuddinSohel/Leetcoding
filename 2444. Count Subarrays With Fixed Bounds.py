class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        # consider subarray containing most recent minK and maxK, and  ending at i
        left, recent_minK, recent_maxK, count = 0, -1, -1, 0
        for right, num in enumerate(nums):
            if num <  minK or num > maxK:
                left = right + 1
            if num == minK:
                recent_minK = right
            if num == maxK:
                recent_maxK = right
            count += max(0, min(recent_minK, recent_maxK) - left + 1)
        return count
"""
You are given an integer array nums and two integers minK and maxK.

A fixed-bound subarray of nums is a subarray that satisfies the following conditions:

The minimum value in the subarray is equal to minK.
The maximum value in the subarray is equal to maxK.
Return the number of fixed-bound subarrays.

A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [1,3,5,2,7,5], minK = 1, maxK = 5
Output: 2
Explanation: The fixed-bound subarrays are [1,3,5] and [1,3,5,2].
Example 2:

Input: nums = [1,1,1,1], minK = 1, maxK = 1
Output: 10
Explanation: Every subarray of nums is a fixed-bound subarray. There are 10 possible subarrays.
 

Constraints:

2 <= nums.length <= 105
1 <= nums[i], minK, maxK <= 106
"""