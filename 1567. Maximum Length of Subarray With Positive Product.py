class Solution:
    def getMaxLen(self, nums: List[int]) -> int:
        # dp(i, +) = max subarray length ending at i with positive product(nums[i] must be included)
        # dp(i, -) = max subarray length ending at i with negative product (nums[i] must be included)
        # based on sign of nums[i] and dp[i-1, +] and dp[i-1, -] we can calculate new subarray ending at i
        max_postive_arr_len = max_negative_arr_len = max_len = 0
        for num in nums:
            if num == 0:
                 max_postive_arr_len = max_negative_arr_len = 0
            elif num > 0:
                max_postive_arr_len += 1
                if max_negative_arr_len:
                    max_negative_arr_len += 1
            else:
                temp = max_postive_arr_len
                max_postive_arr_len = (max_negative_arr_len + 1) if max_negative_arr_len else 0
                max_negative_arr_len = temp + 1
            max_len = max(max_len, max_postive_arr_len)
        return max_len
"""
Given an array of integers nums, find the maximum length of a subarray where the product of all its elements is positive.

A subarray of an array is a consecutive sequence of zero or more values taken out of that array.

Return the maximum length of a subarray with positive product.

 

Example 1:

Input: nums = [1,-2,-3,4]
Output: 4
Explanation: The array nums already has a positive product of 24.
Example 2:

Input: nums = [0,1,-2,-3,-4]
Output: 3
Explanation: The longest subarray with positive product is [1,-2,-3] which has a product of 6.
Notice that we cannot include 0 in the subarray since that'll make the product 0 which is not positive.
Example 3:

Input: nums = [-1,-2,-3,0,1]
Output: 2
Explanation: The longest subarray with positive product is [-1,-2] or [-2,-3].
 

Constraints:

1 <= nums.length <= 105
-109 <= nums[i] <= 109
"""