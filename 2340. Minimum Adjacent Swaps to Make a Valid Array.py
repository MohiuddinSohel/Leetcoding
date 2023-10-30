class Solution:
    def minimumSwaps(self, nums: List[int]) -> int:
        smallest, smallest_index, largest, largest_index = nums[0], 0, nums[0], 0
        for i in range(1, len(nums)):
            if smallest > nums[i]:
                smallest, smallest_index = nums[i], i
            if largest <= nums[i]:
                largest, largest_index = nums[i], i
        # print(smallest, smallest_index, largest, largest_index)
        if smallest_index == largest_index:
            return 0
        swap = len(nums) - 1 - largest_index + smallest_index
        if smallest_index > largest_index:
            swap -= 1
        return swap
    """
    You are given a 0-indexed integer array nums.

Swaps of adjacent elements are able to be performed on nums.

A valid array meets the following conditions:

The largest element (any of the largest elements if there are multiple) is at the rightmost position in the array.
The smallest element (any of the smallest elements if there are multiple) is at the leftmost position in the array.
Return the minimum swaps required to make nums a valid array.

 

Example 1:

Input: nums = [3,4,5,5,3,1]
Output: 6
Explanation: Perform the following swaps:
- Swap 1: Swap the 3rd and 4th elements, nums is then [3,4,5,3,5,1].
- Swap 2: Swap the 4th and 5th elements, nums is then [3,4,5,3,1,5].
- Swap 3: Swap the 3rd and 4th elements, nums is then [3,4,5,1,3,5].
- Swap 4: Swap the 2nd and 3rd elements, nums is then [3,4,1,5,3,5].
- Swap 5: Swap the 1st and 2nd elements, nums is then [3,1,4,5,3,5].
- Swap 6: Swap the 0th and 1st elements, nums is then [1,3,4,5,3,5].
It can be shown that 6 swaps is the minimum swaps required to make a valid array.
Example 2:
Input: nums = [9]
Output: 0
Explanation: The array is already valid, so we return 0.
 

Constraints:

1 <= nums.length <= 105
1 <= nums[i] <= 105
    """