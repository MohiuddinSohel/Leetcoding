from collections import Counter, defaultdict
from typing import List


class Solution:
    def canDistribute(self, nums: List[int], quantity: List[int]) -> bool:
        def dfsWithMemo(index, mask):
            if mask == end_mask:
                return True
            elif index == len(counter_key):
                return False
            elif (index, mask) not in dp:
                sub_mask = remaining_mask = end_mask ^ mask
                # generate all submask of current remaining mask
                while sub_mask > 0:
                    # choose counter[counter_key[index]] to fill sub_mask
                    if mask_total[sub_mask] <= counter[counter_key[index]]:
                        dp[(index, mask)] = dfsWithMemo(index + 1, mask | sub_mask)
                        if dp[(index, mask)]:
                            return dp[(index, mask)]
                    sub_mask = (sub_mask - 1) & remaining_mask
                # not choosing counter[counter_key[index]]
                dp[(index, mask)] = dfsWithMemo(index + 1, mask)
            return dp[(index, mask)]

        m, counter = len(quantity), Counter(nums)
        # have to fit m quantity in m largest nums count, not in all nums
        counter_key = sorted(counter, key=lambda x: counter[x])[-m:]
        end_mask, dp, mask_total = (1 << m) - 1, defaultdict(int), defaultdict(int)

        # generate all_mask and there corresponding total
        for i in range(end_mask + 1):
            total, current_bit, current_mask = 0, 0, i
            for current_bit in range(m):
                if current_mask & (1 << current_bit):
                    total += quantity[current_bit]
            mask_total[current_mask] = total

        return dfsWithMemo(0, 0)

        def backtrack(q_i):
            if q_i == len(quantity):
                return True
            for key in counter_key:
                if counter[key] < quantity[q_i]:
                    continue
                counter[key] -= quantity[q_i]
                if backtrack(q_i + 1):
                    return True
                counter[key] += quantity[q_i]
            return False

        counter = Counter(nums)
        # have to fit m quantity in m largest nums count, not in all nums
        counter_key = sorted(counter, key=lambda x: counter[x])[-len(quantity):]
        # if biggest quantity can not fit in any container, we do not have to check other combination
        quantity.sort(reverse=True)
        return backtrack(0)


"""
You are given an array of n integers, nums, where there are at most 50 unique values in the array. You are also given an array of m customer order quantities, quantity, where quantity[i] is the amount of integers the ith customer ordered. Determine if it is possible to distribute nums such that:

The ith customer gets exactly quantity[i] integers,
The integers the ith customer gets are all equal, and
Every customer is satisfied.
Return true if it is possible to distribute nums according to the above conditions.



Example 1:

Input: nums = [1,2,3,4], quantity = [2]
Output: false
Explanation: The 0th customer cannot be given two different integers.
Example 2:

Input: nums = [1,2,3,3], quantity = [2]
Output: true
Explanation: The 0th customer is given [3,3]. The integers [1,2] are not used.
Example 3:

Input: nums = [1,1,2,2], quantity = [2,2]
Output: true
Explanation: The 0th customer is given [1,1], and the 1st customer is given [2,2].


Constraints:

n == nums.length
1 <= n <= 105
1 <= nums[i] <= 1000
m == quantity.length
1 <= m <= 10
1 <= quantity[i] <= 105
There are at most 50 unique values in nums.
"""