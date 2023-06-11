class Solution:
    @staticmethod
    def maxValue(n: int, index: int, maxSum: int) -> int:
        def is_possible_greedy(current):
            c_sum = 0
            # number of element, total = index + 1
            if index + 1 >= current:
                c_sum = ( index + 1 - current) * 1 + current * (current + 1) / 2
            else:
                # (1/2)*n(2a + (n-1) *d)
                # number of element, total = index + 1
                # first element of sequence, a = current - (index + 1) + 1 = current - total + 1
                c_sum = (1/2) * (index + 1) * (2 * (current - (index + 1) + 1) + (index + 1 - 1) * 1)

            # number of element = n - index
            if n - index >= current:
                c_sum += (n - index - current) * 1 + current * (current + 1) / 2
            else:
                # (1/2)*n(2a + (n-1) *d)
                # number of element, total = n - index
                # first element of sequence, a = current - (n - index ) + 1 = current - total + 1
                c_sum += (1/2) * (n - index) * (2 * (current - (n - index ) + 1) + (n - index - 1) * 1)

            c_sum -= current # counted curent 2 times, so minus 1 times
            return c_sum <= maxSum

        def binary_search():
            l, h = 1, maxSum
            while l <= h:
                m = l + (h - l) // 2
                if is_possible_greedy(m):
                    l = m + 1
                else:
                    h = m - 1
            return h

        return binary_search()


if __name__ == '__main__':
    n, index, maxSum = 6, 1, 10
    print(Solution.maxValue(n, index, maxSum ))


"""
You are given three positive integers: n, index, and maxSum. You want to construct an array nums (0-indexed) that satisfies the following conditions:

nums.length == n
nums[i] is a positive integer where 0 <= i < n.
abs(nums[i] - nums[i+1]) <= 1 where 0 <= i < n-1.
The sum of all the elements of nums does not exceed maxSum.
nums[index] is maximized.
Return nums[index] of the constructed array.

Note that abs(x) equals x if x >= 0, and -x otherwise.

 

Example 1:

Input: n = 4, index = 2,  maxSum = 6
Output: 2
Explanation: nums = [1,2,2,1] is one array that satisfies all the conditions.
There are no arrays that satisfy all the conditions and have nums[2] == 3, so 2 is the maximum nums[2].
Example 2:

Input: n = 6, index = 1,  maxSum = 10
Output: 3
 

Constraints:

1 <= n <= maxSum <= 109
0 <= index < n
"""