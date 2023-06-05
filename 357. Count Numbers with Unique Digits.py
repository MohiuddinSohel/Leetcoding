class Solution:
    @staticmethod
    def countNumbersWithUniqueDigits(n: int) -> int:
        def digit_dp(index, tight, mask):
            if index == len(nums):
                return 1

            elif dp[index][tight][mask] == -1:
                count = 0
                limit = int(nums[index]) if tight else 9

                for current_digit in range(limit + 1):
                    if mask & (1 << current_digit):  # visited
                        continue
                    new_mask = 0 if (current_digit == 0 and mask == 0) else (mask | (1 << current_digit))
                    new_tight = True if (tight and int(nums[index]) == current_digit) else False
                    count += digit_dp(index + 1, new_tight, new_mask)

                dp[index][tight][mask] = count
            return dp[index][tight][mask]

        nums = str((10 ** n) - 1)
        dp = [[[-1 for _ in range(2 ** 10)] for _ in range(2)] for _ in range(len(nums))]
        return digit_dp(0, True, 0)


if __name__ == '__main__':
    n = 2
    print(Solution.countNumbersWithUniqueDigits(n))

"""
Given an integer n, return the count of all numbers with unique digits, x, where 0 <= x < 10^n.

 

Example 1:

Input: n = 2
Output: 91
Explanation: The answer should be the total numbers in the range of 0 ≤ x < 100, excluding 11,22,33,44,55,66,77,88,99
Example 2:

Input: n = 0
Output: 1
 

Constraints:

0 <= n <= 8
"""