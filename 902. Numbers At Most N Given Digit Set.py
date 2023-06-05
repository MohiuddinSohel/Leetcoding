from typing import List


class Solution:
    @staticmethod
    def atMostNGivenDigitSet(digits: List[str], n: int) -> int:
        def digit_dp(index, tight, is_leading_zero):
            if index == len(nums):
                return not is_leading_zero

            elif dp[index][tight][is_leading_zero] == -1:
                count = 0
                # can put single or multiple 0 only at beginning of number
                # if we put 0 at the beginning, the tight variable will be False.
                # New number will be less than n since new number will have less digit than n
                if is_leading_zero:
                    count += digit_dp(index + 1, False, is_leading_zero)

                for current_digit in digits:
                    if tight and int(current_digit) > int(nums[index]):
                        break
                    new_tight = tight and int(nums[index]) == int(current_digit)
                    count += digit_dp(index + 1, new_tight, False)
                dp[index][tight][is_leading_zero] = count

            return dp[index][tight][is_leading_zero]

        nums = str(n)
        dp = [[[-1 for _ in range(2)] for _ in range(2)] for _ in range(len(nums))]
        return digit_dp(0, True, True)


if __name__ == '__main__':
    digits, n = ["1","4","9"], 1000000000
    print(Solution.atMostNGivenDigitSet(digits, n))

"""
Given an array of digits which is sorted in non-decreasing order. You can write numbers using each digits[i] as many times as we want. For example, if digits = ['1','3','5'], we may write numbers such as '13', '551', and '1351315'.

Return the number of positive integers that can be generated that are less than or equal to a given integer n.

 

Example 1:

Input: digits = ["1","3","5","7"], n = 100
Output: 20
Explanation: 
The 20 numbers that can be written are:
1, 3, 5, 7, 11, 13, 15, 17, 31, 33, 35, 37, 51, 53, 55, 57, 71, 73, 75, 77.
Example 2:

Input: digits = ["1","4","9"], n = 1000000000
Output: 29523
Explanation: 
We can write 3 one digit numbers, 9 two digit numbers, 27 three digit numbers,
81 four digit numbers, 243 five digit numbers, 729 six digit numbers,
2187 seven digit numbers, 6561 eight digit numbers, and 19683 nine digit numbers.
In total, this is 29523 integers that can be written using the digits array.
Example 3:

Input: digits = ["7"], n = 8
Output: 1
 

Constraints:

1 <= digits.length <= 9
digits[i].length == 1
digits[i] is a digit from '1' to '9'.
All the values in digits are unique.
digits is sorted in non-decreasing order.
1 <= n <= 109
"""