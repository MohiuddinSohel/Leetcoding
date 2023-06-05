class Solution:
    @staticmethod
    def findIntegers(n: int) -> int:
        def convert_to_binary(n):
            result = []
            while n:
                result.append(str(n & 1))
                n >>= 1
            return ''.join(reversed(result))

        def digit_dp(index, tight, is_previous_one):
            if index == len(nums):
                return 1

            elif dp[index][tight][is_previous_one] == -1:
                limit = int(nums[index]) if tight else 1
                limit = 0 if is_previous_one else limit
                count = 0
                for cur_digit in range(limit + 1):
                    new_tight = tight and cur_digit == int(nums[index])
                    count += digit_dp(index + 1, new_tight, cur_digit == 1)
                dp[index][tight][is_previous_one] = count

            return dp[index][tight][is_previous_one]

        nums = convert_to_binary(n)
        dp = [[[-1 for _ in range(2)] for _ in range(2)] for _ in range(len(nums))]
        return digit_dp(0, True, False)


if __name__ == '__main__':
    n = 13
    print(Solution.findIntegers(n))

"""
Given a positive integer n, return the number of the integers in the range [0, n] whose binary representations do not contain consecutive ones.

 

Example 1:

Input: n = 5
Output: 5
Explanation:
Here are the non-negative integers <= 5 with their corresponding binary representations:
0 : 0
1 : 1
2 : 10
3 : 11
4 : 100
5 : 101
Among them, only integer 3 disobeys the rule (two consecutive ones) and the other 5 satisfy the rule. 
Example 2:

Input: n = 1
Output: 2
Example 3:

Input: n = 2
Output: 3
 

Constraints:

1 <= n <= 109
"""