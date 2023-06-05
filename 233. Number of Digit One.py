class Solution:
    @staticmethod
    def countDigitOne(n: int) -> int:

        # if two numbers prefix are same, they count of 1 in both prefix needs to be counted seperately
        # thus return count after forming the whole number
        def digitDP(index, tight, count):
            if index == len(nums):
                return count

            elif dp[index][tight][count] == -1:
                current_count, limit = 0, (int(nums[index]) + 1) if tight else 10
                for digit in range(limit):
                    new_tight = (tight and digit == int(nums[index]))
                    current_count += digitDP(index + 1, new_tight, count + (digit == 1))
                dp[index][tight][count] = current_count

            return dp[index][tight][count]

        nums = str(n)
        dp = [[[-1 for _ in range(len(nums) + 1)] for _ in range(2)] for _ in range(len(nums))]
        return digitDP(0, True, 0)


if __name__ == '__main__':
    n = 13
    print(Solution.countDigitOne(n))


"""
Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.

 

Example 1:

Input: n = 13
Output: 6
Example 2:

Input: n = 0
Output: 0
 

Constraints:

0 <= n <= 109
"""