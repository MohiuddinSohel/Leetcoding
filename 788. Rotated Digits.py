class Solution:
    @staticmethod
    def rotatedDigits(n: int) -> int:
        def digit_dp(index, tight, changed):
            # for 0, changed will be False, so no care for 0
            if index == len(nums):
                return changed

            elif dp[index][tight][changed] == -1:
                limit = int(nums[index]) if tight else 9
                count = 0

                for current_digit in range(limit + 1):
                    if current_digit not in rotation_map:
                        continue
                    new_changed = True if (changed or rotation_map[current_digit] != current_digit) else False
                    new_tight = True if (tight and int(nums[index]) == current_digit) else False
                    count += digit_dp(index + 1, new_tight, new_changed)
                dp[index][tight][changed] = count

            return dp[index][tight][changed]

        rotation_map = {0: 0, 1: 1, 8: 8, 2: 5, 5: 2, 6: 9, 9: 6}
        nums = str(n)
        dp = [[[-1 for _ in range(2)] for _ in range(2)] for _ in range(len(nums))]
        return digit_dp(0, True, False)


if __name__ == '__main__':
    n = 10
    print(Solution.rotatedDigits(n))

"""
An integer x is a good if after rotating each digit individually by 180 degrees, we get a valid number that is different from x. Each digit must be rotated - we cannot choose to leave it alone.

A number is valid if each digit remains a digit after rotation. For example:

0, 1, and 8 rotate to themselves,
2 and 5 rotate to each other (in this case they are rotated in a different direction, in other words, 2 or 5 gets mirrored),
6 and 9 rotate to each other, and
the rest of the numbers do not rotate to any other number and become invalid.
Given an integer n, return the number of good integers in the range [1, n].

 

Example 1:

Input: n = 10
Output: 4
Explanation: There are four good numbers in the range [1, 10] : 2, 5, 6, 9.
Note that 1 and 10 are not good numbers, since they remain unchanged after rotating.
Example 2:

Input: n = 1
Output: 0
Example 3:

Input: n = 2
Output: 1
 

Constraints:

1 <= n <= 104
"""