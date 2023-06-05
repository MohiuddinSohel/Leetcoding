class Solution:
    # can be solved with the unique/distinct vertion also by subtracting from total
    @staticmethod
    def numDupDigitsAtMostN(n: int) -> int:
        def digit_dp(index, tight, repeated, mask):
            if index == len(nums):
                return repeated

            elif dp[index][tight][repeated][mask] == -1:
                count, limit = 0, int(nums[index]) if tight else 9

                for current_digit in range(limit + 1):
                    new_repeated = repeated or (mask & (1 << current_digit)) != 0
                    new_tight = tight and int(nums[index]) == current_digit
                    new_mask = 0 if current_digit == 0 and mask == 0 else (mask | (1 << current_digit))
                    count += digit_dp(index + 1, new_tight, new_repeated, new_mask)
                dp[index][tight][repeated][mask] = count

            return dp[index][tight][repeated][mask]

        nums = str(n)
        dp = [[[[-1 for _ in range(2 ** 10)] for _ in range(2)] for _ in range(2)] for _ in range(len(nums))]
        return digit_dp(0, True, False, 0)


if __name__ == '__main__':
    n = 1000
    print(Solution.numDupDigitsAtMostN(n))