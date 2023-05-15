class Solution:
    # order of operation matters here
    def maxScore(self, nums) -> int:
        def gcd(n1, n2):
            while n1:
                temp = n1
                n1 = n2 % n1
                n2 = temp
            return n2

        def dfs(mask, operation):
            if (operation << 1) == len(nums):
                return 0
            elif dp[mask] == -1:
                for index1 in range(len(nums) - 1):
                    if 1 << index1 & mask:
                        continue
                    for index2 in range(index1 + 1, len(nums)):
                        if (1 << index2) & mask:
                            continue
                        new_mask = mask | (1 << index1) | (1 << index2)
                        dp[mask] = max(dp[mask],
                                       (operation + 1) * gcd(nums[index1], nums[index2]) + dfs(new_mask, operation + 1))

            return dp[mask]

        dp = [-1] * (2 ** (len(nums)))
        return dfs(0, 0)


if __name__ == '__main__':
    nums = [171651, 546244, 880754, 412358]
    sol = Solution()
    print(sol.maxScore(nums))


'''
You are given nums, an array of positive integers of size 2 * n. You must perform n operations on this array.

In the ith operation (1-indexed), you will:

Choose two elements, x and y.
Receive a score of i * gcd(x, y).
Remove x and y from nums.
Return the maximum score you can receive after performing n operations.

The function gcd(x, y) is the greatest common divisor of x and y.
'''
