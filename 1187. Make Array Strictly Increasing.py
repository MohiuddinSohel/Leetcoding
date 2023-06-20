import math
from collections import defaultdict
from typing import List


class Solution:
    def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
        def next_greater(target):
            l, r = 0, len(arr2) - 1
            while l <= r:
                m = l + (r - l) // 2
                if arr2[m] <= target:
                    l = m + 1
                else:
                    r = m - 1
            return l

        def iterative1D(arr1, arr2):
            # narrative changed from recursive to dp,
            # if we put arr1[i] or arr2[some] at index i based on previous number which is in dp[i-1]
            len_arr1, len_arr2, int_max = len(arr1), len(arr2), math.inf
            dp = [None] * 2
            arr2.sort()
            dp[0] = {-1: 0}
            for index in range(1, len_arr1 + 1):
                dp[index & 1] = defaultdict(lambda: float('inf'))
                for previous in dp[1 - (index & 1)]:
                    if previous < arr1[index - 1]:  # not replace
                        dp[index & 1][arr1[index - 1]] = min(dp[index & 1][arr1[index - 1]],
                                                             dp[1 - (index & 1)][previous])

                    next_ind = next_greater(previous)

                    if next_ind < len_arr2:  # replace
                        dp[index & 1][arr2[next_ind]] = min(dp[index & 1][arr2[next_ind]],
                                                            1 + dp[1 - (index & 1)][previous])
            return min(dp[index & 1].values()) if dp[index & 1] else -1

        return iterative1D(arr1, arr2)

        def iterative(arr1, arr2):
            len_arr1, len_arr2, int_max = len(arr1), len(arr2), math.inf
            dp = [None] * (len(arr1) + 1)
            arr2.sort()
            dp[0] = {-1: 0}
            for index in range(1, len_arr1 + 1):
                dp[index] = defaultdict(lambda: float('inf'))
                for previous in dp[index - 1]:
                    if previous < arr1[index - 1]:  # not replace
                        dp[index][arr1[index - 1]] = min(dp[index][arr1[index - 1]], dp[index - 1][previous])

                    next_ind = next_greater(previous)

                    if next_ind < len_arr2:  # replace
                        dp[index][arr2[next_ind]] = min(dp[index][arr2[next_ind]], 1 + dp[index - 1][previous])
            return min(dp[-1].values()) if dp[-1] else -1

        return iterative(arr1, arr2)

        def dfs(current_index, previous):
            if current_index == len_arr1:
                return 0
            elif (current_index, previous) not in dp:
                dp[(current_index, previous)] = int_max

                if previous < arr1[current_index]:  # not replace
                    dp[(current_index, previous)] = dfs(current_index + 1, arr1[current_index])

                next_ind = next_greater(previous)

                if next_ind < len_arr2:  # replace
                    dp[(current_index, previous)] = min(dp[(current_index, previous)],
                                                        1 + dfs(current_index + 1, arr2[next_ind]))

            return dp[(current_index, previous)]

        len_arr1, len_arr2, int_max, dp = len(arr1), len(arr2), math.inf, defaultdict(int)
        arr2.sort()
        # dp = [[-1 for _ in range(max(max(arr1), max(arr2)) + 1)] for _ in range(len_arr1)] # Memory limit exceed
        ret = dfs(0, -1)
        return ret if ret != int_max else -1


if __name__ == '__main__':
    arr1 = [1, 5, 3, 6, 7]
    arr2 = [4, 3, 1]
    print(Solution().makeArrayIncreasing(arr1, arr2))