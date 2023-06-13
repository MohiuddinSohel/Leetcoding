import heapq
from typing import List
from collections import deque


class Solution:
    def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
        def dynamic_p():
            dp, max_sum = nums[:], max(nums)
            for i in range(len(nums)):
                for j in range(max(0, i - k), i):
                    dp[i] = max(dp[i], dp[j] + nums[i])
                max_sum = max(max_sum, dp[i])
            return max_sum

        def seq_sum_with_priority_queue():
            max_heap, max_sum = [], max(nums)

            for i, num in enumerate(nums):
                current_seq_sum = num
                if max_heap:
                    current_seq_sum = max(current_seq_sum, -max_heap[0][0] + num)
                max_sum = max(max_sum, current_seq_sum)

                heapq.heappush(max_heap, (-current_seq_sum, i))

                # remove the max sum if it exceeds the k limit
                while max_heap and max_heap[0][1] <= i - k:
                    heapq.heappop(max_heap)

            return max_sum

        def monotonically_decreasing_seq_sum():
            decreasing_queue, maxSum = deque(), max(nums)

            for i, num in enumerate(nums):
                current_seq_sum = num
                if decreasing_queue:
                    current_seq_sum = max(decreasing_queue[0][1] + num, current_seq_sum)
                maxSum = max(maxSum, current_seq_sum)

                # maintain monotonically decreasing queue such that max seq sum is at index 0
                while decreasing_queue and current_seq_sum > decreasing_queue[-1][1]:
                    decreasing_queue.pop()

                decreasing_queue.append((i, current_seq_sum))

                # remove all maximum outside window subsequence sum from consideration,
                # decreasing_queue[0][0] <= i - k, we are checking for every iteration so equality check is enough
                if decreasing_queue and decreasing_queue[0][0] == i - k:
                    decreasing_queue.popleft()

            return maxSum

        return seq_sum_with_priority_queue()
        return monotonically_decreasing_seq_sum()


if __name__ == '__main__':
    nums, k = [10,-2,-10,-5,20], 2
    print(Solution().constrainedSubsetSum(nums, k))