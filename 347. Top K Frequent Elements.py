from typing import List
from collections import Counter, defaultdict
import heapq
import random


class Solution:
    def heap_sort(self, nums, k):
        counter = Counter(nums)
        return heapq.nlargest(k, counter.keys(), key=lambda x: counter[x])

    def bucket_sort(self, nums, k):
        counter, bucket = Counter(nums), defaultdict(list)

        for key in counter:
            bucket[counter[key]].append(key)

        result = []
        for length in range(len(nums), 0, -1):
            if k <= 0:
                break
            if length in bucket:
                k -= len(bucket[length])
                result.extend(bucket[length])
        return result

    def quick_select(self, nums: List[int], k: int) -> List[int]:
        counter = Counter(nums)
        arr = list(counter.keys())
        index = self.quickSelect(arr, counter, 0, len(arr) - 1, len(arr) - k)
        return arr[index:]

    def partition(self, nums, freq, l, r, pivotIndex):
        p = l

        pivotV = freq[nums[pivotIndex]]
        nums[r], nums[pivotIndex] = nums[pivotIndex], nums[r]

        while l < r:
            if freq[nums[l]] < pivotV:
                nums[l], nums[p] = nums[p], nums[l]
                p += 1
            l += 1
        nums[p], nums[r] = nums[r], nums[p]
        return p

    def quickSelect(self, nums, freq, l, r, k):
        if l == r:
            return l

        pivot = random.randint(l, r)
        pivot_index = self.partition(nums, freq, l, r, pivot)
        if k == pivot_index:
            return pivot_index
        elif k < pivot_index:
            return self.quickSelect(nums, freq, l, pivot_index - 1, k)
        else:
            return self.quickSelect(nums, freq, pivot_index + 1, r, k)


if __name__ == '__main__':
    nums, k = [1, 1, 1, 2, 2, 3], 2
    print(Solution().quick_select(nums, k))
    print(Solution().heap_sort(nums, k))
    print(Solution().bucket_sort(nums, k))
