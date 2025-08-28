from typing import List


class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        line = [0] * 102
        for num in nums:
            line[num[0]] += 1
            line[num[1] + 1] -= 1

        current, count = 0, 0
        for i in range(102):
            current += line[i]
            if current > 0:
                count += 1
        return count