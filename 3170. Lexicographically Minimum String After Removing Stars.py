import heapq


class Solution:
    def clearStars(self, s: str) -> str:
        heap, removal  = [], set()
        for i, c in enumerate(s):
            if c == '*':
                _, index = heapq.heappop(heap)
                removal.add(-index)
            else:
                heapq.heappush(heap, (c, -i))
        result = []
        for i, c in enumerate(s):
            if c == '*' or i in removal:
                continue
            result.append(c)
        return "".join(result)

