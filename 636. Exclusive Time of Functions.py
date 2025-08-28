from typing import List


class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        stack, result = [], [0]*n # (id, start, penalty)
        for log in logs:
            s_log = log.split(":")
            if s_log[1] == 'end':
                f_id, start, penalty = stack.pop()
                run_time = int(s_log[2]) - start + 1
                if stack:
                    stack[-1][-1] += run_time
                result[f_id] += run_time  - penalty
            else:
                stack.append([int(s_log[0]), int(s_log[2]), 0])
        return result
