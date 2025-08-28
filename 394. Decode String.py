class Solution:
    def decodeString(self, s: str) -> str:
        stack, number = [], 0
        for i, c in enumerate(s):
            if c.isdigit():
                number *= 10
                number += ord(c) - ord('0')
            elif c == ']':
                current = ""
                while stack[-1] != '[':
                    current = stack.pop() + current
                stack.pop()
                count = stack.pop()
                stack.append(current * count)
            else:
                if c == '[':
                    stack.append(number)
                    number = 0
                stack.append(c)
        return ''.join(stack)