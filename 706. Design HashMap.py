class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.capacity, self.entryCount, self.loadFactor = 16, 0, 0.75
        self.hashMap = [None] * self.capacity

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        h = self.hash(key)
        node, previous = self.hashMap[h], None
        self.entryCount += 1

        if not node:
            self.hashMap[h] = Node(key, value)
        else:
            while node:
                if node.key == key:
                    node.val = value
                    self.entryCount -= 1
                    return
                previous = node
                node = node.next
            previous.next = Node(key, value)

        if float(self.entryCount) / float(self.capacity) > self.loadFactor:
            self.rehash()

    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        node = self.hashMap[self.hash(key)]
        while node:
            if node.key == key:
                return node.val
            node = node.next
        return -1

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        hashKey = self.hash(key)
        node, previous = self.hashMap[hashKey], None
        while node:
            if node.key == key:
                if previous:
                    previous.next = node.next
                else:
                    self.hashMap[hashKey] = node.next
                self.entryCount -= 1
                return
            previous = node
            node = node.next

    def hash(self, key):
        return key % self.capacity

    def rehash(self):
        self.capacity, prehMap, self.entryCount = self.capacity * 2, self.hashMap, 0
        self.hashMap = [None] * self.capacity

        for mEntry in prehMap:
            while mEntry:
                self.put(mEntry.key, mEntry.val)
                mEntry = mEntry.next


class Node:
    def __init__(self, k, v, next=None):
        self.key = k
        self.val = v
        self.next = next

if __name__ == '__main__':
# Your MyHashMap object will be instantiated and called as such:
    obj = MyHashMap()
    obj.put(3,9)
    param_2 = obj.get(3)
    obj.remove(3)