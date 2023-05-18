class Node:
    def __init__(self, key, next=None):
        self.val, self.next = key, next


class MyHashSet:

    def __init__(self):
        self.load_factor, self.capacity, self.key_count = 0.75, 16, 0
        self.hashset = [None] * self.capacity

    def add(self, key: int) -> None:
        node, h = Node(key), self.hash(key)
        entry = self.hashset[h]
        self.key_count += 1

        if not entry:
            self.hashset[h] = node
        else:
            previous = None
            while entry:
                if entry.val == key:
                    self.key_count -= 1
                    return
                previous = entry
                entry = entry.next
            previous.next = node

        if float(self.key_count) / float(self.capacity) > self.load_factor:
            self.rehash()

    def remove(self, key: int) -> None:
        h = self.hash(key)
        entry = self.hashset[h]

        dummy_head = previous = Node(0, entry)
        while entry:
            if entry.val == key:
                previous.next = entry.next
                self.key_count -= 1
                self.hashset[h] = dummy_head.next
                return

            previous = entry
            entry = entry.next

    def contains(self, key: int) -> bool:
        entry = self.hashset[self.hash(key)]
        while entry:
            if entry.val == key:
                return True
            entry = entry.next
        return False

    def hash(self, key) -> int:  # hash based on key space, index is the key space since we use array aka list
        return key % self.capacity

    def rehash(self) -> None:  # double the capacity
        self.capacity *= 2
        self.key_count, tmp_hashset = 0, self.hashset
        self.hashset = [None] * self.capacity
        for entry in tmp_hashset:
            while entry:
                self.add(entry.val)
                entry = entry.next

if __name__ == '__main__':
# Your MyHashSet object will be instantiated and called as such:

    key = 1
    obj = MyHashSet()
    obj.add(key)
    obj.remove(key)
    param_3 = obj.contains(key)