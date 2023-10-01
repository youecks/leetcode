###########################
## Linked List Traversal ##
###########################

# Iterative
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

a = Node('A')
b = Node('B')
c = Node('C')
d = Node('D')

a.next = b
b.next = c
c.next = d

# Iterative Traversal
def print_list(head):
    current = head
    while current is not None:
        print(current.val)
        current = current.next

# Recursive Traversal
def print_list(head):
    if head is None:
        return 
    print(head.val)
    print_list(head.next)

print_list(b)


########################
## Linked List Values ##
########################

# Write a function, linked_list_values, that takes in the head of a linked list as an argument. The function should return a list containing all values of the nodes in the linked list.
def linked_list_values(head):
  values = []
  current = head
  while current is not None:
    values.append(current.val)
    current = current.next
  return values