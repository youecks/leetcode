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

# Iterative
def linked_list_values(head):
  values = []
  current = head
  while current is not None:
    values.append(current.val)
    current = current.next
  return values

# Recursive

def linked_list_values(head):
  values = []
  _linked_list_values(head, values)
  return values

def _linked_list_values(head, values):
  if head is None:
    return
  values.append(head.val)
  _linked_list_values(head.next, values)


##############
## Sum List ##
##############

# Iterative
class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

def sum_list(head):
  
  current = head
  total_sum = 0
  
  while current is not None:
    total_sum += current.val
    current = current.next 
  
  
  return total_sum

# n = number of nodes
# Time: O(n)
# Space: O(1)


# Recursive
def sum_list(head):
  if head is None:
    return 0
  return head.val + sum_list(head.next)

# n = number of nodes
# Time: O(n)
# Space: O(n)


######################
## Linked List Find ##
######################

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None

a = Node("a")
b = Node("b")
c = Node("c")
d = Node("d")

a.next = b
b.next = c
c.next = d

# Iterative
def linked_list_find(head, target):
  
    current = head
  
    while current is not None:
        if current.val == target:
            return True 
        current = current.next 
    return False

# n = number of nodes
# Time: O(n)
# Space: O(1)

# Recursive

def linked_list_find(head, target):
  if head is None:
    return False
  if head.val == target:
    return True
  return linked_list_find(head.next, target)

# n = number of nodes
# Time: O(n)
# Space: O(n)

####################
## Get Node Value ##
####################

# Recursive

class Node:
   def __init__(self, val):
      self.val = val
      self.next = None

def get_node_value(head, index):
  if head is None:
    return None
  if index == 0:
     return head.val
  
  return get_node_value(head.next, index - 1)

# Iterative

class Node:
  def __init__(self, val):
    self.val = val
    self.next = None
    
def get_node_value(head, index):
  current = head
  count = 0
  
  while current is not None:
    if count == index:
      return current.val
    count += 1
    current = current.next
  return None
    





##################
## Reverse List ##
##################

# Iterative
class Node:
  def __init__(self, val):
    self.val = val
    self.next = None
    
def reverse_list(head):
  prev = None 
  current = head
  while current is not None:
      next = current.next
      current.next = prev
      prev = current
      current = next
  return prev

# Recursive

def reverse_list(head, prev = None):
   if head is None:
      return prev
   next = head.next
   head.next = prev
   return reverse_list(next, head)

#################
## Zipper List ##
#################

# Iterative

def zipper_lists(head_1, head_2):
  tail = head_1
  current_1 = head_1.next
  current_2 = head_2
  count = 0
  while current_1 is not None and current_2 is not None:
    if count % 2 == 0:
      tail.next = current_2
      current_2 = current_2.next
    else:
      tail.next = current_1
      current_1 = current_1.next
    tail = tail.next
    count += 1
    
  if current_1 is not None:
    tail.next = current_1
  if current_2 is not None:
    tail.next = current_2
    
  return head_1

# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(1)

# Recursive

def zipper_lists(head_1, head_2):
  if head_1 is None and head_2 is None:
    return None
  if head_1 is None:
    return head_2
  if head_2 is None:
    return head_1
  next_1 = head_1.next
  next_2 = head_2.next
  head_1.next = head_2
  head_2.next = zipper_lists(next_1, next_2)
  return head_1  

# n = length of list 1
# m = length of list 2
# Time: O(min(n, m))
# Space: O(min(n, m))