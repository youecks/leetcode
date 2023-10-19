def minimum_island(grid):
  visited = set()
  min_size = float("inf")
  for r in range(len(grid)):
    for c in range(len(grid[0])):
      size = explore_size(grid, r, c, visited)
      if size > 0 and size < min_size:
        min_size = size
  return min_size

def explore_size(grid, r, c, visited):
  row_inbounds = 0 <= r < len(grid)
  col_inbounds = 0 <= c < len(grid[0])
  if not row_inbounds or not col_inbounds:
    return 0
  
  if grid[r][c] == 'W':
    return 0
  
  pos = (r, c)
  if pos in visited:
    return 0
  visited.add(pos)
  
  size = 1
  size += explore_size(grid, r - 1, c, visited)
  size += explore_size(grid, r + 1, c, visited)  
  size += explore_size(grid, r, c - 1, visited)
  size += explore_size(grid, r, c + 1, visited)
  return size

#r = number of rows
# c = number of columns
# Time: O(rc)
# Space: O(rc)