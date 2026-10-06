# Algorithms Notes

## Searching

### Linear Search
Checks every element one by one. Time complexity: O(n).

### Binary Search
Works on sorted arrays. Repeatedly divides the search interval in half.
Time complexity: O(log n). Requires the array to be sorted beforehand.

## Sorting

### Quicksort
Average case O(n log n), worst case O(n^2) if pivot selection is poor.
Uses divide-and-conquer with a pivot element.

### Merge Sort
Always O(n log n). Stable sort, but requires O(n) extra space.

## Big-O Cheat Sheet
- O(1): constant time (e.g., array index access)
- O(log n): binary search
- O(n): linear scan
- O(n log n): efficient sorting (merge sort, quicksort average case)
- O(n^2): nested loops, bubble sort
