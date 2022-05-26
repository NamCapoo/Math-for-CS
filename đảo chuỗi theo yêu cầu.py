from msilib.schema import SelfReg
from typing_extensions import Self


class Node:
    def __init__(self, data):
	    self.data = data;
        SelfReg.next = None;
class Solution:
    def reverse(self,head, k):
        current = head;
        next = None;
        prev = None;
        count = 0;
        while (current is not None and count < k):
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
            count +=1;
        if (next is not None):
            head.next = self.reverse(next,k)
        return prev