#Định nghĩa các lớp cơ bản cho môi trường đóng gói (Container Loading Problem - CLP)
import random

class Item:
    def __init__(self, id, l, w, h):
        self.id = id
        self.l = l
        self.w = w
        self.h = h
        self.x = 0
        self.y = 0
        self.z = 0
        self.is_packed = False
        
        self.color = f'rgb({random.randint(50,255)}, {random.randint(50,255)}, {random.randint(50,255)})'

    @property
    def volume(self):
        return self.l * self.w * self.h

class Container:
    def __init__(self, length, width, height):
        self.L = length
        self.W = width
        self.H = height
        self.packed_items = []

    def add_item(self, item):
        self.packed_items.append(item)

    @property
    def packed_volume(self):
        return sum(item.volume for item in self.packed_items)