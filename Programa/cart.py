"""Modelo simples de carrinho de compras."""

from config import PRODUCTS


class CartItem:
    def __init__(self, class_name):
        info = PRODUCTS.get(class_name, {"display_name": class_name, "price": 0.0})
        self.class_name = class_name
        self.display_name = info["display_name"]
        self.price = info["price"]
        self.qty = 0

    @property
    def subtotal(self):
        return self.qty * self.price


class Cart:
    def __init__(self):
        self.items = {}  # class_name -> CartItem (mantem ordem de insercao)

    def add(self, class_name):
        if class_name not in PRODUCTS:
            return None
        if class_name not in self.items:
            self.items[class_name] = CartItem(class_name)
        self.items[class_name].qty += 1
        return self.items[class_name]

    def remove(self, class_name):
        self.items.pop(class_name, None)

    @property
    def total(self):
        return sum(item.subtotal for item in self.items.values())

    def clear(self):
        self.items.clear()
