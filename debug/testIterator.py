
class A:
    def __init__(self):
        self.iter = self.B()

    class B:
        def __init__(self):
            self.a =0

        def __iter__(self):
            return self

        def __next__(self):
            if self.a < 5:
                self.a = self.a +1
                return self.a

            else:
                raise StopIteration()
        def next(self):
            return self.__next__()


    def next_mini(self):
        try:
            batch = self.iter.next()
        except StopIteration:
            self.iter = self.B()
            batch = self.iter.next()

        return batch

if __name__ == '__main__':
    a = A()
    while True:
        mini = a.next_mini()
        print(mini)