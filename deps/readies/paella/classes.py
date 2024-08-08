
import functools

#----------------------------------------------------------------------------------------------

def noctor(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        raise TypeError("No default constrcutor for {}".format(self.__class__.__name__))
    return wrapper

#----------------------------------------------------------------------------------------------

class CtorDecorator(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        def ctor(*args, **kwargs):
            x = klass.__new__(klass)
            self.f(x, *args, **kwargs)
            return x
        def bound_ctor(*args, **kwargs):
            self.f(obj, *args, **kwargs)
        return ctor if obj is None else bound_ctor

ctor=CtorDecorator
