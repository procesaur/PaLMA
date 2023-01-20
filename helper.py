from os import name, remove


def limit(x, min, max):
    if isinstance(x, str):
        if len(x) > max:
            x = x[:max]
    else:
        if x < min:
            x = min
        if x > max:
            x = max
    return x


def isWindows():
    return name == 'nt'


def trydel(file):
    try:
        remove(file)
    except:
        pass


def perp2prob(x):
    if x > 0:
        return 1/x
    else:
        return 1
