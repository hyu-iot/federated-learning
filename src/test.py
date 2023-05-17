
def funA(a=1, b=2):
    print(f"a: {a}, b: {b}")

config = {
    'a' : 2,
    'b' : 3,
}

funA(**config)

fa = funA
fa(**config)

config2 = {
    **config,
    'c': 5
}
print(config2)