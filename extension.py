#extemsion.py
def get_ext(fname):
    """

    :param fname:
    :return: the extension of file fname.
    """
    dot=fname.rfind('.')
    if dot==-1: #fname中没有
        return ''
    else:
        return fname[dot+1:]

print(get_ext('hello.text'))
print(get_ext('pizza.py'))
