# import IPython.nbformat.current as nbf
# nb = nbf.read(open('main.py', 'r'), 'py')
# nbf.write(nb, open('main.ipynb', 'w'), 'ipynb')

from IPython.nbformat import v3, v4

with open("main.py") as fpin:
    text = fpin.read()

nbook = v3.reads_py(text)
nbook = v4.upgrade(nbook)  # Upgrade v3 to v4

jsonform = v4.writes(nbook) + "\n"
with open("main.ipynb", "w") as fpout:
    fpout.write(jsonform)