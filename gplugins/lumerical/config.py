# Used to display Lumerical GUIs, log status, and plot results
DEBUG_LUMERICAL = False

# Enable dopant generation in lbr process file. Set this to False if using Lumerical versions 2021 or prior (Layer Builder
# has issues with dopants for these versions of Lumerical).
ENABLE_DOPING = True

um = 1e-6
cm = 1e-2
marker_list = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "P",
    "*",
    "h",
    "+",
    "X",
    "D",
] * 10
