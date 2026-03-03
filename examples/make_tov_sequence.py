#!/usr/bin/env python

import argparse
import numpy as np
from tovpy.eos import EOSTabular
from tovpy.utils import Utils

parser = argparse.ArgumentParser(
    description="Computes a TOV sequence given an EOS table"
)
parser.add_argument("eos", help="EOS table in RNS format")
parser.add_argument(
    "--pmin", default=-12, type=float, help="log of minimum central pressure"
)
parser.add_argument(
    "--pmax", default=-9, type=float, help="log of maximum central pressure"
)
parser.add_argument(
    "--np", default=50, type=int, help="number of central pressure points"
)
parser.add_argument(
    "--backend", default="scipy", choices=["scipy", "numba"],
    help="ODE solver backend (default: scipy)"
)
parser.add_argument("-o", "--output", help="Output filename")
args = parser.parse_args()

eos = EOSTabular("tabular", filename=args.eos)
utils = Utils(eos=eos, p=np.logspace(args.pmin, args.pmax, args.np), path=".",
              ode_backend=args.backend)

leven = [2, 3, 4]  # even-parity multipoles (example values)
lodd  = [2, 3, 4]  # odd-parity multipoles (example values)

utils.Love_txt(leven=leven, lodd=lodd, filename=args.output)
