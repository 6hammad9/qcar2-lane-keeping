#!/bin/bash
pip3 install --break-system-packages --quiet casadi 2>&1 | tail -3
python3 -c "import casadi; print('casadi OK', casadi.__version__)"
