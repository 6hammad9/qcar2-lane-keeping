#!/bin/bash
# Compatibility alias. The supervised stop already escalates INT -> TERM ->
# KILL for each recorded process group and sends a zero command first.
exec bash "$(cd "$(dirname "$0")" && pwd)/stop_all.sh"
