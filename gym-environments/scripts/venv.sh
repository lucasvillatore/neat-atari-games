#!/bin/bash

python -m venv venv
. venv/bin/activate

pip install -e .
pip install -r requirements.txt