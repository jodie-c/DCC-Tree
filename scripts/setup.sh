#!/bin/bash

cd $(pwd)

python -m dcc-env ./envs/
source ./envs/dcc-env/bin/activate
pip install -r ./envs/requirements-dcc.txt
cp -f ./envs/numpyro-changes/* ./envs/dcc-env/lib/python3.8/site-packages/numpyro/infer/
deactivate

exit 0
