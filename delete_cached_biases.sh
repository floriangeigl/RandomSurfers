#!/bin/bash
echo "delete"
find /opt/datasets/ -maxdepth 2 -name '*.bias'
find /opt/datasets/ -maxdepth 2 -name '*.bias' | xargs -n1 -I@ rm @
echo "done"
find /opt/datasets/ -maxdepth 2 -name '*.bias'