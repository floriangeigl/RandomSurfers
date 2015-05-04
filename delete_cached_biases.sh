#!/bin/bash
find /opt/datasets/ -maxdepth 2 -name '*.bias' | xargs -n1 -I@ rm @
find /opt/datasets/ -maxdepth 2 -name '*.bias'