# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md

all:
	python setup.py build_ext --inplace --user
	python setup.py install --user

clean:
	rm -rf *.o *.a *.so test build
