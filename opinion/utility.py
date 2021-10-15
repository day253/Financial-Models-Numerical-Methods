#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time


def GetRunTime(func):
    def call_func(*args, **kwargs):
        begin_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        Run_time = end_time - begin_time
        print(str(func.__name__)+" Total_Cost: "+str(Run_time))
        return ret
    return call_func
