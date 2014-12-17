#!/usr/bin/env python
import re
import sys

lines = file('foo').readlines()

record = []

header = ["vectorsize as 2^x", "array filltime (microsec)", "accelerate time (microsec)",
          "metal gpu time (microsec)", "cpu time (microsec)", "metal relative to cpu", "metal relative to accelerate"]

print "\t".join(header)

for line in lines:
    if "2014-12" in line: # skip noise
        continue
    res = re.findall(r'((\d+).(\d+))', line)
    res2 = re.findall(r'((\d+)\^(\d+))', line)

    #print ["RES2", res2]
    #print len(res)
    #print res
    if len(res) == 0:
        continue
    elif len(res2) > 0:
        #print "LINE: ", line
        #print "=====>>>>>>", len(header), len(record)
        if len(record) > 0:
            assert len(record) == len(header), [record,header]
            print "\t".join(record)
        record = []
        #print "\n###############################"
        #print "new record: ", res
        expexpression = res2[0][0]
        #print "####", expexpression
        record.append(str(expexpression))
    else:
        #print line, res[0][0]
        record.append(str(res[0][0]))

if len(record) > 0:
    assert len(record) == len(header)
    print "\t".join(record)
