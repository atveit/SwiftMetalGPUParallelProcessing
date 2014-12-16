SwiftMetalGPUParallelProcessing
===============================

Data Parallel Processing with Swift and Metal on GPU for iOS8 (and beyond)

Metal is an alternative to OpenGL for graphics processing, but for general data-parallel programming for GPUs it is an alternative to OpenCL and Cuda. This (simple) example shows how to use Metal with Swift for calculating the Sigmoid function (Sigmoid function is frequently occurring in machine learning settings, e.g. for Deep Learning and Kernel Methods/Support Vector Machines).

If you want to read up on Metal I recommend having a look at https://developer.apple.com/metal/ (Metal Programming Guide, Metal Shading Language and Metal Framework Reference)

See http://memkite.com/blog/2014/12/15/data-parallel-programming-with-metal-and-swift-for-iphoneipad-gpu/ for a blog post describing this code.

The code is in the ViewController.swift and Shaders.metal - direct links:

https://github.com/atveit/SwiftMetalGPUParallelProcessing/blob/master/SwiftMetalProcessing/ViewController.swift

https://github.com/atveit/SwiftMetalGPUParallelProcessing/blob/master/SwiftMetalProcessing/Shaders.metal
