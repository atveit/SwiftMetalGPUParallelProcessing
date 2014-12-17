//
//  ViewController.Swift
//  SwiftMetalProcessing
//
//  Created by Amund Tveit on 15/12/14.
//  Copyright (c) 2014 Amund Tveit. All rights reserved.
//
import UIKit
import Metal
import QuartzCore
import Darwin
import Accelerate

class ViewController: UIViewController {
    
    
    override func viewDidLoad() {
        super.viewDidLoad()

        
        for(var i = 5; i<23; ++i) {
            
            let start0 = CACurrentMediaTime()

            let maxcount = Int(pow(2.0,Float(i)))
            println("#############################################")
            println("==> count = \(maxcount) - 2^\(i)")
            // prepare original input data – a Swift array
            var myvector = [Float](count: maxcount, repeatedValue: 0)
            for (index, value) in enumerate(myvector) {
                myvector[index] = Float(index)
            }
            
            let stop0 = CACurrentMediaTime()
            let delta0 = (stop0-start0)*1000000.0
            println("filling array took \(delta0) microseconds")
            
            var mynegativeVector = [Float](count: maxcount, repeatedValue: 0)
            for(index, value) in enumerate(mynegativeVector) {
                mynegativeVector[index] = Float(-index)
            }
            
            // calculate exp(-x)
            var expMinusX = [Float](count: maxcount, repeatedValue:0)
            var oneVec = [Float](count:maxcount, repeatedValue:1.0)
            var negOneVec = [Float](count:maxcount, repeatedValue:-1.0)
            
            //println(oneVec)
            //println(negOneVec)
            // oneVec contains 1+exp(-x)
            var finalResultVector = [Float](count:maxcount, repeatedValue:0)
            var localcount:Int32 = Int32(mynegativeVector.count)
            
            let start5 = CACurrentMediaTime()
            
            // calculation
            vvexpf(&expMinusX, &mynegativeVector, &localcount)
            cblas_saxpy(Int32(oneVec.count), 1.0, &expMinusX, 1, &oneVec, 1)
            vvpowf(&finalResultVector, &negOneVec, &oneVec, &localcount)
            assert(finalResultVector[0] == 0.5)

            let stop5 = CACurrentMediaTime()
            let delta5 = (stop0-start0)*1000000.0
            println("Accelerate approach took \(delta5) microseconds")
            
            // initialize Metal
            
            // START BENCHMARK
//            let start = CACurrentMediaTime()

            
            var (device, commandQueue, defaultLibrary, commandBuffer, computeCommandEncoder) = initMetal()
            
            
            // set up a compute pipeline with Sigmoid function and add it to encoder
            let sigmoidProgram = defaultLibrary.newFunctionWithName("sigmoid")
            var pipelineErrors = NSErrorPointer()
            var computePipelineFilter = device.newComputePipelineStateWithFunction(sigmoidProgram!, error: pipelineErrors)
            computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
            
            
            computeCommandEncoder.setComputePipelineState(computePipelineFilter!)
            
            
            // calculate byte length of input data - myvector
            var myvectorByteLength = myvector.count*sizeofValue(myvector[0])
            
            
            // create a MTLBuffer - input data that the GPU and Metal and produce
            var inVectorBuffer = device.newBufferWithBytes(&myvector, length: myvectorByteLength, options: nil)
            
            //   set the input vector for the Sigmoid() function, e.g. inVector
            //    atIndex: 0 here corresponds to buffer(0) in the Sigmoid function
            computeCommandEncoder.setBuffer(inVectorBuffer, offset: 0, atIndex: 0)
            
            // d. create the output vector for the Sigmoid() function, e.g. outVector
            //    atIndex: 1 here corresponds to buffer(1) in the Sigmoid function
            var resultdata = [Float](count:myvector.count, repeatedValue: 0)
            var outVectorBuffer = device.newBufferWithBytes(&resultdata, length: myvectorByteLength, options: nil)
            computeCommandEncoder.setBuffer(outVectorBuffer, offset: 0, atIndex: 1)
            
            // hardcoded to 32 for now (recommendation: read about threadExecutionWidth)
            var threadsPerGroup = MTLSize(width:32,height:1,depth:1)
            var numThreadgroups = MTLSize(width:(myvector.count+31)/32, height:1, depth:1)
            computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
            
            computeCommandEncoder.endEncoding()
            
            //        let start = CACurrentMediaTime()
            
            let start = CACurrentMediaTime()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            let stop = CACurrentMediaTime()

            
            //        let stop = CACurrentMediaTime()
            //        let deltaMicroseconds = (stop-start) * (1.0*10e6)
            //        println("cold GPU: runtime in microsecs : \(deltaMicroseconds)")
            
            
            // a. Get GPU data
            // outVectorBuffer.contents() returns UnsafeMutablePointer roughly equivalent to char* in C
            var data = NSData(bytesNoCopy: outVectorBuffer.contents(),
                length: myvector.count*sizeof(Float), freeWhenDone: false)
            // b. prepare Swift array large enough to receive data from GPU
            var finalResultArray = [Float](count: myvector.count, repeatedValue: 0)
            
            // c. get data from GPU into Swift array
            data.getBytes(&finalResultArray, length:myvector.count * sizeof(Float))
            assert(finalResultVector[0] == 0.5)
            
            // STOP BENCHMARK
            
            let deltaMicroseconds = (stop-start) * (1.0*10e6)
            println("cold GPU: runtime in microsecs : \(deltaMicroseconds)")
            
            let start3 = CACurrentMediaTime()
            
            // timing without
            /*
            for (index, value) in enumerate(myvector) {
                finalResultArray[index] = 1.0 / (1.0 + exp(-myvector[index]))
            }
*/
            var fra = NSMutableArray(capacity: myvector.count)
            let ccount = myvector.count
            for j in 0..<ccount {
                fra[j] = 1.0/(1.0 + exp(-myvector[j]))
            }
            
            let stop3 = CACurrentMediaTime()
            let deltaMicroseconds3 = (stop3-start3) * (1.0*10e6)
            println("CPU: runtime in microsecs : \(deltaMicroseconds3)")
            
            let relativeSpeed = deltaMicroseconds3/deltaMicroseconds
            println("Metal was \(relativeSpeed) times faster than CPU")
            
            let relativeToAccelerate = delta5/deltaMicroseconds
            println("Metal was \(relativeToAccelerate) times faster than Accelerate Framework")

        }
        
        
        
        exit(0)
    }
    
    func initMetal() -> (MTLDevice, MTLCommandQueue, MTLLibrary, MTLCommandBuffer,
        MTLComputeCommandEncoder){
            // Get access to iPhone or iPad GPU
            var device = MTLCreateSystemDefaultDevice()
            
            // Queue to handle an ordered list of command buffers
            var commandQueue = device.newCommandQueue()
            
            // Access to Metal functions that are stored in Shaders.metal file, e.g. sigmoid()
            var defaultLibrary = device.newDefaultLibrary()
            
            // Buffer for storing encoded commands that are sent to GPU
            var commandBuffer = commandQueue.commandBuffer()
            
            // Encoder for GPU commands
            var computeCommandEncoder = commandBuffer.computeCommandEncoder()
            
            return (device, commandQueue, defaultLibrary!, commandBuffer, computeCommandEncoder)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
        if(self.isViewLoaded() && self.view.window == nil) {
            self.view = nil
        }
    }
    
    
}

