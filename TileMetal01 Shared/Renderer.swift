//
//  Renderer.swift
//  TileMetal01 Shared
//
//  Created by gzonelee on 2023/08/05.
//

// Our platform independent renderer class

import Metal
import MetalKit
import simd

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100

let maxBuffersInFlight = 3

enum RendererError: Error {
    case badVertexDescriptor
}

class Renderer: NSObject, MTKViewDelegate {
    
    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    
    var pipelineState: MTLRenderPipelineState
    var pipelineStateFloor: MTLRenderPipelineState
    var shadowPipelineState: MTLRenderPipelineState
    
    var depthState: MTLDepthStencilState
    var colorMap: MTLTexture
    var floorMap: MTLTexture
    var pinkMap: MTLTexture
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    var uniformBufferOffset = 0
    
    var uniformBufferIndex = 0
    
    var uniforms: UnsafeMutablePointer<Uniforms>
    
    var projectionMatrix: matrix_float4x4 = matrix_float4x4()
    var shadowProjectionMatrix: matrix_float4x4 = matrix_float4x4()
    
    var rotation: Float = 0
    
    var boxMesh: MTKMesh
    var ellipsoidMesh: MTKMesh
    
    var sunPosition: float3 = [0, 10, -10.0]
    var shadowTexture: MTLTexture?
    
    init?(metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        
        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight
        
        guard let buffer = self.device.makeBuffer(length:uniformBufferSize, options:[MTLResourceOptions.storageModeShared]) else { return nil }
        dynamicUniformBuffer = buffer
        
        self.dynamicUniformBuffer.label = "UniformBuffer"
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:Uniforms.self, capacity:1)
        
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.sampleCount = 1
        
        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()
        
        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }
        
        do {
            pipelineStateFloor = try Renderer.buildRenderPipelineFloorWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }
        
        do {
            shadowPipelineState = try Renderer.buildRenderPipelineShadowWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }
        
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDescriptor.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor:depthStateDescriptor) else { return nil }
        depthState = state
        
        do {
            ellipsoidMesh = try Renderer.buildEllipsoidMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }
        do {
            boxMesh = try Renderer.buildBoxMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }
        
        do {
            colorMap = try Renderer.loadTexture(device: device, textureName: "ColorMap")
            floorMap = try Renderer.loadTexture(device: device, textureName: "Floor")
            pinkMap = try Renderer.loadTexture(device: device, textureName: "Pink")
        } catch {
            print("Unable to load texture. Error info: \(error)")
            return nil
        }
        
        super.init()
        
    }
    
    class func makeTexture(device: MTLDevice, width: CGFloat, height: CGFloat, pixelFormat: MTLPixelFormat, usage: MTLTextureUsage, storageMode: MTLStorageMode) -> MTLTexture? {
        
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat, width: Int(width), height: Int(height), mipmapped: false)
        descriptor.usage = usage
        descriptor.storageMode = storageMode
        
        return device.makeTexture(descriptor: descriptor)
    }
    
    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices
        
        let mtlVertexDescriptor = MTLVertexDescriptor()
        
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue
        
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue
        
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction = MTLVertexStepFunction.perVertex
        
        return mtlVertexDescriptor
    }
    
    class func buildRenderPipelineShadowWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object
        
        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: "vertexShadowShader")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline for shadow"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        
        pipelineDescriptor.colorAttachments[0].pixelFormat = .invalid
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    class func buildRenderPipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object
        
        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "fragmentShader")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.rasterSampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        pipelineDescriptor.stencilAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    class func buildRenderPipelineFloorWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object
        
        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "fragmentFloorShader")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline for floor"
        pipelineDescriptor.rasterSampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        pipelineDescriptor.stencilAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    class func buildBoxMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
        
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
        let mdlMesh = MDLMesh.newBox(withDimensions: SIMD3<Float>(1, 1, 1),
                                     segments: SIMD3<UInt32>(2, 2, 2),
                                     geometryType: MDLGeometryType.triangles,
                                     inwardNormals:false,
                                     allocator: metalAllocator)
        
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
        
        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        
        mdlMesh.vertexDescriptor = mdlVertexDescriptor
        
        return try MTKMesh(mesh:mdlMesh, device:device)
    }
    
    class func buildEllipsoidMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
        
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
        let mdlMesh = MDLMesh.newEllipsoid(withRadii: vector_float3(1, 1, 1),
                                           radialSegments: 36,
                                           verticalSegments: 20,
                                           geometryType: .triangles,
                                           inwardNormals: false,
                                           hemisphere: false,
                                           allocator: metalAllocator)
        
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
        
        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        
        mdlMesh.vertexDescriptor = mdlVertexDescriptor
        
        return try MTKMesh(mesh:mdlMesh, device:device)
    }
    
    class func loadTexture(device: MTLDevice,
                           textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling
        
        let textureLoader = MTKTextureLoader(device: device)
        
        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]
        
        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
        
    }
    
    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight
        
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:Uniforms.self, capacity:1)
    }
    
    private func updateGameState() {
        /// Update any game state before rendering
        
        uniforms[0].projectionMatrix = projectionMatrix
        
        let viewMatrix = matrix4x4_translation(0.0, 0.0, 18.0)
        uniforms[0].viewMatrix = viewMatrix
        rotation += 0.01
        
        uniforms[0].shadowProjectionMatrix = shadowProjectionMatrix
        uniforms[0].shadowViewMatrix = float4x4(eye: sunPosition, center: [0, 0, 0], up: [0, 1, 0])
    }
    
    func draw(in view: MTKView) {
        /// Per frame updates hare
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
                semaphore.signal()
            }
            
            self.updateDynamicBufferState()
            
            self.updateGameState()
            
            // shadow pipeline
            let shadowPassDescriptor = MTLRenderPassDescriptor()
            shadowPassDescriptor.depthAttachment.texture = shadowTexture
            shadowPassDescriptor.depthAttachment.loadAction = .clear
            shadowPassDescriptor.depthAttachment.storeAction = .store
            
            if let shadowEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: shadowPassDescriptor) {
                shadowEncoder.setRenderPipelineState(shadowPipelineState)
                shadowEncoder.setDepthStencilState(depthState)
                shadowEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                shadowEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                
                drawScene(renderEncoder: shadowEncoder, showFloor: false)
                shadowEncoder.endEncoding()
            }
            
            /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
            ///   holding onto the drawable and blocking the display pipeline any longer than necessary
            let renderPassDescriptor = view.currentRenderPassDescriptor
            
            if let renderPassDescriptor = renderPassDescriptor, let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                
                /// Final pass rendering code here
                renderEncoder.label = "Primary Render Encoder"
                
                
                renderEncoder.setCullMode(.front)
                
                renderEncoder.setFrontFacing(.counterClockwise)
                
                renderEncoder.setRenderPipelineState(pipelineState)
                
                renderEncoder.setDepthStencilState(depthState)
                
                renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                renderEncoder.setFragmentTexture(shadowTexture, index: TextureIndex.shadow.rawValue)
                
                drawScene(renderEncoder: renderEncoder, showFloor: true)
                
                renderEncoder.pushDebugGroup("Draw light")
                let m = float4x4(translation: sunPosition)
                drawBox(modelMatrix: m, renderEncoder: renderEncoder, texture: pinkMap)
                renderEncoder.popDebugGroup()
 
                
                renderEncoder.endEncoding()
                
                if let drawable = view.currentDrawable {
                    commandBuffer.present(drawable)
                }
            }
            
            commandBuffer.commit()
        }
    }
    
    func drawScene(renderEncoder: MTLRenderCommandEncoder, showFloor: Bool) {
        let rotationAxis = SIMD3<Float>(0, 1, 0)
        let rotationMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        
        renderEncoder.pushDebugGroup("Draw Box1")
        drawBox(modelMatrix: rotationMatrix * float4x4(translation: [0, 2, 0]) * float4x4.init(scaling: [1, 1, 1]), renderEncoder: renderEncoder, texture: colorMap)
        renderEncoder.popDebugGroup()
        
        renderEncoder.pushDebugGroup("Draw Box2")
        var m = rotationMatrix * float4x4(translation: [-4, 0, 9]) * float4x4(scaling: [2, 1, 1])
        drawBox(modelMatrix: m, renderEncoder: renderEncoder, texture: colorMap)
        renderEncoder.popDebugGroup()
        
        renderEncoder.pushDebugGroup("Draw Ellipsoid")
        m = rotationMatrix * float4x4(translation: [8, 2, 0]) * float4x4(scaling: [1, 3, 2])
        drawEllipsoid(modelMatrix: m, renderEncoder: renderEncoder, texture: pinkMap)
        renderEncoder.popDebugGroup()
        
        
        if showFloor {
            renderEncoder.pushDebugGroup("Draw floor")
            //        renderEncoder.setRenderPipelineState(pipelineStateFloor)
            m = rotationMatrix * float4x4(translation: [0, -2, 0]) * float4x4.init(scaling: [20, 0.1, 20])
            drawBox(modelMatrix: m, renderEncoder: renderEncoder, texture: floorMap)
            renderEncoder.popDebugGroup()
        }
    }
    
    func drawBox(modelMatrix: float4x4, renderEncoder: MTLRenderCommandEncoder, texture: MTLTexture?) {
        let mu = ModelUniforms(modelMatrix: modelMatrix)
        withUnsafePointer(to: mu) { up in
            renderEncoder.setVertexBytes(up,
                                         length: MemoryLayout<ModelUniforms>.stride,
                                         index: BufferIndex.modelUniforms.rawValue)
            renderEncoder.setFragmentBytes(up,
                                         length: MemoryLayout<ModelUniforms>.stride,
                                         index: BufferIndex.modelUniforms.rawValue)
        }
        drawMesh(mesh: boxMesh, renderEncoder: renderEncoder, texture: texture)
    }
    
    func drawEllipsoid(modelMatrix: float4x4, renderEncoder: MTLRenderCommandEncoder, texture: MTLTexture?) {
        let mu = ModelUniforms(modelMatrix: modelMatrix)
        withUnsafePointer(to: mu) { up in
            renderEncoder.setVertexBytes(up,
                                         length: MemoryLayout<ModelUniforms>.stride,
                                         index: BufferIndex.modelUniforms.rawValue)
        }
        drawMesh(mesh: ellipsoidMesh, renderEncoder: renderEncoder, texture: texture)
    }
    
    func drawMesh(mesh: MTKMesh, renderEncoder: MTLRenderCommandEncoder, texture: MTLTexture?) {
            for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
            guard let layout = element as? MDLVertexBufferLayout else {
                return
            }
            
            if layout.stride != 0 {
                let buffer = mesh.vertexBuffers[index]
                renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
            }
        }
        
        
        renderEncoder.setFragmentTexture(texture, index: TextureIndex.color.rawValue)
        
        for submesh in mesh.submeshes {
            renderEncoder.drawIndexedPrimitives(
                type: submesh.primitiveType,
                indexCount: submesh.indexCount,
                indexType: submesh.indexType,
                indexBuffer: submesh.indexBuffer.buffer,
                indexBufferOffset: submesh.indexBuffer.offset)
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here
        
        let aspect = Float(size.width) / Float(size.height)
        projectionMatrix = float4x4(projectionFov: radians_from_degrees(65), near: 0.1, far: 100.0, aspect: aspect, lhs: true)
        
        var w: Float = 20
        var h: Float = 20
        if size.width != 0 && size.height != 0 {
            if size.width > size.height {
                w = w * aspect
            }
            else {
                h = h / aspect
            }
        }
        shadowProjectionMatrix = orthographicMatrix(left: -w, right: w, bottom: -h, top: h, near: 0.01, far: 100)
        shadowTexture = Self.makeTexture(device: device, width: 2048, height: 2048, pixelFormat: .depth32Float, usage: [.shaderRead, .renderTarget], storageMode: .private)
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -1),
                                         vector_float4( 0,  0, zs * nearZ, 0)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}

func orthographicMatrix(left: Float, right: Float, bottom: Float, top: Float, near: Float, far: Float) -> float4x4 {
    let scaleX = 2.0 / (right - left)
    let scaleY = 2.0 / (top - bottom)
    let scaleZ = 1.0 / (far - near)

    let translationX = (right + left) / (left - right)
    let translationY = (top + bottom) / (bottom - top)
    let translationZ = near / (near - far)

    let matrix = float4x4([
        simd_float4(scaleX, 0, 0, 0),
        simd_float4(0, scaleY, 0, 0),
        simd_float4(0, 0, scaleZ, 0),
        simd_float4(translationX, translationY, translationZ, 1)
    ])

    return matrix
}
