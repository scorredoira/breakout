
import Foundation

// Red neuronal densa escrita a mano: capas lineales con ReLU entre ellas y salida lineal.
// Los anchos de entrada de cada capa se rellenan con ceros hasta un múltiplo de 16 para
// que todos los bucles internos trabajen con vectores SIMD de 16 floats.

let lane = 16

@inline(__always)
func roundUp(_ n: Int) -> Int { (n + lane - 1) / lane * lane }

/// Producto escalar de `n` floats (`n` múltiplo de 16).
@inline(__always)
func dot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int) -> Float {
    let ra = UnsafeRawPointer(a)
    let rb = UnsafeRawPointer(b)
    var s0 = SIMD16<Float>.zero
    var s1 = SIMD16<Float>.zero
    var j = 0
    while j + 2 * lane <= n {
        s0 += ra.loadUnaligned(fromByteOffset: j * 4, as: SIMD16<Float>.self)
            * rb.loadUnaligned(fromByteOffset: j * 4, as: SIMD16<Float>.self)
        s1 += ra.loadUnaligned(fromByteOffset: (j + lane) * 4, as: SIMD16<Float>.self)
            * rb.loadUnaligned(fromByteOffset: (j + lane) * 4, as: SIMD16<Float>.self)
        j += 2 * lane
    }
    if j < n {
        s0 += ra.loadUnaligned(fromByteOffset: j * 4, as: SIMD16<Float>.self)
            * rb.loadUnaligned(fromByteOffset: j * 4, as: SIMD16<Float>.self)
    }
    return (s0 + s1).sum()
}

/// y += k * x sobre `n` floats (`n` múltiplo de 16).
@inline(__always)
func axpy(_ y: UnsafeMutablePointer<Float>, _ x: UnsafePointer<Float>, _ k: Float, _ n: Int) {
    let ry = UnsafeMutableRawPointer(y)
    let rx = UnsafeRawPointer(x)
    let kv = SIMD16<Float>(repeating: k)
    var j = 0
    while j < n {
        let v = ry.loadUnaligned(fromByteOffset: j * 4, as: SIMD16<Float>.self)
            + kv * rx.loadUnaligned(fromByteOffset: j * 4, as: SIMD16<Float>.self)
        ry.storeBytes(of: v, toByteOffset: j * 4, as: SIMD16<Float>.self)
        j += lane
    }
}

func allocateFloats(_ count: Int) -> UnsafeMutablePointer<Float> {
    let raw = UnsafeMutableRawPointer.allocate(byteCount: max(1, count) * 4, alignment: 64)
    let pointer = raw.bindMemory(to: Float.self, capacity: max(1, count))
    pointer.initialize(repeating: 0, count: max(1, count))
    return pointer
}

struct Layer {
    let inputs: Int
    let outputs: Int
    /// Ancho de fila en memoria: `inputs` redondeado a múltiplo de 16.
    let stride: Int
    let weights: Int
    let biases: Int
}

/// Memoria de trabajo de un hilo: activaciones y gradientes intermedios de una muestra.
public final class Workspace {
    let activations: [UnsafeMutablePointer<Float>]
    let deltas: [UnsafeMutablePointer<Float>]

    init(network: Network) {
        var activations: [UnsafeMutablePointer<Float>] = []
        var deltas: [UnsafeMutablePointer<Float>] = []
        for size in network.sizes {
            activations.append(allocateFloats(roundUp(size)))
            deltas.append(allocateFloats(roundUp(size)))
        }
        self.activations = activations
        self.deltas = deltas
    }

    deinit {
        for pointer in activations { pointer.deallocate() }
        for pointer in deltas { pointer.deallocate() }
    }

    /// Donde se escribe la entrada antes de `forward`.
    public var input: UnsafeMutablePointer<Float> { activations[0] }
    /// Salida de la red tras `forward`.
    public var output: UnsafeMutablePointer<Float> { activations[activations.count - 1] }
}

public final class Network {
    public let sizes: [Int]
    let layers: [Layer]
    /// Número de floats del bloque de parámetros (incluye el relleno, que siempre vale cero).
    public let parameterCount: Int
    public let parameters: UnsafeMutablePointer<Float>
    /// Contador libre que viaja con el fichero (iteraciones de entrenamiento).
    public var iteration: UInt32 = 0

    public init(sizes: [Int]) {
        precondition(sizes.count >= 2)
        self.sizes = sizes
        var layers: [Layer] = []
        var offset = 0
        for l in 0..<(sizes.count - 1) {
            let stride = roundUp(sizes[l])
            let weights = offset
            offset += stride * sizes[l + 1]
            let biases = offset
            offset += roundUp(sizes[l + 1])
            layers.append(Layer(inputs: sizes[l], outputs: sizes[l + 1], stride: stride, weights: weights, biases: biases))
        }
        self.layers = layers
        parameterCount = offset
        parameters = allocateFloats(offset)
    }

    deinit { parameters.deallocate() }

    public func makeWorkspace() -> Workspace { Workspace(network: self) }

    /// Inicialización de He en las capas ocultas y casi cero en la de salida.
    public func randomize(seed: UInt64, outputScale: Float = 0.01) {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        func uniform() -> Float {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            return Float(z >> 40) / Float(1 << 24)
        }
        func normal() -> Float {
            let u1 = max(uniform(), 1e-7)
            return (-2 * log(u1)).squareRoot() * cos(2 * Float.pi * uniform())
        }
        for (index, layer) in layers.enumerated() {
            let last = index == layers.count - 1
            let std = last ? outputScale : (2 / Float(layer.inputs)).squareRoot()
            for i in 0..<layer.outputs {
                for j in 0..<layer.inputs {
                    parameters[layer.weights + i * layer.stride + j] = normal() * std
                }
                parameters[layer.biases + i] = 0
            }
        }
    }

    /// Calcula la salida para la entrada escrita en `workspace.input`.
    public func forward(_ workspace: Workspace) {
        for (index, layer) in layers.enumerated() {
            let input = UnsafePointer(workspace.activations[index])
            let output = workspace.activations[index + 1]
            let weights = UnsafePointer(parameters + layer.weights)
            let biases = parameters + layer.biases
            let last = index == layers.count - 1
            for i in 0..<layer.outputs {
                let z = biases[i] + dot(weights + i * layer.stride, input, layer.stride)
                output[i] = last ? z : max(0, z)
            }
        }
    }

    /// Retropropagación: dado el gradiente de la pérdida respecto a la salida (`outputGradient`),
    /// suma en `gradient` el gradiente respecto a cada parámetro. Requiere un `forward` previo
    /// con el mismo `workspace`.
    public func backward(_ workspace: Workspace, outputGradient: UnsafePointer<Float>, gradient: UnsafeMutablePointer<Float>) {
        let top = layers.count
        for i in 0..<sizes[top] { workspace.deltas[top][i] = outputGradient[i] }

        for index in stride(from: top - 1, through: 0, by: -1) {
            let layer = layers[index]
            let input = UnsafePointer(workspace.activations[index])
            let output = workspace.activations[index + 1]
            let delta = workspace.deltas[index + 1]
            let below = workspace.deltas[index]
            let weights = UnsafePointer(parameters + layer.weights)
            let weightGradient = gradient + layer.weights
            let biasGradient = gradient + layer.biases
            let last = index == top - 1
            let propagate = index > 0
            if propagate { below.update(repeating: 0, count: layer.stride) }

            for i in 0..<layer.outputs {
                // Derivada de ReLU: pasa el gradiente solo si la neurona estaba activa.
                let d = (last || output[i] > 0) ? delta[i] : 0
                if d == 0 { continue }
                biasGradient[i] += d
                axpy(weightGradient + i * layer.stride, input, d, layer.stride)
                if propagate { axpy(below, weights + i * layer.stride, d, layer.stride) }
            }
        }
    }

    // MARK: - Fichero

    private static let magic: [UInt8] = Array("BRKNN001".utf8)

    public func save(to path: String) throws {
        var data = Data(Network.magic)
        func append(_ value: UInt32) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }
        append(iteration)
        append(UInt32(sizes.count))
        for size in sizes { append(UInt32(size)) }
        data.append(UnsafeBufferPointer(start: parameters, count: parameterCount))
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    public static func load(from path: String) -> Network? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }
        var cursor = 0
        func readUInt32() -> UInt32? {
            guard cursor + 4 <= data.count else { return nil }
            var value: UInt32 = 0
            for k in 0..<4 { value |= UInt32(data[data.startIndex + cursor + k]) << (8 * UInt32(k)) }
            cursor += 4
            return value
        }
        guard data.count >= 8, Array(data.prefix(8)) == magic else { return nil }
        cursor = 8
        guard let iteration = readUInt32(), let count = readUInt32(), count >= 2, count <= 16 else { return nil }
        var sizes: [Int] = []
        for _ in 0..<count {
            guard let size = readUInt32(), size > 0, size <= 65536 else { return nil }
            sizes.append(Int(size))
        }
        let network = Network(sizes: sizes)
        guard data.count - cursor == network.parameterCount * 4 else { return nil }
        data.withUnsafeBytes { raw in
            let source = raw.baseAddress!.advanced(by: cursor)
            UnsafeMutableRawPointer(network.parameters).copyMemory(from: source, byteCount: network.parameterCount * 4)
        }
        network.iteration = iteration
        return network
    }
}

/// Optimizador Adam sobre un bloque plano de parámetros.
public final class Adam {
    let count: Int
    let m: UnsafeMutablePointer<Float>
    let v: UnsafeMutablePointer<Float>
    var step = 0
    public var beta1: Float = 0.9
    public var beta2: Float = 0.999
    public var epsilon: Float = 1e-5

    public init(count: Int) {
        self.count = count
        m = allocateFloats(count)
        v = allocateFloats(count)
    }

    deinit {
        m.deallocate()
        v.deallocate()
    }

    public func update(parameters: UnsafeMutablePointer<Float>, gradient: UnsafePointer<Float>, learningRate: Float) {
        step += 1
        let correction1 = 1 - pow(beta1, Float(step))
        let correction2 = 1 - pow(beta2, Float(step))
        let rate = learningRate * correction2.squareRoot() / correction1
        for i in 0..<count {
            let g = gradient[i]
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            parameters[i] -= rate * m[i] / (v[i].squareRoot() + epsilon)
        }
    }
}
