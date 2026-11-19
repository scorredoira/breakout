#if canImport(Darwin)
import Darwin
#endif

/// Generador determinista (SplitMix64): misma semilla, misma partida.
public struct RNG {
    public var state: UInt64

    public init(seed: UInt64) {
        state = seed &+ 0x9E37_79B9_7F4A_7C15
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniforme en [0, 1).
    public mutating func uniform() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }

    /// Entero en [0, n).
    public mutating func int(_ n: Int) -> Int {
        Int(next() % UInt64(n))
    }

    /// Normal estándar (Box-Muller).
    public mutating func normal() -> Float {
        let u1 = max(uniform(), 1e-7)
        let u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * Float.pi * u2)
    }
}
