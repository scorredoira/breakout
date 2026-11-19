import Foundation

/// Una decisión guardada durante el juego, con lo que hace falta para aprender de ella.
public struct PPOSample {
    public var action: Int
    /// Logaritmo de la probabilidad que tenía la acción cuando se eligió.
    public var oldLogProbability: Float
    public var advantage: Float
    /// Retorno observado: lo que debería predecir la cabeza de valor.
    public var target: Float

    public init(action: Int, oldLogProbability: Float, advantage: Float, target: Float) {
        self.action = action
        self.oldLogProbability = oldLogProbability
        self.advantage = advantage
        self.target = target
    }
}

public struct PPOSettings {
    public var clip: Float = 0.2
    public var valueWeight: Float = 0.5
    public var entropyWeight: Float = 0.01
    public init() {}
}

/// Convierte `count` logits en probabilidades y sus logaritmos, de forma estable.
@inline(__always)
public func softmax(_ logits: UnsafePointer<Float>, count: Int, probabilities: UnsafeMutablePointer<Float>, logProbabilities: UnsafeMutablePointer<Float>) {
    var top = logits[0]
    for k in 1..<count { top = max(top, logits[k]) }
    var sum: Float = 0
    for k in 0..<count {
        probabilities[k] = exp(logits[k] - top)
        sum += probabilities[k]
    }
    let logSum = log(sum)
    for k in 0..<count {
        probabilities[k] /= sum
        logProbabilities[k] = logits[k] - top - logSum
    }
}

/// Pérdida de PPO para una muestra y su gradiente respecto a la salida de la red.
/// La salida son `actions` logits seguidos del valor estimado.
///
///     pérdida = -min(r·A, recorte(r)·A) + cv·½(v − objetivo)² − ce·entropía
///
/// donde r es el cociente entre la probabilidad nueva y la antigua de la acción tomada.
@inline(__always)
public func ppoLossAndGradient(
    output: UnsafePointer<Float>,
    actions: Int,
    sample: PPOSample,
    settings: PPOSettings,
    gradient: UnsafeMutablePointer<Float>
) -> Float {
    precondition(actions <= 8)
    return withUnsafeTemporaryAllocation(of: Float.self, capacity: 16) { scratch in
        let p = scratch.baseAddress!
        let logP = p + 8
        softmax(output, count: actions, probabilities: p, logProbabilities: logP)

        let ratio = exp(logP[sample.action] - sample.oldLogProbability)
        let unclipped = ratio * sample.advantage
        let clipped = max(1 - settings.clip, min(1 + settings.clip, ratio)) * sample.advantage
        // El gradiente solo fluye por el cociente cuando el término sin recortar es el mínimo.
        let active = unclipped <= clipped

        var entropy: Float = 0
        for k in 0..<actions { entropy -= p[k] * logP[k] }

        let value = output[actions]
        let valueError = value - sample.target

        for k in 0..<actions {
            var g: Float = 0
            if active {
                g = -sample.advantage * ratio * ((k == sample.action ? 1 : 0) - p[k])
            }
            g += settings.entropyWeight * p[k] * (logP[k] + entropy)
            gradient[k] = g
        }
        gradient[actions] = settings.valueWeight * valueError

        return -min(unclipped, clipped)
            + settings.valueWeight * 0.5 * valueError * valueError
            - settings.entropyWeight * entropy
    }
}
