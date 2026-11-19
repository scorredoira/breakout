import Brain
import BreakoutCore
import Foundation

struct TrainOptions {
    var modelPath = "model.bin"
    var logPath = "train.csv"
    var threads = max(1, ProcessInfo.processInfo.activeProcessorCount - 2)
    var environments = 512
    var horizon = 128
    var epochs = 4
    var minibatch = 4096
    var learningRate: Float = 3e-4
    var gamma: Float = 0.995
    var lambda: Float = 0.95
    var maxGradientNorm: Float = 0.5
    var minutes: Double?
    var iterations: Int?
    var fresh = false
    var seed: UInt64 = 1
}

// Recompensas: la red no ve la puntuación, ve esto.
private let rewardBrick: Float = 1
private let rewardLevel: Float = 5
private let rewardLifeLost: Float = -5
private let rewardPowerUp: Float = 1
/// Las partidas de entrenamiento se cortan a los 5 minutos de juego.
private let episodeTickLimit = 120 * 60 * 5
private let maxStartLevel = 12
private let maxTrainingSpeedScale: Float = 5
private let fastEvalScale: Float = 3

/// PPO: muchas partidas en paralelo juegan con la red actual, y con lo que ha pasado
/// se ajusta la red para hacer más probables las acciones que salieron mejor de lo esperado.
final class Trainer {
    let options: TrainOptions
    let network: Network
    private let adam: Adam
    private let settings = PPOSettings()
    private var rng: RNG

    private let obsSize = Game.observationSize
    private let total: Int

    private let games: UnsafeMutablePointer<Game>
    private let gameRNG: UnsafeMutablePointer<RNG>

    private let observations: UnsafeMutablePointer<Float>
    private let actions: UnsafeMutablePointer<UInt8>
    private let logProbabilities: UnsafeMutablePointer<Float>
    private let values: UnsafeMutablePointer<Float>
    private let rewards: UnsafeMutablePointer<Float>
    private let dones: UnsafeMutablePointer<Float>
    private let advantages: UnsafeMutablePointer<Float>
    private let targets: UnsafeMutablePointer<Float>
    private let order: UnsafeMutablePointer<Int32>

    private let workspaces: [Workspace]
    private let threadGradients: [UnsafeMutablePointer<Float>]
    private let threadOutput: [UnsafeMutablePointer<Float>]
    private let gradient: UnsafeMutablePointer<Float>
    private let threadScore: UnsafeMutablePointer<Int>
    private let threadLost: UnsafeMutablePointer<Int>
    private let threadLoss: UnsafeMutablePointer<Double>

    init(options: TrainOptions, network: Network) {
        self.options = options
        self.network = network
        adam = Adam(count: network.parameterCount)
        rng = RNG(seed: options.seed &+ UInt64(network.iteration) &* 7919)
        total = options.environments * options.horizon

        games = UnsafeMutablePointer<Game>.allocate(capacity: options.environments)
        gameRNG = UnsafeMutablePointer<RNG>.allocate(capacity: options.environments)
        for index in 0..<options.environments {
            var local = RNG(seed: rng.next())
            (games + index).initialize(to: Trainer.newGame(&local))
            (gameRNG + index).initialize(to: local)
        }

        observations = UnsafeMutablePointer<Float>.allocate(capacity: total * obsSize)
        actions = UnsafeMutablePointer<UInt8>.allocate(capacity: total)
        logProbabilities = UnsafeMutablePointer<Float>.allocate(capacity: total)
        values = UnsafeMutablePointer<Float>.allocate(capacity: total)
        rewards = UnsafeMutablePointer<Float>.allocate(capacity: total)
        dones = UnsafeMutablePointer<Float>.allocate(capacity: total)
        advantages = UnsafeMutablePointer<Float>.allocate(capacity: total)
        targets = UnsafeMutablePointer<Float>.allocate(capacity: total)
        order = UnsafeMutablePointer<Int32>.allocate(capacity: total)
        for index in 0..<total { order[index] = Int32(index) }

        var workspaces: [Workspace] = []
        var threadGradients: [UnsafeMutablePointer<Float>] = []
        var threadOutput: [UnsafeMutablePointer<Float>] = []
        for _ in 0..<options.threads {
            workspaces.append(network.makeWorkspace())
            let g = UnsafeMutablePointer<Float>.allocate(capacity: network.parameterCount)
            g.initialize(repeating: 0, count: network.parameterCount)
            threadGradients.append(g)
            let o = UnsafeMutablePointer<Float>.allocate(capacity: 16)
            o.initialize(repeating: 0, count: 16)
            threadOutput.append(o)
        }
        self.workspaces = workspaces
        self.threadGradients = threadGradients
        self.threadOutput = threadOutput
        gradient = UnsafeMutablePointer<Float>.allocate(capacity: network.parameterCount)
        gradient.initialize(repeating: 0, count: network.parameterCount)
        threadScore = UnsafeMutablePointer<Int>.allocate(capacity: options.threads)
        threadLost = UnsafeMutablePointer<Int>.allocate(capacity: options.threads)
        threadLoss = UnsafeMutablePointer<Double>.allocate(capacity: options.threads)
    }

    private static func newGame(_ rng: inout RNG) -> Game {
        var game = Game(seed: rng.next(), shuffledLevels: true, startLevel: 1 + rng.int(maxStartLevel))
        // La mitad de las partidas, a velocidad normal; el resto, entre x1 y x5 (más frecuentes las bajas).
        if rng.uniform() < 0.5 {
            game.setSpeedScale(exp(rng.uniform() * log(maxTrainingSpeedScale)))
        }
        return game
    }

    struct RolloutStats {
        var pointsPerMinute: Double
        var livesLostPerMinute: Double
    }

    /// Juega `horizon` decisiones en cada partida y calcula ventajas (GAE) y retornos.
    func rollout() -> RolloutStats {
        let threads = options.threads
        let environments = options.environments
        let horizon = options.horizon
        let gamma = options.gamma
        let lambda = options.lambda
        let obsSize = self.obsSize

        DispatchQueue.concurrentPerform(iterations: threads) { worker in
            let workspace = workspaces[worker]
            let first = environments * worker / threads
            let last = environments * (worker + 1) / threads
            var scoreGain = 0
            var lost = 0
            withUnsafeTemporaryAllocation(of: Float.self, capacity: 16) { scratch in
                let p = scratch.baseAddress!
                let logP = p + 8
                for e in first..<last {
                    for t in 0..<horizon {
                        let i = e * horizon + t
                        let obs = observations + i * obsSize
                        games[e].observe(into: obs)
                        workspace.input.update(from: obs, count: obsSize)
                        network.forward(workspace)
                        softmax(workspace.output, count: actionCount, probabilities: p, logProbabilities: logP)

                        let u = gameRNG[e].uniform()
                        var action = 0
                        var cumulative = p[0]
                        while action < actionCount - 1 && u >= cumulative {
                            action += 1
                            cumulative += p[action]
                        }
                        actions[i] = UInt8(action)
                        logProbabilities[i] = logP[action]
                        values[i] = workspace.output[actionCount]

                        var reward: Float = 0
                        let before = games[e].score
                        for _ in 0..<ticksPerAction {
                            let result = games[e].step(move: actionMove(action))
                            reward += Float(result.bricksHit) * rewardBrick
                            if result.levelCleared { reward += rewardLevel }
                            reward += Float(result.powerUps) * rewardPowerUp
                            if result.lostLife {
                                reward += rewardLifeLost
                                lost += 1
                            }
                        }
                        scoreGain += games[e].score - before
                        rewards[i] = reward

                        if games[e].over || games[e].ticks >= episodeTickLimit {
                            dones[i] = 1
                            games[e] = Trainer.newGame(&gameRNG[e])
                        } else {
                            dones[i] = 0
                        }
                    }

                    // Valor del estado siguiente al último, para cerrar la cuenta.
                    games[e].observe(into: workspace.input)
                    network.forward(workspace)
                    var nextValue = workspace.output[actionCount]
                    var advantage: Float = 0
                    for t in stride(from: horizon - 1, through: 0, by: -1) {
                        let i = e * horizon + t
                        let alive = 1 - dones[i]
                        let delta = rewards[i] + gamma * nextValue * alive - values[i]
                        advantage = delta + gamma * lambda * alive * advantage
                        advantages[i] = advantage
                        targets[i] = advantage + values[i]
                        nextValue = values[i]
                    }
                }
            }
            threadScore[worker] = scoreGain
            threadLost[worker] = lost
        }

        var score = 0
        var lost = 0
        for worker in 0..<threads {
            score += threadScore[worker]
            lost += threadLost[worker]
        }
        let minutes = Double(total * ticksPerAction) / 120 / 60
        return RolloutStats(pointsPerMinute: Double(score) / minutes, livesLostPerMinute: Double(lost) / minutes)
    }

    /// Ajusta la red con las decisiones recogidas. Devuelve la pérdida media.
    func update() -> Double {
        let threads = options.threads
        let count = network.parameterCount

        // Ventajas con media 0 y desviación 1: estabiliza el tamaño del paso.
        var mean: Double = 0
        for i in 0..<total { mean += Double(advantages[i]) }
        mean /= Double(total)
        var variance: Double = 0
        for i in 0..<total {
            let d = Double(advantages[i]) - mean
            variance += d * d
        }
        let std = Float((variance / Double(total)).squareRoot()) + 1e-8
        for i in 0..<total { advantages[i] = (advantages[i] - Float(mean)) / std }

        var lossSum: Double = 0
        var batches = 0
        for _ in 0..<options.epochs {
            for i in stride(from: total - 1, to: 0, by: -1) {
                let j = rng.int(i + 1)
                let swap = order[i]
                order[i] = order[j]
                order[j] = swap
            }
            var start = 0
            while start < total {
                let size = min(options.minibatch, total - start)
                let base = start
                DispatchQueue.concurrentPerform(iterations: threads) { worker in
                    let workspace = workspaces[worker]
                    let local = threadGradients[worker]
                    let outputGradient = threadOutput[worker]
                    local.update(repeating: 0, count: count)
                    var loss: Double = 0
                    let first = base + size * worker / threads
                    let last = base + size * (worker + 1) / threads
                    for position in first..<last {
                        let i = Int(order[position])
                        workspace.input.update(from: observations + i * obsSize, count: obsSize)
                        network.forward(workspace)
                        let sample = PPOSample(
                            action: Int(actions[i]),
                            oldLogProbability: logProbabilities[i],
                            advantage: advantages[i],
                            target: targets[i]
                        )
                        loss += Double(ppoLossAndGradient(
                            output: workspace.output, actions: actionCount,
                            sample: sample, settings: settings, gradient: outputGradient
                        ))
                        network.backward(workspace, outputGradient: outputGradient, gradient: local)
                    }
                    threadLoss[worker] = loss
                }

                gradient.update(repeating: 0, count: count)
                let scale = 1 / Float(size)
                for worker in 0..<threads {
                    let local = threadGradients[worker]
                    for k in 0..<count { gradient[k] += local[k] * scale }
                    lossSum += threadLoss[worker] / Double(size)
                }
                var norm: Float = 0
                for k in 0..<count { norm += gradient[k] * gradient[k] }
                norm = norm.squareRoot()
                if norm > options.maxGradientNorm {
                    let shrink = options.maxGradientNorm / norm
                    for k in 0..<count { gradient[k] *= shrink }
                }
                adam.update(parameters: network.parameters, gradient: gradient, learningRate: options.learningRate)
                batches += 1
                start += size
            }
        }
        return lossSum / Double(max(1, batches))
    }
}

func runTraining(_ options: TrainOptions) {
    let network: Network
    if !options.fresh, FileManager.default.fileExists(atPath: options.modelPath) {
        guard let loaded = loadNetwork(from: options.modelPath) else {
            print("\(options.modelPath) es de otra forma de red; usa --fresh para empezar de cero")
            exit(1)
        }
        network = loaded
        print("sigo desde \(options.modelPath) (iteración \(network.iteration))")
    } else {
        network = makeNetwork()
        network.randomize(seed: options.seed)
        print("red nueva: \(network.sizes.map(String.init).joined(separator: " → "))")
    }

    let trainer = Trainer(options: options, network: network)
    let perIteration = options.environments * options.horizon
    print("\(options.threads) hilos · \(options.environments) partidas · \(perIteration) decisiones por iteración")

    let baseline = evaluate(network: nil, threads: options.threads)
    print(String(format: "referencia (pala automática): %.0f puntos en 3 min, %.2f vidas perdidas", baseline.score, baseline.livesLost))
    let fastBaseline = evaluate(network: nil, speedScale: fastEvalScale, threads: options.threads)
    print(String(format: "referencia con la bola x%.0f: %.0f puntos en 3 min, %.2f vidas perdidas", fastEvalScale, fastBaseline.score, fastBaseline.livesLost))

    if !FileManager.default.fileExists(atPath: options.logPath) || options.fresh {
        try? "iteracion,decisiones,segundos,puntos_min_entrenando,vidas_min_entrenando,puntos_3min,vidas_3min\n"
            .write(toFile: options.logPath, atomically: true, encoding: .utf8)
    }

    let start = Date()
    var done = 0
    var best = -Double.infinity
    while true {
        let stats = trainer.rollout()
        let loss = trainer.update()
        network.iteration += 1
        done += 1
        try? network.save(to: options.modelPath)

        let elapsed = Date().timeIntervalSince(start)
        let evaluateNow = done % 25 == 0
        if done % 5 == 0 || evaluateNow {
            var line = String(
                format: "it %5d · %5.0f s · entrenando %5.0f pts/min, %.2f vidas/min · pérdida %.3f",
                Int(network.iteration), elapsed, stats.pointsPerMinute, stats.livesLostPerMinute, loss
            )
            if evaluateNow {
                let result = evaluate(network: network, threads: options.threads)
                best = max(best, result.score)
                line += String(format: " · EVAL %.0f puntos en 3 min (ref. %.0f), %.2f vidas", result.score, baseline.score, result.livesLost)
                let fast = evaluate(network: network, speedScale: fastEvalScale, threads: options.threads)
                line += String(format: " · x%.0f: %.0f (ref. %.0f), %.2f vidas", fastEvalScale, fast.score, fastBaseline.score, fast.livesLost)
                let row = String(
                    format: "%d,%d,%.0f,%.1f,%.3f,%.1f,%.3f\n",
                    Int(network.iteration), Int(network.iteration) * perIteration, elapsed,
                    stats.pointsPerMinute, stats.livesLostPerMinute, result.score, result.livesLost
                )
                if let handle = FileHandle(forWritingAtPath: options.logPath) {
                    handle.seekToEndOfFile()
                    handle.write(row.data(using: .utf8)!)
                    handle.closeFile()
                }
            }
            print(line)
            fflush(stdout)
        }

        if let limit = options.iterations, done >= limit { break }
        if let limit = options.minutes, elapsed >= limit * 60 { break }
    }
    let final = evaluate(network: network, threads: options.threads)
    print(String(format: "fin: %.0f puntos en 3 min (referencia %.0f), %.2f vidas perdidas", final.score, baseline.score, final.livesLost))
    let finalFast = evaluate(network: network, speedScale: fastEvalScale, threads: options.threads)
    print(String(format: "fin con la bola x%.0f: %.0f puntos en 3 min (referencia %.0f), %.2f vidas perdidas", fastEvalScale, finalFast.score, fastBaseline.score, finalFast.livesLost))
}

/// Comprueba la retropropagación contra derivadas numéricas.
func gradientCheck() -> Bool {
    let network = Network(sizes: [20, 32, 32, actionCount + 1])
    network.randomize(seed: 3, outputScale: 0.5)
    let workspace = network.makeWorkspace()
    var rng = RNG(seed: 11)
    var settings = PPOSettings()
    settings.entropyWeight = 0.05

    let batch = 32
    var inputs: [[Float]] = []
    var samples: [PPOSample] = []
    for _ in 0..<batch {
        inputs.append((0..<20).map { _ in rng.normal() })
        samples.append(PPOSample(
            action: rng.int(actionCount),
            oldLogProbability: -1.1 + 0.2 * rng.normal(),
            advantage: rng.normal(),
            target: rng.normal()
        ))
    }

    let count = network.parameterCount
    let analytic = UnsafeMutablePointer<Float>.allocate(capacity: count)
    analytic.initialize(repeating: 0, count: count)
    let outputGradient = UnsafeMutablePointer<Float>.allocate(capacity: 16)
    defer {
        analytic.deallocate()
        outputGradient.deallocate()
    }

    func loss(accumulate: Bool) -> Double {
        var sum: Double = 0
        for b in 0..<batch {
            for k in 0..<20 { workspace.input[k] = inputs[b][k] }
            network.forward(workspace)
            sum += Double(ppoLossAndGradient(
                output: workspace.output, actions: actionCount,
                sample: samples[b], settings: settings, gradient: outputGradient
            ))
            if accumulate {
                network.backward(workspace, outputGradient: outputGradient, gradient: analytic)
            }
        }
        return sum
    }
    let center = loss(accumulate: true)

    var worst: Double = 0
    var checked = 0
    let epsilon: Float = 2e-3
    while checked < 200 {
        let index = rng.int(count)
        let original = network.parameters[index]
        network.parameters[index] = original + epsilon
        let up = loss(accumulate: false)
        network.parameters[index] = original - epsilon
        let down = loss(accumulate: false)
        network.parameters[index] = original
        let exact = Double(analytic[index])
        if exact == 0 && up == down { continue } // relleno o neurona apagada
        // La pérdida tiene esquinas (ReLU y el recorte de PPO). Si el paso cruza una, la
        // derivada centrada no vale, pero la de uno de los dos lados sí tiene que coincidir.
        let step = Double(epsilon)
        let candidates = [(up - down) / (2 * step), (up - center) / step, (center - down) / step]
        let error = candidates.map { abs($0 - exact) / max(0.05, abs($0) + abs(exact)) }.min()!
        worst = max(worst, error)
        checked += 1
    }
    print(String(format: "gradientes: peor error relativo %.4f en %d parámetros", worst, checked))
    return worst < 0.03
}
