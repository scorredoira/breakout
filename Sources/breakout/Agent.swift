import Brain
import BreakoutCore
import Foundation

/// Acciones de la red: 0 = izquierda, 1 = quieta, 2 = derecha.
let actionCount = 3

@inline(__always)
func actionMove(_ action: Int) -> Float { Float(action) - 1 }

/// La red decide 60 veces por segundo (cada 2 ticks de física).
let ticksPerAction = 2

func makeNetwork() -> Network {
    Network(sizes: [Game.observationSize, 128, 128, actionCount + 1])
}

/// Lee una red del disco. Acepta también las entrenadas antes de añadir entradas al final:
/// ocupan la misma memoria (el relleno era cero), así que se copian tal cual.
func loadNetwork(from path: String) -> Network? {
    guard let loaded = Network.load(from: path) else { return nil }
    let expected = makeNetwork()
    if loaded.sizes == expected.sizes { return loaded }
    guard loaded.sizes.dropFirst().elementsEqual(expected.sizes.dropFirst()),
          loaded.sizes[0] < expected.sizes[0],
          loaded.parameterCount == expected.parameterCount
    else { return nil }
    expected.parameters.update(from: loaded.parameters, count: loaded.parameterCount)
    expected.iteration = loaded.iteration
    return expected
}

/// Una red entrenada a los mandos: elige siempre la acción más probable.
final class Agent {
    let network: Network
    private let workspace: Workspace

    init(network: Network) {
        self.network = network
        workspace = network.makeWorkspace()
    }

    func move(_ game: Game) -> Float {
        game.observe(into: workspace.input)
        network.forward(workspace)
        let out = workspace.output
        var best = 0
        for k in 1..<actionCount where out[k] > out[best] { best = k }
        return actionMove(best)
    }
}

struct EvalResult {
    var score: Double
    var livesLost: Double
}

/// Puntos conseguidos en `seconds` de juego desde el nivel 1, de media sobre `games` partidas fijas.
/// Con `network == nil` juega la pala automática de referencia.
func evaluate(network: Network?, games: Int = 32, seconds: Int = 180, speedScale: Float = 1, threads: Int) -> EvalResult {
    let scores = UnsafeMutablePointer<Int>.allocate(capacity: games)
    let lost = UnsafeMutablePointer<Int>.allocate(capacity: games)
    defer {
        scores.deallocate()
        lost.deallocate()
    }
    let limit = seconds * 120
    let workers = max(1, min(threads, games))
    DispatchQueue.concurrentPerform(iterations: workers) { worker in
        let agent = network.map { Agent(network: $0) }
        for index in stride(from: worker, to: games, by: workers) {
            let seed = UInt64(1000 + index)
            var game = Game(seed: seed, shuffledLevels: true)
            game.setSpeedScale(speedScale)
            var bot = TrackerBot(seed: seed)
            var move: Float = 0
            while !game.over && game.ticks < limit {
                if let agent {
                    if game.ticks % ticksPerAction == 0 { move = agent.move(game) }
                    game.step(move: move)
                } else {
                    bot.observe(game.step(move: bot.move(game)))
                }
            }
            scores[index] = game.score
            lost[index] = Game.startLives - game.lives
        }
    }
    var result = EvalResult(score: 0, livesLost: 0)
    for index in 0..<games {
        result.score += Double(scores[index])
        result.livesLost += Double(lost[index])
    }
    result.score /= Double(games)
    result.livesLost /= Double(games)
    return result
}
