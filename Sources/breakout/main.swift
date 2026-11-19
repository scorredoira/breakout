import AppKit
import BreakoutCore
import Foundation

/// Juega partidas sin gráficos con la pala automática y comprueba que el juego no se rompe.
func selfTest() -> Bool {
    var ok = true
    var totalScore = 0
    var totalLevels = 0
    var finished = 0
    let games = 20
    let start = Date()
    var totalTicks = 0
    var caught = 0
    var mostBalls = 0
    for seed in 0..<games {
        var game = Game(seed: UInt64(seed), shuffledLevels: true)
        var bot = TrackerBot(seed: UInt64(seed))
        let limit = 120 * 60 * 20 // 20 minutos de juego
        while !game.over && game.ticks < limit {
            let result = game.step(move: bot.move(game))
            bot.observe(result)
            caught += result.powerUps
            mostBalls = max(mostBalls, game.balls.count)
            if game.balls.contains(where: { $0.x < 0 || $0.x > Game.width || $0.y < 0 || !$0.vx.isFinite }) {
                print("FALLO semilla \(seed) tick \(game.ticks): una bola fuera del campo")
                ok = false
                break
            }
            let inside = game.ballX >= 0 && game.ballX <= Game.width && game.ballY >= 0
            let speed = (game.ballVX * game.ballVX + game.ballVY * game.ballVY).squareRoot()
            if !inside || !speed.isFinite || speed > Game.maxSpeed * 1.001 {
                print("FALLO semilla \(seed) tick \(game.ticks): bola (\(game.ballX), \(game.ballY)) velocidad \(speed)")
                ok = false
                break
            }
        }
        if game.over { finished += 1 }
        totalScore += game.score
        totalLevels += game.level - 1
        totalTicks += game.ticks
    }
    let seconds = Date().timeIntervalSince(start)
    print("pala automática: \(totalScore / games) puntos y \(Double(totalLevels) / Double(games)) niveles de media en 20 minutos de juego")
    print("partidas perdidas antes del límite: \(finished) de \(games)")
    print("potenciadores recogidos: \(caught / games) por partida · máximo de bolas a la vez: \(mostBalls)")
    print(String(format: "%.1f millones de ticks por segundo en un núcleo", Double(totalTicks) / seconds / 1e6))
    return ok
}

func parseTrainOptions(_ arguments: [String]) -> TrainOptions {
    var options = TrainOptions()
    var index = 1
    func value() -> String {
        index += 1
        guard index < arguments.count else {
            print("falta el valor de \(arguments[index - 1])")
            exit(2)
        }
        return arguments[index]
    }
    while index < arguments.count {
        switch arguments[index] {
        case "--model": options.modelPath = value()
        case "--log": options.logPath = value()
        case "--threads": options.threads = max(1, Int(value()) ?? options.threads)
        case "--minutes": options.minutes = Double(value())
        case "--iters": options.iterations = Int(value())
        case "--lr": options.learningRate = Float(value()) ?? options.learningRate
        case "--gamma": options.gamma = Float(value()) ?? options.gamma
        case "--fresh": options.fresh = true
        default:
            print("opción desconocida: \(arguments[index])")
            exit(2)
        }
        index += 1
    }
    return options
}

/// Prueba sin ventana: la vista real juega unos segundos (con la IA si hay red) y se guarda la imagen.
func snapshot(_ arguments: [String]) -> Bool {
    let path = arguments.count > 1 ? arguments[1] : "snapshot.png"
    let seconds = arguments.count > 2 ? Double(arguments[2]) ?? 20 : 20
    let view = GameView(frame: NSRect(x: 0, y: 0, width: 600, height: 720))
    view.modelPath = arguments.count > 3 ? arguments[3] : "model.bin"
    view.aiPlaying = true
    for _ in 0..<Int(seconds * 60) { view.tick(elapsed: 1.0 / 60) }
    guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return false }
    view.cacheDisplay(in: view.bounds, to: rep)
    guard let data = rep.representation(using: .png, properties: [:]) else { return false }
    return (try? data.write(to: URL(fileURLWithPath: path))) != nil
}

let arguments = Array(CommandLine.arguments.dropFirst())
switch arguments.first ?? "play" {
case "play", "watch":
    // play: juegas tú. watch: empieza jugando la IA. Con la tecla I se cambia.
    let modelPath = arguments.count > 1 ? arguments[1] : "model.bin"
    runApp(modelPath: modelPath, startWithAI: arguments.first == "watch")
case "train":
    runTraining(parseTrainOptions(arguments))
case "selftest":
    let gameOK = selfTest()
    let gradientsOK = gradientCheck()
    exit(gameOK && gradientsOK ? 0 : 1)
case "snapshot":
    exit(snapshot(arguments) ? 0 : 1)
default:
    print("uso: breakout [play | watch | train [--minutes N] [--iters N] [--threads N] [--fresh] | selftest]")
    exit(2)
}
