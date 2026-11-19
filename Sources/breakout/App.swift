import AppKit
import Brain
import BreakoutCore

/// Vista del juego: bucle a paso fijo, teclado y ratón.
final class GameView: NSView {
    private var game = Game(seed: UInt64.random(in: 0...UInt64.max))
    private var bestHuman = 0
    private var bestAI = 0
    private var paused = false

    // IA: la red se lee de `modelPath` y se vuelve a leer cuando el entrenamiento la actualiza.
    var modelPath = "model.bin"
    var aiPlaying = false
    private var agent: Agent?
    private var modelDate: Date?
    private var lastModelCheck: Double = 0
    private var aiMove: Float = 0
    private var speedUp = 1
    private var ballScale: Float = 1
    private var gameOverSince: Double?

    private var leftDown = false
    private var rightDown = false
    private var mouseTarget: Float?

    private var timer: Timer?
    private var lastTime = CACurrentMediaTime()
    private var accumulator: Double = 0

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    func start() {
        let timer = Timer(timeInterval: 1.0 / 120.0, repeats: true) { [weak self] _ in self?.tick() }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    /// `elapsed` solo se pasa en las pruebas sin ventana; en la app se mide el reloj.
    func tick(elapsed: Double? = nil) {
        let now = CACurrentMediaTime()
        if elapsed != nil || now - lastModelCheck > 1 {
            lastModelCheck = now
            reloadModelIfChanged()
        }
        let driving = aiPlaying && agent != nil
        accumulator += (elapsed ?? min(0.1, now - lastTime)) * Double(driving ? speedUp : 1)
        lastTime = now
        let dt = Double(Game.dt)
        var stepped = false
        while accumulator >= dt {
            accumulator -= dt
            if !paused && !game.over {
                if driving, let agent {
                    if game.ticks % ticksPerAction == 0 { aiMove = agent.move(game) }
                    game.step(move: aiMove)
                } else {
                    game.step(move: humanMove())
                }
                stepped = true
            }
        }
        if driving {
            bestAI = max(bestAI, game.score)
        } else {
            bestHuman = max(bestHuman, game.score)
        }
        // La IA vuelve a empezar sola al acabar una partida.
        if game.over && driving {
            if let since = gameOverSince {
                if now - since > 2 { restart() }
            } else {
                gameOverSince = now
            }
        }
        if stepped || paused || game.over { needsDisplay = true }
    }

    private func reloadModelIfChanged() {
        let attributes = try? FileManager.default.attributesOfItem(atPath: modelPath)
        guard let date = attributes?[.modificationDate] as? Date, date != modelDate else { return }
        if let network = loadNetwork(from: modelPath) {
            agent = Agent(network: network)
            modelDate = date
            needsDisplay = true
        }
    }

    /// Teclado: a tope a un lado. Ratón: la pala persigue al cursor, a la misma velocidad máxima.
    private func humanMove() -> Float {
        if leftDown != rightDown { return leftDown ? -1 : 1 }
        if let target = mouseTarget {
            return (target - game.paddleX) / (game.paddleSpeed * Game.dt)
        }
        return 0
    }

    override func draw(_ dirtyRect: NSRect) {
        var hud = HUD()
        hud.paused = paused
        if abs(ballScale - 1) > 0.01 { hud.note = String(format: "bola x%.1f", ballScale) }
        let minutes = Double(game.ticks) / 120 / 60
        let pace = minutes > 0.05 ? " · \(Int(Double(game.score) / minutes)) pts/min" : ""
        if aiPlaying, let agent {
            hud.pilot = "IA (iteración \(agent.network.iteration))" + pace + (speedUp > 1 ? " · x\(speedUp)" : "")
            hud.best = bestAI
        } else {
            hud.pilot = "TÚ" + pace
            hud.best = bestHuman
            if aiPlaying { hud.note = "no hay red en \(modelPath)" }
        }
        render(game: game, hud: hud, in: bounds)
    }

    // MARK: - Entrada

    override func keyDown(with event: NSEvent) {
        switch event.keyCode {
        case 123, 0: // ← / A
            leftDown = true
            mouseTarget = nil
        case 124, 2: // → / D
            rightDown = true
            mouseTarget = nil
        case 35: // P
            if !event.isARepeat { paused.toggle() }
        case 15: // R
            if !event.isARepeat { restart() }
        case 34: // I: cambia entre jugar tú y ver a la IA
            if !event.isARepeat {
                aiPlaying.toggle()
                restart()
            }
        case 126: // ↑: bola más rápida
            ballScale = min(Game.maxSpeedScale, ballScale + (ballScale < 2.95 ? 0.1 : 0.5))
            game.setSpeedScale(ballScale)
        case 125: // ↓: bola más lenta
            ballScale = max(0.5, ballScale - (ballScale > 3.05 ? 0.5 : 0.1))
            game.setSpeedScale(ballScale)
        case 18: speedUp = 1 // 1, 2, 3: velocidad a la que juega la IA
        case 19: speedUp = 4
        case 20: speedUp = 16
        case 49: // Espacio
            if game.over { restart() } else if !event.isARepeat { paused.toggle() }
        case 53: // Esc
            NSApp.terminate(nil)
        default:
            super.keyDown(with: event)
        }
    }

    override func keyUp(with event: NSEvent) {
        switch event.keyCode {
        case 123, 0: leftDown = false
        case 124, 2: rightDown = false
        default: super.keyUp(with: event)
        }
    }

    override func mouseMoved(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        let scale = min(bounds.width / CGFloat(Game.width), bounds.height / CGFloat(Game.height))
        let ox = bounds.midX - CGFloat(Game.width) * scale / 2
        mouseTarget = Float((point.x - ox) / scale)
    }

    override func mouseDragged(with event: NSEvent) {
        mouseMoved(with: event)
    }

    private func restart() {
        game = Game(seed: UInt64.random(in: 0...UInt64.max))
        game.setSpeedScale(ballScale)
        paused = false
        gameOverSince = nil
        aiMove = 0
        accumulator = 0
        lastTime = CACurrentMediaTime()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?
    var modelPath = "model.bin"
    var startWithAI = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        let size = NSSize(width: 600, height: 720)
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Rompeladrillos"
        window.contentAspectRatio = size
        window.contentMinSize = NSSize(width: 300, height: 360)
        window.backgroundColor = .black
        window.acceptsMouseMovedEvents = true

        let view = GameView(frame: NSRect(origin: .zero, size: size))
        view.modelPath = modelPath
        view.aiPlaying = startWithAI
        window.contentView = view
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        view.start()
        self.window = window

        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

func runApp(modelPath: String, startWithAI: Bool) {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    delegate.modelPath = modelPath
    delegate.startWithAI = startWithAI
    app.delegate = delegate
    app.setActivationPolicy(.regular)

    // Menú mínimo, para que funcione Cmd+Q.
    let menu = NSMenu()
    let appItem = NSMenuItem()
    menu.addItem(appItem)
    let appMenu = NSMenu()
    appMenu.addItem(NSMenuItem(title: "Salir", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
    appItem.submenu = appMenu
    app.mainMenu = menu

    withExtendedLifetime(delegate) { app.run() }
}
