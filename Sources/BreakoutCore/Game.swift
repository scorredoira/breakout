#if canImport(Darwin)
import Darwin
#endif

/// Lo que ha pasado en un tick de física.
public struct StepResult {
    public var bricksHit = 0
    public var paddleHit = false
    public var lostLife = false
    public var levelCleared = false
    /// Potenciadores recogidos con la pala.
    public var powerUps = 0
    public init() {}
}

public struct Ball {
    public var x: Float
    public var y: Float
    public var vx: Float
    public var vy: Float
}

/// Potenciadores: caen de algunos ladrillos al romperlos y se recogen con la pala.
public enum PowerUp: UInt8, CaseIterable {
    /// Cada bola se convierte en tres.
    case multiball
    /// Pala más ancha durante unos segundos.
    case wide
    /// Las bolas atraviesan los ladrillos y los rompen de un golpe durante unos segundos.
    case fire
    /// Una vida más.
    case life
}

/// Un potenciador cayendo.
public struct Capsule {
    public var x: Float
    public var y: Float
    public var kind: PowerUp
}

/// Rompeladrillos: lógica pura, determinista y sin gráficos.
/// El campo mide `width` x `height`, con el origen arriba a la izquierda y la Y hacia abajo.
public struct Game {
    // Campo y física
    public static let width: Float = 1.0
    public static let height: Float = 1.2
    public static let dt: Float = 1.0 / 120.0

    // Pala
    public static let paddleW: Float = 0.16
    public static let paddleH: Float = 0.02
    public static let paddleY: Float = 1.12
    public static let paddleSpeed: Float = 1.2
    public static let maxBounceAngle: Float = 60 * .pi / 180

    // Bola
    public static let ballR: Float = 0.012
    public static let baseSpeed: Float = 0.85
    public static let maxSpeed: Float = 1.5
    /// Lo máximo que avanza la bola en un trozo de tick (lo que recorre a `maxSpeed`, con margen de redondeo).
    static let maxTravelPerStep: Float = maxSpeed * dt * 1.001
    public static let maxSpeedScale: Float = 10
    public static let minVerticalFraction: Float = 0.3
    public static let launchDelay: Float = 0.6
    public static let maxBalls = 8

    // Ladrillos
    public static let cols = 10
    public static let rows = 8
    public static let brickLeft: Float = 0.03
    public static let brickTop: Float = 0.14
    public static let brickW: Float = (width - 2 * brickLeft) / Float(cols)
    public static let brickH: Float = 0.036

    // Potenciadores
    public static let capsuleW: Float = 0.07
    public static let capsuleH: Float = 0.028
    public static let capsuleSpeed: Float = 0.35
    public static let dropChance: Float = 0.12
    public static let maxCapsules = 4
    public static let wideFactor: Float = 1.6
    public static let wideSeconds: Float = 12
    public static let fireSeconds: Float = 8
    public static let maxLives = 5

    public static let startLives = 3
    public static let patternCount = 6

    /// Bolas y potenciadores que ve la red (las más urgentes).
    static let seenBalls = 4
    static let seenCapsules = 2
    /// Tamaño del vector que ve la red.
    public static let observationSize = 10 + seenBalls * 6 + seenCapsules * 8 + cols * rows

    /// Bolas en juego. Antes del saque hay una sola, pegada a la pala.
    public private(set) var balls: [Ball] = []
    public private(set) var capsules: [Capsule] = []
    /// Segundos que quedan de pala ancha y de bola de fuego.
    public private(set) var wideTimer: Float = 0
    public private(set) var fireTimer: Float = 0

    public private(set) var speed: Float = Game.baseSpeed
    public private(set) var launched = false
    public private(set) var launchTimer: Float = Game.launchDelay

    public private(set) var paddleX: Float = Game.width / 2

    /// Golpes que le quedan a cada ladrillo (0 = no hay), fila a fila.
    public private(set) var bricks = [UInt8](repeating: 0, count: Game.cols * Game.rows)
    public private(set) var bricksLeft = 0
    public private(set) var bricksTotal = 0

    public private(set) var lives = Game.startLives
    public private(set) var score = 0
    public private(set) var level = 1
    public private(set) var ticks = 0
    public private(set) var over = false

    private var rng: RNG
    private var patternOffset: Int

    /// - Parameters:
    ///   - seed: semilla de la partida.
    ///   - shuffledLevels: si es `true`, el orden de los niveles depende de la semilla
    ///     (para entrenar con variedad); si no, siempre empieza por el mismo nivel.
    ///   - startLevel: nivel inicial (los niveles altos llevan la bola más rápida).
    public init(seed: UInt64, shuffledLevels: Bool = false, startLevel: Int = 1) {
        rng = RNG(seed: seed)
        patternOffset = shuffledLevels ? rng.int(Game.patternCount) : 0
        level = max(1, startLevel)
        balls.reserveCapacity(Game.maxBalls)
        capsules.reserveCapacity(Game.maxCapsules)
        buildLevel()
        resetBall()
    }

    /// Ancho actual de la pala.
    public var paddleWidth: Float { wideTimer > 0 ? Game.paddleW * Game.wideFactor : Game.paddleW }

    // MARK: - La bola más urgente

    /// Tiempo aproximado hasta que una bola llega a la altura de la pala.
    @inline(__always)
    private func timeToPaddle(_ ball: Ball) -> Float {
        if ball.vy > 0 { return (Game.paddleY - ball.y) / ball.vy }
        if ball.vy < 0 { return (ball.y + Game.paddleY) / -ball.vy }
        return 0
    }

    /// La bola que antes va a llegar abajo: la que hay que atender primero.
    public var mainBall: Ball {
        var best = balls.first ?? Ball(x: paddleX, y: Game.paddleY - Game.ballR, vx: 0, vy: 0)
        var bestTime = timeToPaddle(best)
        for ball in balls.dropFirst() {
            let time = timeToPaddle(ball)
            if time < bestTime {
                best = ball
                bestTime = time
            }
        }
        return best
    }

    public var ballX: Float { mainBall.x }
    public var ballY: Float { mainBall.y }
    public var ballVX: Float { mainBall.vx }
    public var ballVY: Float { mainBall.vy }

    // MARK: - Paso de simulación

    /// Avanza un tick. `move` va de -1 (izquierda a tope) a 1 (derecha a tope).
    @discardableResult
    public mutating func step(move: Float) -> StepResult {
        var result = StepResult()
        if over { return result }
        ticks += 1
        // Con la bola muy rápida el tick se parte en trozos, para que en ninguno
        // avance tanto como para atravesar la pala o un ladrillo sin tocarlos.
        let travel = launched ? speed * Game.dt : 0
        let parts = max(1, Int((travel / Game.maxTravelPerStep).rounded(.up)))
        let dt = Game.dt / Float(parts)
        for _ in 0..<parts {
            advance(move: move, dt: dt, &result)
            if over { break }
        }
        return result
    }

    private mutating func advance(move: Float, dt: Float, _ result: inout StepResult) {
        if wideTimer > 0 { wideTimer = max(0, wideTimer - dt) }
        if fireTimer > 0 { fireTimer = max(0, fireTimer - dt) }

        var half = paddleWidth / 2
        paddleX += max(-1, min(1, move)) * paddleSpeed * dt
        paddleX = max(half, min(Game.width - half, paddleX))

        if !launched {
            balls[0].x = paddleX
            balls[0].y = Game.paddleY - Game.ballR
            launchTimer -= dt
            if launchTimer <= 0 { launch() }
            return
        }

        // Potenciadores cayendo.
        var k = 0
        while k < capsules.count {
            capsules[k].y += Game.capsuleSpeed * max(1, speedScale) * dt
            let capsule = capsules[k]
            let top = capsule.y - Game.capsuleH / 2
            let bottom = capsule.y + Game.capsuleH / 2
            if bottom >= Game.paddleY, top <= Game.paddleY + Game.paddleH, abs(capsule.x - paddleX) <= half + Game.capsuleW / 2 {
                capsules.remove(at: k)
                apply(capsule.kind)
                result.powerUps += 1
                half = paddleWidth / 2
                continue
            }
            if top > Game.height {
                capsules.remove(at: k)
                continue
            }
            k += 1
        }

        let r = Game.ballR
        let bricksBottom = Game.brickTop + Float(Game.rows) * Game.brickH
        var i = 0
        while i < balls.count {
            var ball = balls[i]
            ball.x += ball.vx * dt
            ball.y += ball.vy * dt

            if ball.x < r {
                ball.x = r
                ball.vx = abs(ball.vx)
            } else if ball.x > Game.width - r {
                ball.x = Game.width - r
                ball.vx = -abs(ball.vx)
            }
            if ball.y < r {
                ball.y = r
                ball.vy = abs(ball.vy)
            }

            // Pala: el ángulo de salida depende de dónde pega la bola.
            if ball.vy > 0,
               ball.y + r >= Game.paddleY,
               ball.y <= Game.paddleY + Game.paddleH,
               abs(ball.x - paddleX) <= half + r
            {
                let offset = max(-1, min(1, (ball.x - paddleX) / half))
                let angle = offset * Game.maxBounceAngle
                ball.vx = speed * sin(angle)
                ball.vy = -speed * cos(angle)
                ball.y = Game.paddleY - r
                result.paddleHit = true
            }

            if ball.y - r > Game.height {
                balls.remove(at: i)
                continue
            }
            balls[i] = ball

            if ball.y - r < bricksBottom, ball.y + r > Game.brickTop {
                // Nivel superado: se ha montado el siguiente y solo queda la bola del saque.
                if collideBricks(i, &result) { return }
            }
            i += 1
        }

        // Solo se pierde vida cuando se escapa la última bola.
        if balls.isEmpty {
            lives -= 1
            result.lostLife = true
            if lives <= 0 {
                over = true
            }
            resetBall()
        }
    }

    /// Choque de una bola con los ladrillos. Devuelve `true` si con ese golpe se ha superado el nivel.
    private mutating func collideBricks(_ ballIndex: Int, _ result: inout StepResult) -> Bool {
        var ball = balls[ballIndex]
        let r = Game.ballR
        let c0 = max(0, Int((ball.x - r - Game.brickLeft) / Game.brickW))
        let c1 = min(Game.cols - 1, Int((ball.x + r - Game.brickLeft) / Game.brickW))
        let r0 = max(0, Int((ball.y - r - Game.brickTop) / Game.brickH))
        let r1 = min(Game.rows - 1, Int((ball.y + r - Game.brickTop) / Game.brickH))
        if c0 > c1 || r0 > r1 { return false }

        var best = -1
        var bestD2 = r * r
        var nx: Float = 0
        var ny: Float = 0
        for row in r0...r1 {
            for col in c0...c1 {
                let index = row * Game.cols + col
                if bricks[index] == 0 { continue }
                let x0 = Game.brickLeft + Float(col) * Game.brickW
                let y0 = Game.brickTop + Float(row) * Game.brickH
                let cx = max(x0, min(x0 + Game.brickW, ball.x))
                let cy = max(y0, min(y0 + Game.brickH, ball.y))
                let dx = ball.x - cx
                let dy = ball.y - cy
                let d2 = dx * dx + dy * dy
                if d2 < bestD2 {
                    best = index
                    bestD2 = d2
                    nx = dx
                    ny = dy
                }
            }
        }
        if best < 0 { return false }

        if fireTimer > 0 {
            // Bola de fuego: no rebota y rompe el ladrillo de un golpe.
            bricks[best] = 0
        } else {
            if nx == 0 && ny == 0 {
                ball.vy = -ball.vy
            } else if abs(nx) > abs(ny) {
                ball.vx = nx > 0 ? abs(ball.vx) : -abs(ball.vx)
            } else {
                ball.vy = ny > 0 ? abs(ball.vy) : -abs(ball.vy)
            }
            balls[ballIndex] = ball
            bricks[best] -= 1
        }
        score += 10
        result.bricksHit += 1
        if bricks[best] == 0 {
            bricksLeft -= 1
            if bricksLeft > 0, capsules.count < Game.maxCapsules, rng.uniform() < Game.dropChance {
                let col = best % Game.cols
                let row = best / Game.cols
                capsules.append(Capsule(
                    x: Game.brickLeft + (Float(col) + 0.5) * Game.brickW,
                    y: Game.brickTop + (Float(row) + 0.5) * Game.brickH,
                    kind: randomPowerUp()
                ))
            }
        }

        if bricksLeft == 0 {
            score += 100
            level += 1
            result.levelCleared = true
            buildLevel()
            resetBall()
            return true
        }
        updateSpeed()
        return false
    }

    // MARK: - Potenciadores

    private mutating func randomPowerUp() -> PowerUp {
        let roll = rng.uniform()
        if roll < 0.35 { return .multiball }
        if roll < 0.65 { return .wide }
        if roll < 0.90 { return .fire }
        return .life
    }

    private mutating func apply(_ kind: PowerUp) {
        switch kind {
        case .multiball:
            // De cada bola salen dos más, abiertas en abanico, hasta el máximo.
            let existing = balls.count
            for index in 0..<existing {
                for turn in [Float(0.4), -0.4] where balls.count < Game.maxBalls {
                    let source = balls[index]
                    let c = cos(turn)
                    let s = sin(turn)
                    var copy = source
                    copy.vx = source.vx * c - source.vy * s
                    copy.vy = source.vx * s + source.vy * c
                    balls.append(copy)
                }
            }
            updateSpeed()
        case .wide:
            wideTimer = Game.wideSeconds
        case .fire:
            fireTimer = Game.fireSeconds
        case .life:
            lives = min(Game.maxLives, lives + 1)
        }
    }

    // MARK: - Bola

    /// Multiplicador manual de la velocidad de la bola (1 = la normal del juego).
    public private(set) var speedScale: Float = 1

    /// Velocidad actual de la pala: se acelera junto con la bola, para que siga siendo posible llegar.
    public var paddleSpeed: Float { Game.paddleSpeed * max(1, speedScale) }

    /// Cambia el multiplicador y lo aplica ya a las bolas en vuelo.
    public mutating func setSpeedScale(_ scale: Float) {
        let clamped = max(0.5, min(Game.maxSpeedScale, scale))
        let factor = clamped / speedScale
        speedScale = clamped
        speed *= factor
        for index in balls.indices {
            balls[index].vx *= factor
            balls[index].vy *= factor
        }
    }

    private func levelSpeed() -> Float {
        min(1.3, Game.baseSpeed + 0.05 * Float(level - 1))
    }

    /// La bola acelera según se vacía el nivel.
    private mutating func updateSpeed() {
        let cleared = 1 - Float(bricksLeft) / Float(max(1, bricksTotal))
        speed = min(Game.maxSpeed, levelSpeed() * (1 + 0.25 * cleared)) * speedScale
        let minVY = Game.minVerticalFraction * speed
        for index in balls.indices {
            var ball = balls[index]
            let current = (ball.vx * ball.vx + ball.vy * ball.vy).squareRoot()
            if current > 0 {
                ball.vx *= speed / current
                ball.vy *= speed / current
            }
            // Evita trayectorias casi horizontales que no acaban nunca.
            if abs(ball.vy) < minVY {
                ball.vy = ball.vy < 0 ? -minVY : minVY
                let vx = (speed * speed - minVY * minVY).squareRoot()
                ball.vx = ball.vx < 0 ? -vx : vx
            }
            balls[index] = ball
        }
    }

    /// Vuelve al saque: una bola pegada a la pala, sin potenciadores cayendo ni efectos activos.
    private mutating func resetBall() {
        launched = false
        launchTimer = Game.launchDelay
        speed = levelSpeed() * speedScale
        balls.removeAll(keepingCapacity: true)
        balls.append(Ball(x: paddleX, y: Game.paddleY - Game.ballR, vx: 0, vy: 0))
        capsules.removeAll(keepingCapacity: true)
        wideTimer = 0
        fireTimer = 0
    }

    private mutating func launch() {
        let angle = (rng.uniform() * 2 - 1) * 35 * Float.pi / 180
        balls[0].vx = speed * sin(angle)
        balls[0].vy = -speed * cos(angle)
        launched = true
    }

    // MARK: - Niveles

    private mutating func buildLevel() {
        let cols = Game.cols
        let rows = Game.rows
        var grid = [UInt8](repeating: 0, count: cols * rows)
        let pattern = (level - 1 + patternOffset) % Game.patternCount
        switch pattern {
        case 0: // Muro: seis filas, las dos de arriba duras
            for row in 0..<6 {
                for col in 0..<cols { grid[row * cols + col] = row < 2 ? 2 : 1 }
            }
        case 1: // Damero
            for row in 0..<rows {
                for col in 0..<cols where (row + col) % 2 == 0 { grid[row * cols + col] = 1 }
            }
        case 2: // Pirámide invertida, con el borde duro
            for row in 0..<5 {
                for col in row..<(cols - row) {
                    let edge = col == row || col == cols - row - 1
                    grid[row * cols + col] = edge ? 2 : 1
                }
            }
        case 3: // Columnas
            for row in 0..<rows {
                for col in 0..<cols where col % 3 != 2 { grid[row * cols + col] = col % 3 == 0 ? 2 : 1 }
            }
        case 4: // Marco con núcleo
            for row in 0..<rows {
                for col in 0..<cols {
                    let frame = row == 0 || row == rows - 1 || col == 0 || col == cols - 1
                    let core = (3...4).contains(row) && (3...6).contains(col)
                    if frame { grid[row * cols + col] = 1 }
                    if core { grid[row * cols + col] = 2 }
                }
            }
        default: // Aleatorio y simétrico
            for row in 0..<rows {
                for col in 0..<(cols / 2) {
                    let roll = rng.uniform()
                    let hp: UInt8 = roll < 0.4 ? 0 : (roll < 0.85 ? 1 : 2)
                    grid[row * cols + col] = hp
                    grid[row * cols + (cols - 1 - col)] = hp
                }
            }
            if !grid.contains(where: { $0 > 0 }) { grid[0] = 1 }
        }
        bricks = grid
        bricksLeft = grid.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }
        bricksTotal = bricksLeft
    }

    // MARK: - Lo que ve la red

    /// Escribe el estado normalizado en `out` (`observationSize` valores): la pala y los efectos,
    /// las cuatro bolas que antes van a llegar abajo, los dos potenciadores más bajos y los ladrillos.
    public func observe(into out: UnsafeMutablePointer<Float>) {
        out[0] = paddleX / Game.width * 2 - 1
        out[1] = paddleWidth / Game.paddleW - 1
        out[2] = launched ? 1 : 0
        out[3] = Float(lives) / Float(Game.startLives)
        out[4] = speed / Game.maxSpeed
        out[5] = Float(bricksLeft) / Float(max(1, bricksTotal))
        out[6] = speedScale - 1
        out[7] = Float(balls.count) / Float(Game.maxBalls)
        out[8] = wideTimer / Game.wideSeconds
        out[9] = fireTimer / Game.fireSeconds

        // Bolas, de la más urgente a la menos.
        var used: UInt32 = 0
        var k = 10
        for _ in 0..<Game.seenBalls {
            var best = -1
            var bestTime = Float.infinity
            for index in balls.indices where used & (1 << UInt32(index)) == 0 {
                let time = timeToPaddle(balls[index])
                if time < bestTime {
                    best = index
                    bestTime = time
                }
            }
            if best >= 0 {
                used |= 1 << UInt32(best)
                let ball = balls[best]
                out[k] = 1
                out[k + 1] = ball.x / Game.width * 2 - 1
                out[k + 2] = ball.y / Game.height * 2 - 1
                out[k + 3] = ball.vx / Game.maxSpeed
                out[k + 4] = ball.vy / Game.maxSpeed
                out[k + 5] = (ball.x - paddleX) * 2
            } else {
                for j in 0..<6 { out[k + j] = 0 }
            }
            k += 6
        }

        // Potenciadores, del más bajo al más alto.
        used = 0
        for _ in 0..<Game.seenCapsules {
            var best = -1
            var bestY = -Float.infinity
            for index in capsules.indices where used & (1 << UInt32(index)) == 0 && capsules[index].y > bestY {
                best = index
                bestY = capsules[index].y
            }
            for j in 0..<8 { out[k + j] = 0 }
            if best >= 0 {
                used |= 1 << UInt32(best)
                let capsule = capsules[best]
                out[k] = 1
                out[k + 1] = capsule.x / Game.width * 2 - 1
                out[k + 2] = capsule.y / Game.height * 2 - 1
                out[k + 3] = (capsule.x - paddleX) * 2
                out[k + 4 + Int(capsule.kind.rawValue)] = 1
            }
            k += 8
        }

        for i in 0..<(Game.cols * Game.rows) {
            out[k + i] = Float(bricks[i]) * 0.5
        }
    }
}
