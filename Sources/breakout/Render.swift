import AppKit
import BreakoutCore

/// Textos que se pintan encima del campo.
struct HUD {
    var pilot = "TÚ"
    var best = 0
    var paused = false
    var note: String? = nil
}

private func color(_ hex: UInt32, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(
        red: CGFloat((hex >> 16) & 0xFF) / 255,
        green: CGFloat((hex >> 8) & 0xFF) / 255,
        blue: CGFloat(hex & 0xFF) / 255,
        alpha: alpha
    )
}

private let rowColors: [UInt32] = [
    0xFF5D73, 0xFF8A4C, 0xFFC145, 0xB8D94A, 0x4FD08A, 0x3CC6D6, 0x5B9BFF, 0xA07BFF,
]

private func drawText(_ text: String, at point: CGPoint, size: CGFloat, hex: UInt32, alpha: CGFloat = 1, align: NSTextAlignment = .left) {
    let font = NSFont.monospacedSystemFont(ofSize: size, weight: .bold)
    let attributes: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: NSColor(cgColor: color(hex, alpha)) ?? .white,
    ]
    let string = NSAttributedString(string: text, attributes: attributes)
    let width = string.size().width
    var origin = point
    if align == .center { origin.x -= width / 2 }
    if align == .right { origin.x -= width }
    string.draw(at: origin)
}

/// Color y letra de cada potenciador.
private func powerUpLook(_ kind: PowerUp) -> (UInt32, String) {
    switch kind {
    case .multiball: return (0xA07BFF, "x3")
    case .wide: return (0x7FE3FF, "<>")
    case .fire: return (0xFF8A4C, "F")
    case .life: return (0xFF5D73, "+1")
    }
}

/// Pinta la partida en el contexto actual (coordenadas con la Y hacia abajo).
func render(game: Game, hud: HUD, in bounds: CGRect) {
    guard let ctx = NSGraphicsContext.current?.cgContext else { return }

    ctx.setFillColor(color(0x070A14))
    ctx.fill(bounds)

    let scale = min(bounds.width / CGFloat(Game.width), bounds.height / CGFloat(Game.height))
    let fieldW = CGFloat(Game.width) * scale
    let fieldH = CGFloat(Game.height) * scale
    let ox = bounds.midX - fieldW / 2
    let oy = bounds.midY - fieldH / 2
    let field = CGRect(x: ox, y: oy, width: fieldW, height: fieldH)

    func rect(_ x: Float, _ y: Float, _ w: Float, _ h: Float) -> CGRect {
        CGRect(x: ox + CGFloat(x) * scale, y: oy + CGFloat(y) * scale, width: CGFloat(w) * scale, height: CGFloat(h) * scale)
    }

    // Fondo del campo
    ctx.saveGState()
    ctx.clip(to: field)
    let gradient = CGGradient(
        colorsSpace: CGColorSpaceCreateDeviceRGB(),
        colors: [color(0x0E1430), color(0x0A0E1F)] as CFArray,
        locations: [0, 1]
    )!
    ctx.drawLinearGradient(gradient, start: CGPoint(x: ox, y: oy), end: CGPoint(x: ox, y: oy + fieldH), options: [])

    // Ladrillos
    let gap = 0.004 * scale
    let radius = 0.006 * scale
    for row in 0..<Game.rows {
        for col in 0..<Game.cols {
            let hp = game.bricks[row * Game.cols + col]
            if hp == 0 { continue }
            let cell = rect(
                Game.brickLeft + Float(col) * Game.brickW,
                Game.brickTop + Float(row) * Game.brickH,
                Game.brickW,
                Game.brickH
            ).insetBy(dx: gap / 2, dy: gap / 2)
            let path = CGPath(roundedRect: cell, cornerWidth: radius, cornerHeight: radius, transform: nil)
            ctx.addPath(path)
            ctx.setFillColor(color(rowColors[row % rowColors.count], hp >= 2 ? 1 : 0.78))
            ctx.fillPath()
            if hp >= 2 {
                // Ladrillo duro: marco interior claro
                let inner = cell.insetBy(dx: 0.006 * scale, dy: 0.006 * scale)
                ctx.addPath(CGPath(roundedRect: inner, cornerWidth: radius / 2, cornerHeight: radius / 2, transform: nil))
                ctx.setStrokeColor(color(0xFFFFFF, 0.75))
                ctx.setLineWidth(max(1, 0.003 * scale))
                ctx.strokePath()
            }
        }
    }

    // Pala
    let paddleWidth = game.paddleWidth
    let paddle = rect(game.paddleX - paddleWidth / 2, Game.paddleY, paddleWidth, Game.paddleH)
    ctx.addPath(CGPath(roundedRect: paddle, cornerWidth: paddle.height / 2, cornerHeight: paddle.height / 2, transform: nil))
    ctx.setFillColor(color(game.wideTimer > 0 ? 0x7FE3FF : 0xE8ECFF))
    ctx.fillPath()

    // Potenciadores cayendo: una pastilla de color con su letra.
    for capsule in game.capsules {
        let (hex, letter) = powerUpLook(capsule.kind)
        let pill = rect(capsule.x - Game.capsuleW / 2, capsule.y - Game.capsuleH / 2, Game.capsuleW, Game.capsuleH)
        ctx.addPath(CGPath(roundedRect: pill, cornerWidth: pill.height / 2, cornerHeight: pill.height / 2, transform: nil))
        ctx.setFillColor(color(hex))
        ctx.fillPath()
        drawText(letter, at: CGPoint(x: pill.midX, y: pill.minY + pill.height * 0.08), size: pill.height * 0.72, hex: 0x070A14, align: .center)
    }

    // Bolas, con halo (naranjas cuando son de fuego)
    let r = Game.ballR
    let fire = game.fireTimer > 0
    for ball in game.balls {
        ctx.setFillColor(color(fire ? 0xFF8A4C : 0xFFFFFF, fire ? 0.3 : 0.12))
        ctx.fillEllipse(in: rect(ball.x - r * 2.2, ball.y - r * 2.2, r * 4.4, r * 4.4))
        ctx.setFillColor(color(fire ? 0xFFC145 : 0xFFFFFF))
        ctx.fillEllipse(in: rect(ball.x - r, ball.y - r, r * 2, r * 2))
    }
    ctx.restoreGState()

    // Efectos activos: una barra que se vacía, justo encima de la pala.
    var effectY = Game.paddleY - 0.075
    for (seconds, total, hex, name) in [
        (game.wideTimer, Game.wideSeconds, UInt32(0x7FE3FF), "PALA ANCHA"),
        (game.fireTimer, Game.fireSeconds, UInt32(0xFF8A4C), "FUEGO"),
    ] where seconds > 0 {
        let bar = rect(0.03, effectY, 0.2, 0.008)
        ctx.setFillColor(color(0xFFFFFF, 0.12))
        ctx.fill(bar)
        ctx.setFillColor(color(hex))
        ctx.fill(CGRect(x: bar.minX, y: bar.minY, width: bar.width * CGFloat(seconds / total), height: bar.height))
        drawText(name, at: CGPoint(x: bar.maxX + 0.012 * scale, y: bar.minY - 0.008 * scale), size: 0.018 * scale, hex: hex)
        effectY -= 0.03
    }

    // Marcador
    let textSize = 0.032 * scale
    let top = oy + 0.035 * scale
    drawText("PUNTOS \(game.score)", at: CGPoint(x: ox + 0.03 * scale, y: top), size: textSize, hex: 0xE8ECFF)
    drawText("NIVEL \(game.level)", at: CGPoint(x: field.midX, y: top), size: textSize, hex: 0x8FA0D8, align: .center)
    let hearts = String(repeating: "●", count: max(0, game.lives))
    drawText(hearts, at: CGPoint(x: field.maxX - 0.03 * scale, y: top), size: textSize, hex: 0xFF5D73, align: .right)

    let small = 0.022 * scale
    let bottom = oy + fieldH - 0.04 * scale
    drawText(hud.pilot, at: CGPoint(x: ox + 0.03 * scale, y: bottom), size: small, hex: 0x8FA0D8)
    drawText("MEJOR \(hud.best)", at: CGPoint(x: field.maxX - 0.03 * scale, y: bottom), size: small, hex: 0x8FA0D8, align: .right)
    if let note = hud.note {
        drawText(note, at: CGPoint(x: field.midX, y: bottom), size: small, hex: 0xFFC145, align: .center)
    }

    // Avisos
    var title: String?
    var subtitle: String?
    if game.over {
        title = "FIN"
        subtitle = "\(game.score) puntos · R para volver a jugar"
    } else if hud.paused {
        title = "PAUSA"
        subtitle = "P para seguir · I cambia entre tú y la IA"
    }
    if let title {
        ctx.setFillColor(color(0x070A14, 0.72))
        ctx.fill(field)
        drawText(title, at: CGPoint(x: field.midX, y: field.midY - 0.07 * scale), size: 0.085 * scale, hex: 0xFFFFFF, align: .center)
        if let subtitle {
            drawText(subtitle, at: CGPoint(x: field.midX, y: field.midY + 0.05 * scale), size: 0.026 * scale, hex: 0x8FA0D8, align: .center)
        }
    }
}
