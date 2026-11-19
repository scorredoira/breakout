/// Referencia sencilla: una pala que sigue a la bola y le pega cada vez en un punto distinto,
/// para no dejarla rebotando en vertical para siempre.
public struct TrackerBot {
    private var rng: RNG
    private var offset: Float = 0

    public init(seed: UInt64) {
        rng = RNG(seed: seed)
        pickOffset()
    }

    public func move(_ game: Game) -> Float {
        (game.ballX + offset - game.paddleX) / (game.paddleSpeed * Game.dt)
    }

    public mutating func observe(_ result: StepResult) {
        if result.paddleHit || result.lostLife { pickOffset() }
    }

    private mutating func pickOffset() {
        offset = (rng.uniform() * 2 - 1) * 0.8 * Game.paddleW / 2
    }
}
