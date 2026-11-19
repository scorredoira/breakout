// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "breakout",
    platforms: [.macOS(.v13)],
    targets: [
        .target(name: "BreakoutCore"),
        .target(name: "Brain"),
        .executableTarget(name: "breakout", dependencies: ["BreakoutCore", "Brain"]),
    ]
)
