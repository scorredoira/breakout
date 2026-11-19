# Breakout

Breakout para macOS en Swift, con una IA (red neuronal entrenada con PPO) que puede jugar sola.

Requiere macOS 13 o posterior y Swift 5.9.

## Arrancar

Desde la carpeta del proyecto:

```sh
swift run -c release breakout          # juegas tú
swift run -c release breakout watch    # empieza jugando la IA
```

La IA usa `model.bin` de la carpeta actual. Para usar otro modelo, pásalo detrás:

```sh
swift run -c release breakout watch model-8min.bin
```

## Teclas

| Tecla | Qué hace |
|---|---|
| ← → o A D, o el ratón | mover la pala |
| Espacio o P | pausa (Espacio reinicia si la partida ha acabado) |
| R | reiniciar |
| I | cambiar entre jugar tú y ver a la IA |
| ↑ ↓ | bola más rápida o más lenta |
| 1, 2, 3 | velocidad a la que juega la IA (x1, x4, x16) |
| Esc o ⌘Q | salir |

## Entrenar la IA

```sh
swift run -c release breakout train --minutes 10
```

Sigue entrenando desde `model.bin` si existe, guarda la red en cada iteración y cada 25 iteraciones
añade una fila de evaluación a `train.csv`. Sin `--minutes` ni `--iters` no para hasta Ctrl+C.

| Opción | Por defecto |
|---|---|
| `--minutes N` | sin límite |
| `--iters N` | sin límite |
| `--threads N` | núcleos − 2 |
| `--model RUTA` | `model.bin` |
| `--log RUTA` | `train.csv` |
| `--lr X` | `3e-4` |
| `--gamma X` | `0.995` |
| `--fresh` | empieza una red nueva (sobrescribe el modelo y el log) |

## Otros comandos

```sh
swift run -c release breakout selftest                          # 20 partidas con la pala automática y comprobación de gradientes
swift run -c release breakout snapshot foto.png 20 model.bin   # juega 20 s sin ventana y guarda la imagen
```

## Código

- `Sources/BreakoutCore`: el juego, sin gráficos.
- `Sources/Brain`: la red neuronal y PPO.
- `Sources/breakout`: la app (ventana, dibujo, teclas), el entrenamiento y los comandos.
