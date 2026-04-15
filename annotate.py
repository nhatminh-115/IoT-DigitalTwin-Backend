import argparse
import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

NODES = ["M1", "M4", "M6", "M7", "M8", "M9", "M10", "M11"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate node positions on a campus image.")
    parser.add_argument("--image", required=True, help="Path to campus image (PNG/JPG)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    coords: dict[str, list[int]] = {}
    idx_state = [0]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(mpimg.imread(args.image))
    ax.set_title(f"Click node position [{NODES[0]}]  ({len(NODES)} nodes remaining)   |  Right click = skip (occluded node)")
    ax.axis("on")

    def _advance(node: str, skipped: bool) -> None:
        idx_state[0] += 1
        remaining = len(NODES) - idx_state[0]
        status = "SKIP" if skipped else "OK"
        print(f"  {node}: {status}")

        if remaining == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(coords, f, indent=2)
            visible = len(coords)
            print(f"Saved {visible}/{len(NODES)} nodes -> {args.output}")
            ax.set_title(f"Done! Saved {visible}/{len(NODES)} nodes -> {args.output}")
            fig.canvas.draw()
            plt.pause(1.5)
            plt.close(fig)
        else:
            next_node = NODES[idx_state[0]]
            ax.set_title(
                f"Click node position [{next_node}]  ({remaining} nodes remaining)"
                f"   |  Right click = skip (occluded node)"
            )
            fig.canvas.draw()

    def onclick(event: object) -> None:
        idx = idx_state[0]
        if idx >= len(NODES):
            return

        node = NODES[idx]

        if event.button == 3:  # right-click -> skip
            _advance(node, skipped=True)
            return

        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        coords[node] = [x, y]

        ax.plot(x, y, "ro", markersize=10)
        ax.annotate(
            node,
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )
        fig.canvas.draw()
        _advance(node, skipped=False)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
