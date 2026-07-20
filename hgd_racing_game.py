import sys

from realtime_hgd_4class import main


if "--game-lsl" not in sys.argv:
    sys.argv.append("--game-lsl")


if __name__ == "__main__":
    main()
