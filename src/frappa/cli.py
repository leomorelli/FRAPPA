import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="frappa", description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True, help="folder with the TIFF stacks")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output folder")
    args = parser.parse_args()

    files = []
    for p in args.input.iterdir():
        if p.suffix.lower() in (".tif", ".tiff"):
            files.append(p)
    files = sorted(files)

    print(f"{len(files)} TIFF file(s) in {args.input}:")
    for f in files:
        print("  ", f.name)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Results will be written to {args.output}")


if __name__ == "__main__":
    main()