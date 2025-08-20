import argparse
import subprocess

filename = 'trainModel.py'  # no leading slash, so it's relative to current folder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    master = "localhost"
    port = "7669"
    num_nodes = 1
    rank = 0  # Single machine = rank 0

    print(f"Running on local machine with {num_nodes} node(s)")

    # Launch trainModel.py locally
    subprocess.run(
        f'python {filename} -m {master} -p {port} -n {num_nodes} -g 1 -nr {rank} --epochs {args.epochs}',
        shell=True
    )

if __name__ == '__main__':
    main()
