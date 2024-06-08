import numpy as np
import tifffile
import argparse


def main():
	parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--input', type = str, required = True)
	parser.add_argument('--output', type = str, required = True)
	parser.add_argument('--max_signal', type = int, default = 1000)
	parser.add_argument('--bg', type = int, default = 100)
	parser.add_argument('--sigma', type = float, default = 5.0)
	args = parser.parse_args()
	
	psf = tifffile.imread(args.input, dtype=np.float32)
	psf = args.max_signal * (psf / np.max(psf))
	psf = np.random.poisson(psf)
	
	bg = np.round(np.random.normal(args.bg * np.ones_like(psf), args.sigma))
	
	psf = psf + bg
	tifffile.imwrite(args.output, psf)


if __name__ == '__main__':
	main()
