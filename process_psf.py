import numpy as np
import tifffile
import argparse


def main():
	parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--input', type = str, required = True)
	parser.add_argument('--output', type = str, required = True)
	parser.add_argument('--sigma', type = float, default = 5)
	parser.add_argument('--bg', type = float, default = 100)
	args = parser.parse_args()
	
	psf = tifffile.imread(args.input, dtype=np.float32)
	psf = psf - args.bg
	psf[psf < 0] = 0

	# Add new z-axis if we have 2D data
	if psf.ndim == 2:
		psf = np.expand_dims(psf, axis=0)

	x = np.linspace(1, psf.shape[1], psf.shape[1])
	x = x - psf.shape[1] / 2
	y = np.linspace(1, psf.shape[2], psf.shape[2])
	y = y - psf.shape[2] / 2
	x, y = np.meshgrid(x, y)
	r = np.sqrt(np.power(x, 2) + np.power(y, 2))

	filter = np.exp(-np.power(r, 2) / (2 * args.sigma**2))
	filter = filter / np.max(filter)

	psf = psf * filter
	tifffile.imwrite(args.output, psf)


if __name__ == '__main__':
	main()
