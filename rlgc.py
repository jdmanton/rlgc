#!/usr/bin/env python

# Richardson-Lucy deconvolution code, using gradient consensus to stop iterations locally
# James Manton, 2023
# jmanton@mrc-lmb.cam.ac.uk
#
# Developed in collaboration with Andy York (Calico), Jan Becker (Oxford) and Craig Russell (EMBL EBI)

import numpy as np
import cupy as cp
import timeit
import tifffile
import argparse

rng = np.random.default_rng()


def main():
	# Get input arguments
	parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--input', type = str, required = True)
	parser.add_argument('--psf', type = str, required = True)
	parser.add_argument('--output', type = str, required = True)
	parser.add_argument('--max_iters', type = int, default = 10)
	parser.add_argument('--reblurred', type = str, required = False)
	parser.add_argument('--process_psf', type = int, default = 1)
	parser.add_argument('--rl_output', type = str, required = False)
	parser.add_argument('--limit', type = float, default = 0.01)
	parser.add_argument('--max_delta', type = float, default = 0.01)
	parser.add_argument('--iters_output', type = str, required = False)
	parser.add_argument('--rl_iters_output', type = str, required = False)
	parser.add_argument('--updates_output', type = str, required = False)
	parser.add_argument('--blur_consensus', type = int, default = 1)
	args = parser.parse_args()

	# Load data
	image = tifffile.imread(args.input)

	# Add new z-axis if we have 2D data
	if image.ndim == 2:
		image = np.expand_dims(image, axis=0)

	# Load and pad PSF if necessary
	psf_temp = tifffile.imread(args.psf)

	# Add new z-axis if we have 2D data
	if psf_temp.ndim == 2:
		psf_temp = np.expand_dims(psf_temp, axis=0)

	# if (args.process_psf):
		# print("Processing PSF...")
		# Take upper left 16x16 pixels to estimate noise level and create appropriate fake noise
		# noisy_region = psf_temp[0:16, 0:16, 0:16]
		# psf = np.random.normal(np.mean(noisy_region), np.std(noisy_region), image.shape)
	# else:
	
	psf = np.zeros(image.shape)
	psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp
	for axis, axis_size in enumerate(psf.shape):
		psf = np.roll(psf, int(axis_size / 2), axis=axis)
	for axis, axis_size in enumerate(psf_temp.shape):
		psf = np.roll(psf, -int(axis_size / 2), axis=axis)
	psf = np.fft.ifftshift(psf)

	# if (args.process_psf):	
		# psf = psf - np.mean(noisy_region)
		# psf[psf < 0] = 0

	# psf = psf_temp

	psf = psf / np.sum(psf)

	# Load data and PSF onto GPU
	image = cp.array(image, dtype=cp.float32)
	psf = cp.array(psf, dtype=cp.float32)

	# Calculate OTF and transpose
	otf = cp.fft.fftn(psf)
	otfT = cp.conjugate(otf)

	# Log which files we're working with and the number of iterations
	print('')
	print('Input file: %s' % args.input)
	print('Input shape: %s' % (image.shape, ))
	print('PSF file: %s' % args.psf)
	print('PSF shape: %s' % (psf_temp.shape, ))
	print('Output file: %s' % args.output)
	print('Maximum number of iterations: %d' % args.max_iters)
	print('PSF processing: %s' % args.process_psf)
	print('')

	# Get dimensions of data
	num_z = image.shape[0]
	num_y = image.shape[1]
	num_x = image.shape[2]
	num_pixels = num_z * num_y * num_x

	# Calculate Richardson-Lucy iterations
	HTones = fftconv(cp.ones_like(image), otfT)
	recon = cp.mean(image) * cp.ones((num_z, num_y, num_x))
	recon_rl = cp.mean(image) * cp.ones((num_z, num_y, num_x))

	if (args.iters_output is not None):
		iters = np.zeros((args.max_iters, num_z, num_y, num_x))
	
	if (args.rl_iters_output is not None):
		rl_iters = np.zeros((args.max_iters, num_z, num_y, num_x))

	if (args.updates_output is not None):
		updates = np.zeros((args.max_iters, num_z, num_y, num_x))

	num_iters = 0
	for iter in range(args.max_iters):
		start_time = timeit.default_timer()

		# Split recorded image into 50:50 images
		# TODO: make this work on the GPU (for some reason, we get repeating blocks with a naive conversion to cupy)
		split1 = rng.binomial(image.get().astype('int64'), p=0.5)
		split1 = cp.array(split1)
		split2 = image - split1

		# Calculate prediction
		Hu = fftconv(recon, otf)

		# Calculate updates for split images and full images (H^T (d / Hu))
		ratio1 = split1 / (0.5 * (Hu + 1E-12))
		ratio2 = split2 / (0.5 * (Hu + 1E-12))
		HTratio1 = fftconv(ratio1, otfT)
		HTratio2 = fftconv(ratio2, otfT)
		ratio = image / (Hu + 1E-12)
		HTratio = fftconv(ratio, otfT)
		HTratio = HTratio / HTones

		# Normalise update steps by H^T(1) and only update pixels in full estimate where split updates agree in 'sign'
		update1 = HTratio1 / HTones
		update2 = HTratio2 / HTones
		if (args.blur_consensus != 0):
			shouldNotUpdate = fftconv(fftconv((update1 - 1) * (update2 - 1), otf), otfT) < 0
		else:
			shouldNotUpdate = (update1 - 1) * (update2 - 1) < 0
		HTratio[shouldNotUpdate] = 1

		# Save previous estimate to check we're not wasting our time updating small values
		previous_recon = recon

		# Update estimate
		recon = recon * HTratio

		# Add to full iterations output if asked to by user
		if (args.iters_output is not None):
			iters[iter, :, :, :] = recon.get()
		
		if (args.updates_output is not None):
			updates[iter, :, :, :] = HTratio.get()

		# Also calculate normal RL update if asked to by user
		if args.rl_output is not None:
			Hu_rl = fftconv(recon_rl, otf)
			ratio_rl = image / (Hu_rl + 1E-12)
			HTratio_rl = fftconv(ratio_rl, otfT)
			recon_rl = recon_rl * HTratio_rl / HTones
			if (args.rl_iters_output is not None):
				rl_iters[iter, :, :, :] = recon_rl.get()

		calc_time = timeit.default_timer() - start_time
		num_updated = num_pixels - cp.sum(shouldNotUpdate)
		max_relative_delta = cp.max((recon - previous_recon) / cp.max(recon))
		print("Iteration %03d completed in %1.3f s. %1.2f %% of image updated. Update range: %1.2f to %1.2f. Largest relative delta = %1.5f" % (iter + 1, calc_time, 100 * num_updated / num_pixels, cp.min(HTratio), cp.max(HTratio), max_relative_delta))

		num_iters = num_iters + 1

		if (num_updated / num_pixels < args.limit):
			print('Hit limit')
			break

		if (max_relative_delta < args.max_delta):
			print('Hit max delta')
			break

		if (max_relative_delta < 5 / cp.max(image)):
			print('Hit auto delta')
			break

	# Reblur, collect from GPU and save if argument given
	if args.reblurred is not None:
		reblurred = fftconv(recon, otf)
		reblurred = reblurred.get()
		tifffile.imwrite(args.reblurred, reblurred, bigtiff=True)

	# Collect reconstruction from GPU and save
	recon = recon.get()
	tifffile.imwrite(args.output, recon, bigtiff=True)

	# Save RL output if argument given
	if args.rl_output is not None:
		recon_rl = recon_rl.get()
		tifffile.imwrite(args.rl_output, recon_rl, bigtiff=True)
	
	# Save full iterations if argument given
	if (args.iters_output is not None):
		tifffile.imwrite(args.iters_output, iters[0:num_iters, :, :, :], bigtiff=True)
	
	# Save full RL iterations if argument given
	if (args.rl_iters_output is not None):
		tifffile.imwrite(args.rl_iters_output, rl_iters[0:num_iters, :, :, :], bigtiff=True)
	
	# Save full updates if argument given
	if (args.updates_output is not None):
		tifffile.imwrite(args.updates_output, updates, bigtiff=True)


def fftconv(x, H):
	return cp.real(cp.fft.ifftn(cp.fft.fftn(x) * H))


if __name__ == '__main__':
	main()
