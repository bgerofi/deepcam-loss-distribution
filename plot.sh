#!/bin/bash
python -u ./deepcam-loss_plotter.py --input deepcam-losses.feather --output deepcam-losses.pdf --logscale
python -u ./deepcam-loss_plotter.py --input deepcam-losses.feather --output deepcam-losses+max5perc.pdf --logscale --split 5
python -u ./deepcam-loss_plotter.py --input deepcam-losses.feather --output deepcam-losses+max2perc.pdf --logscale --split 2
python -u ./deepcam-loss_plotter.py --input deepcam-losses.feather --output deepcam-losses+max1perc.pdf --logscale --split 1
