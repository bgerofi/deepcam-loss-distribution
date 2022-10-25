import pandas as pd

df = None

for i in range(512):
	rank_loss_df = pd.read_feather("losses-rank-%04d.feather" % i)
	df = pd.concat([df, rank_loss_df], ignore_index=True)
	print("{} processed..".format(i))

df.to_feather("deepcam-losses.feather")
