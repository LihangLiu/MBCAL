MBCAL

# Run
For all exp, firstly run:

	cd app/demo_credit_assignment	
## Movielens
Train Evaluators

	sh movielens/ML-simB train 
Train UniRNN

	sh movielens/ML-click train
	sh movielens/ML-rate train
Generate Credits (gt_base, gt, follow_click, gt_globbase), credit_gamma=0.9

	sh movielens/ML-click credit
	sh movielens/ML-rate credit
Train Credits

	sh movielens/ML-credit-click-gt_base train 			# credit_scale = 1.0
	sh movielens/ML-credit-click-gt train 				# 0.01
	sh movielens/ML-credit-click-follow_click train 	# 0.01
	sh movielens/ML-credit-click-gt_globbase train 		# 0.1
	sh movielens/ML-credit-rate-gt_base train 			# 1.0
	sh movielens/ML-credit-rate-gt train 				# 0.01
	sh movielens/ML-credit-rate-follow_click train  	# 0.01
Train RL, gamma=0.9

	sh movielens/ML-rl-q_learning train
	sh movielens/ML-rl-sarsa train
Evaluate, will output to logs/ML-eval-*

	sh movielens/ML-interactive_train parallel_eval


# 20191010

Credit variance
Download trained env model from V100 and then run:

	sh feedgr_duration_large/FGDL-OL-env-click-run0 credit_variance
