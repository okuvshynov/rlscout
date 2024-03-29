This is an in-progress attempt (with very slow progress) to find a solution to some game. 

Most likely candidates are Othello 8x8 or freestyle gomoku on 8x8 - 9x9 boards.

Rough idea is the following:
1. Use self-play Deep RL (think AlphaZero) to get a very strong model for the game
2. Use that model to provide a good ordering for minimax (e.g. alpha-beta pruning)
3. Use that model to estimate probability of cutoff and thus allow better scaling full distributed search.

### script to setup everything for lambda cloud instance

just pull everything we need.

```
wget -O ~/lambda_rlscout_setup.sh https://raw.githubusercontent.com/okuvshynov/rlscout/master/scripts/lambda_setup.sh && chmod +x ~/lambda_rlscout_setup.sh && ~/lambda_rlscout_setup.sh
```

## LIFO order notes

### Utilizing search to estimate value

in Alpha Zero etc. MCTS is used to 'refine' move predictions done by the model. In our case, if we want to make a decision on 'which branches to explore with full search' we need 
to get a refined value estimate + some uncertainty around it.

The procedure itself will be likely similar, but the output of the search would be the expected value for each move. 
There's a difference here: if we only care about 'which move is better' as we want to play as good as possible, we don't need to know if best move will result in win, draw or 'not losing as badly'. In case 
of distributing full search that's crucial piece of information - if we win no matter what, we need to go deeper. If we unlikely to get a cutoff, it makes sense to distribute the search at that level and explore.


### adding value head back

Just looking at probabilities is not very good idea. Let's say we have 2 good moves and both of them will lead to cutting of the search at this branch. There's no point exploring both of them.
What we need to do instead, is to have value model, evaluate value model for each potential move and see how likely we are to get a cutoff from a single move. If likely, avoid sampling at that branch. This is 
similar to younger sibling wait concept.

On the other hand, if all values are fairly low and unlikely to result in cutoff we might have to explore them all. Can we still make this decisions individually, without having to communicate between threads/processes?
Can processes easily come and go?



### Let's check distributed version this way:

0. Try allocate manually and see if we can get below 10 min on 4 cores.
1. Fix model id. We'll use same model for all tests.
2. Run single-thread version with no sampling. Record visited nodes for levels < N
3. Run multithreadded version with sampling up to level N. Check if we visit extra nodes, how many, how often do we duplicate the work


### Some simplification of transposition table

```
2023-05-18 12:45:55,491 1274.22
2023-05-18 12:45:55,491 4 completions 1
2023-05-18 12:45:55,491 5 completions 1
2023-05-18 12:45:55,491 6 completions 3
2023-05-18 12:45:55,491 7 completions 7
2023-05-18 12:45:55,491 8 completions 16
2023-05-18 12:45:55,491 9 completions 40
2023-05-18 12:45:55,491 10 completions 97
2023-05-18 12:45:55,491 11 completions 228
2023-05-18 12:45:55,491 12 completions 622
2023-05-18 12:45:55,491 13 completions 1347
2023-05-18 12:45:55,491 14 completions 3837
2023-05-18 12:45:55,491 15 completions 8142
2023-05-18 12:45:55,491 16 completions 23467
2023-05-18 12:45:55,491 17 completions 48399
2023-05-18 12:45:55,491 18 completions 136691
2023-05-18 12:45:55,491 19 completions 274457
2023-05-18 12:45:55,491 20 completions 744875
2023-05-18 12:45:55,491 21 completions 1442522
2023-05-18 12:45:55,491 22 completions 3892932
2023-05-18 12:45:55,491 23 completions 7576821
2023-05-18 12:45:55,491 24 completions 19049778
2023-05-18 12:45:55,491 25 completions 34786039
2023-05-18 12:45:55,491 26 completions 78708982
2023-05-18 12:45:55,491 27 completions 135727643
2023-05-18 12:45:55,491 28 completions 276409111
2023-05-18 12:45:55,492 29 completions 450768760
2023-05-18 12:45:55,492 30 completions 820965890
2023-05-18 12:45:55,492 31 completions 1227575655
2023-05-18 12:45:55,492 32 completions 2324190090
2023-05-18 12:45:55,492 33 completions 3793534084
2023-05-18 12:45:55,492 34 completions 6091000156
2023-05-18 12:45:55,492 35 completions 5332868884
2023-05-18 12:45:55,492 result = -4
2023-05-18 12:45:55,565 observed total visits = 20599739577 for model_id=450
```

We have more visits total, but is it faster.

A slightly stronger version for a later model  
```
2023-05-18 13:09:27,907 1265.31
2023-05-18 13:09:27,907 4 completions 1
2023-05-18 13:09:27,907 5 completions 1
2023-05-18 13:09:27,907 6 completions 3
2023-05-18 13:09:27,907 7 completions 7
2023-05-18 13:09:27,907 8 completions 16
2023-05-18 13:09:27,907 9 completions 39
2023-05-18 13:09:27,907 10 completions 98
2023-05-18 13:09:27,907 11 completions 228
2023-05-18 13:09:27,907 12 completions 601
2023-05-18 13:09:27,907 13 completions 1303
2023-05-18 13:09:27,907 14 completions 3694
2023-05-18 13:09:27,907 15 completions 8023
2023-05-18 13:09:27,907 16 completions 22997
2023-05-18 13:09:27,907 17 completions 48152
2023-05-18 13:09:27,907 18 completions 134984
2023-05-18 13:09:27,907 19 completions 273190
2023-05-18 13:09:27,907 20 completions 738358
2023-05-18 13:09:27,907 21 completions 1435688
2023-05-18 13:09:27,907 22 completions 3878329
2023-05-18 13:09:27,907 23 completions 7554368
2023-05-18 13:09:27,907 24 completions 19047877
2023-05-18 13:09:27,907 25 completions 34644251
2023-05-18 13:09:27,907 26 completions 78711906
2023-05-18 13:09:27,907 27 completions 134898704
2023-05-18 13:09:27,907 28 completions 275491781
2023-05-18 13:09:27,907 29 completions 446682037
2023-05-18 13:09:27,907 30 completions 814021902
2023-05-18 13:09:27,907 31 completions 1215116770
2023-05-18 13:09:27,907 32 completions 2298282805
2023-05-18 13:09:27,907 33 completions 3756818265
2023-05-18 13:09:27,907 34 completions 6024226652
2023-05-18 13:09:27,907 35 completions 5269704247
2023-05-18 13:09:27,907 result = -4
2023-05-18 13:09:27,981 observed total visits = 20381747277 for model_id=750
```


### to save on model evaluation, how do we do multiple searches in batches?

If we do parallel, we can aggregate somewhere. But cost of one evaluation might get too low and synchronization becomes a bottleneck.
Can we apply similar 'batched approach' as we do with MCTS? 


### evaluate model for larger ply

until level 22 (compared to normal 18):

```
2023-05-17 09:44:11,153 2188.62
2023-05-17 09:44:11,153 4 completions 1
2023-05-17 09:44:11,153 5 completions 1
2023-05-17 09:44:11,153 6 completions 3
2023-05-17 09:44:11,153 7 completions 7
2023-05-17 09:44:11,153 8 completions 16
2023-05-17 09:44:11,153 9 completions 40
2023-05-17 09:44:11,153 10 completions 97
2023-05-17 09:44:11,153 11 completions 228
2023-05-17 09:44:11,153 12 completions 622
2023-05-17 09:44:11,153 13 completions 1347
2023-05-17 09:44:11,153 14 completions 3837
2023-05-17 09:44:11,153 15 completions 8142
2023-05-17 09:44:11,153 16 completions 23464
2023-05-17 09:44:11,153 17 completions 48398
2023-05-17 09:44:11,153 18 completions 136696
2023-05-17 09:44:11,153 19 completions 274409
2023-05-17 09:44:11,153 20 completions 744914
2023-05-17 09:44:11,153 21 completions 1442261
2023-05-17 09:44:11,153 22 completions 3687602
2023-05-17 09:44:11,153 23 completions 6803939
2023-05-17 09:44:11,153 24 completions 16809653
2023-05-17 09:44:11,153 25 completions 30653028
2023-05-17 09:44:11,153 26 completions 69508888
2023-05-17 09:44:11,153 27 completions 117561664
2023-05-17 09:44:11,153 28 completions 237549549
2023-05-17 09:44:11,153 29 completions 366549060
2023-05-17 09:44:11,153 30 completions 645562301
2023-05-17 09:44:11,153 31 completions 926025030
2023-05-17 09:44:11,153 32 completions 1404648218
2023-05-17 09:44:11,153 33 completions 2268002377
2023-05-17 09:44:11,153 34 completions 3649539411
2023-05-17 09:44:11,153 35 completions 3190772340
2023-05-17 09:44:11,153 result = -4
2023-05-17 09:44:11,224 observed total visits = 12936357543 for model_id=450
```

We cut down the number of visits another 2x with the same model, but the wall time is larger due to model evaluation cost


until level 20 (compared to normal 18):

```
--
2023-05-17 10:14:20,988 1319.28
2023-05-17 10:14:20,988 4 completions 1
2023-05-17 10:14:20,988 5 completions 1
2023-05-17 10:14:20,988 6 completions 3
2023-05-17 10:14:20,988 7 completions 7
2023-05-17 10:14:20,988 8 completions 16
2023-05-17 10:14:20,988 9 completions 40
2023-05-17 10:14:20,988 10 completions 97
2023-05-17 10:14:20,988 11 completions 228
2023-05-17 10:14:20,988 12 completions 622
2023-05-17 10:14:20,988 13 completions 1347
2023-05-17 10:14:20,988 14 completions 3837
2023-05-17 10:14:20,988 15 completions 8142
2023-05-17 10:14:20,988 16 completions 23464
2023-05-17 10:14:20,988 17 completions 48398
2023-05-17 10:14:20,988 18 completions 136693
2023-05-17 10:14:20,988 19 completions 274399
2023-05-17 10:14:20,988 20 completions 744882
2023-05-17 10:14:20,988 21 completions 1442122
2023-05-17 10:14:20,988 22 completions 3892743
2023-05-17 10:14:20,988 23 completions 7574821
2023-05-17 10:14:20,988 24 completions 19085001
2023-05-17 10:14:20,988 25 completions 34953215
2023-05-17 10:14:20,988 26 completions 79606591
2023-05-17 10:14:20,988 27 completions 138237015
2023-05-17 10:14:20,988 28 completions 284147827
2023-05-17 10:14:20,988 29 completions 457332235
2023-05-17 10:14:20,988 30 completions 829880425
2023-05-17 10:14:20,988 31 completions 1237048836
2023-05-17 10:14:20,988 32 completions 1932278656
2023-05-17 10:14:20,988 33 completions 3149240499
2023-05-17 10:14:20,988 34 completions 5123533071
2023-05-17 10:14:20,988 35 completions 4514495245
2023-05-17 10:14:20,988 result = -4
2023-05-17 10:14:21,052 observed total visits = 17813990479 for model_id=450
```

### Distributed search - what's left?

We'll do lazy SMP here, and sample moves based on the model we trained. We need to communicate through transposition table.
Option 1 - we do it locally and make it thread-safe.
Option 2 - we do it in truly distributed way, and TT becomes a separate service.
Option 3 - combination of the first 2 options. For some layers have only 'local' TT and for others remote shared TT. 

We'll need sampled move ordering with temperature, where 0 would mean 'greedy' and 1 would mean 'sampled'.

We can simplify TT logic as we don't really care too much about absolute performance here as long as we can experiment in reasonable timeframe.


### new best for 6 residual blocks:

```
2023-05-16 14:01:45,033 1622.08
2023-05-16 14:01:45,033 4 completions 1
2023-05-16 14:01:45,033 5 completions 1
2023-05-16 14:01:45,033 6 completions 3
2023-05-16 14:01:45,033 7 completions 7
2023-05-16 14:01:45,033 8 completions 12
2023-05-16 14:01:45,033 9 completions 36
2023-05-16 14:01:45,033 10 completions 83
2023-05-16 14:01:45,033 11 completions 216
2023-05-16 14:01:45,033 12 completions 526
2023-05-16 14:01:45,033 13 completions 1267
2023-05-16 14:01:45,033 14 completions 3323
2023-05-16 14:01:45,033 15 completions 7785
2023-05-16 14:01:45,033 16 completions 20811
2023-05-16 14:01:45,033 17 completions 47034
2023-05-16 14:01:45,033 18 completions 124918
2023-05-16 14:01:45,033 19 completions 270744
2023-05-16 14:01:45,033 20 completions 749164
2023-05-16 14:01:45,033 21 completions 1640236
2023-05-16 14:01:45,033 22 completions 4305109
2023-05-16 14:01:45,033 23 completions 8887242
2023-05-16 14:01:45,033 24 completions 21522819
2023-05-16 14:01:45,033 25 completions 42219021
2023-05-16 14:01:45,033 26 completions 94011356
2023-05-16 14:01:45,033 27 completions 174697705
2023-05-16 14:01:45,033 28 completions 357152229
2023-05-16 14:01:45,033 29 completions 609709000
2023-05-16 14:01:45,033 30 completions 1112838075
2023-05-16 14:01:45,033 31 completions 1725506865
2023-05-16 14:01:45,033 32 completions 2714484810
2023-05-16 14:01:45,033 33 completions 4482127979
2023-05-16 14:01:45,033 34 completions 7310447932
2023-05-16 14:01:45,033 35 completions 6463968592
2023-05-16 14:01:45,033 result = -4
2023-05-16 14:01:45,117 observed total visits = 25124744901 for model_id=265

--
2023-05-17 08:05:43,460 1397.86
2023-05-17 08:05:43,460 4 completions 1
2023-05-17 08:05:43,460 5 completions 1
2023-05-17 08:05:43,460 6 completions 3
2023-05-17 08:05:43,460 7 completions 7
2023-05-17 08:05:43,460 8 completions 16
2023-05-17 08:05:43,460 9 completions 40
2023-05-17 08:05:43,460 10 completions 97
2023-05-17 08:05:43,460 11 completions 227
2023-05-17 08:05:43,460 12 completions 618
2023-05-17 08:05:43,460 13 completions 1338
2023-05-17 08:05:43,460 14 completions 3792
2023-05-17 08:05:43,460 15 completions 8077
2023-05-17 08:05:43,460 16 completions 23087
2023-05-17 08:05:43,460 17 completions 47907
2023-05-17 08:05:43,460 18 completions 134114
2023-05-17 08:05:43,460 19 completions 271007
2023-05-17 08:05:43,460 20 completions 781236
2023-05-17 08:05:43,460 21 completions 1622006
2023-05-17 08:05:43,460 22 completions 4420361
2023-05-17 08:05:43,460 23 completions 8736284
2023-05-17 08:05:43,460 24 completions 21839091
2023-05-17 08:05:43,460 25 completions 41334742
2023-05-17 08:05:43,460 26 completions 94294927
2023-05-17 08:05:43,460 27 completions 170451783
2023-05-17 08:05:43,460 28 completions 354420717
2023-05-17 08:05:43,460 29 completions 592760698
2023-05-17 08:05:43,460 30 completions 1095463528
2023-05-17 08:05:43,461 31 completions 1673598101
2023-05-17 08:05:43,461 32 completions 2659185481
2023-05-17 08:05:43,461 33 completions 4355345040
2023-05-17 08:05:43,461 34 completions 7141569074
2023-05-17 08:05:43,461 35 completions 6320806330
2023-05-17 08:05:43,461 result = -4
2023-05-17 08:05:43,523 observed total visits = 24537119731 for model_id=450
```


### Do we care about speed at all?

In a way, if we moved from 'solving game' to 'allowing alpha-beta search to be distributed' we don't care too much about how fast in seconds/cycles the process is.
This probably means we can simplify a few things. 

### How do we evaluate model quality wrt to alpha-beta search?

Just from ordering point of view, we can count total number of nodes visited.

Let's test the following:
1. For same number of residual layers, train model for N hours with/without model evaluation. Compare the quality.
2. Train models with 2-6 residual layers.
3. Implement distributed/parallel alpha-beta.
4. How number of visits and time depend on the NN layer cutoff.

### Let's focus on running everything on Mac M1/M2

With smaller models we have now we can skip GPU/TPU things for now and just run on HW we have.

1. Focus on just optimizing minimax
2. Time/Number of nodes as a function of model iteration
3. time/number of visits/total number of visits as a function of N threads/processes


### current issues 

```
[?] Figure out what's going on with multi-threading self-play. Seems like it works ok on OS X, but very slow on Linux - both Pi instances and Lambda in the cloud.
    [+] try different processes. Works great. What's wrong about threading implementation here?
[ ] shutdown time out after small test finished
[+] there might be a bug in incremental data update. Training gets insane errors after a while. Make some unit tests for it.
    [+] add consistent keys to samples. To make sure sample goes to the same partition (train or validation set), we assign random 64 bit int at write time. At read time, we split 64 bit int into 8 bit numbers for each symmetry
        and use that number to deterministically assign sample to training or validation set.
[+] random rollout becomes expensive. Do we improve that OR move back to value model? Improve for now
[ ] use all 8 symmetries to pick alpha-beta move or at least check how different are they
[ ] multi-GPU instances were crashing last time
[?] pass seed to rlslib library at init
    [+] for self-play
[+] we are not cleaning up database.
[ ] just focus on model
```

### Longer training pays off

Under 30 min for model_id=477 with 2 residual blocks.

```
2023-05-08 10:57:28,309 1717.5
2023-05-08 10:57:28,309 4 completions 1
2023-05-08 10:57:28,309 5 completions 1
2023-05-08 10:57:28,309 6 completions 3
2023-05-08 10:57:28,309 7 completions 7
2023-05-08 10:57:28,309 8 completions 16
2023-05-08 10:57:28,309 9 completions 42
2023-05-08 10:57:28,309 10 completions 101
2023-05-08 10:57:28,309 11 completions 240
2023-05-08 10:57:28,309 12 completions 633
2023-05-08 10:57:28,309 13 completions 1383
2023-05-08 10:57:28,309 14 completions 3884
2023-05-08 10:57:28,309 15 completions 8455
2023-05-08 10:57:28,309 16 completions 24218
2023-05-08 10:57:28,309 17 completions 51204
2023-05-08 10:57:28,309 18 completions 144388
2023-05-08 10:57:28,309 19 completions 296695
2023-05-08 10:57:28,309 20 completions 863351
2023-05-08 10:57:28,309 21 completions 1811213
2023-05-08 10:57:28,309 22 completions 4983449
2023-05-08 10:57:28,309 23 completions 9887937
2023-05-08 10:57:28,309 24 completions 24945758
2023-05-08 10:57:28,309 25 completions 47210427
2023-05-08 10:57:28,309 26 completions 108516117
2023-05-08 10:57:28,309 27 completions 195861136
2023-05-08 10:57:28,309 28 completions 409314894
2023-05-08 10:57:28,309 29 completions 683299246
2023-05-08 10:57:28,309 30 completions 1265915745
2023-05-08 10:57:28,309 31 completions 1934935541
2023-05-08 10:57:28,309 32 completions 3078880772
2023-05-08 10:57:28,309 33 completions 5045554668
2023-05-08 10:57:28,309 34 completions 8269799491
2023-05-08 10:57:28,309 35 completions 7324940896
```

### Future: implementing full search on GPU

Can we make a slightly-less efficient but easier adoptable to GPU version of AB search for the last few layers in the search space?
As we plan to do parallel/distributed search anyway, can we do many searches at once at the lower levels? How much wasted cycles will there be?

### Need to make entire thing reproducible

Let's make it deterministic by fixing seed.
Where do we need to do that?

1. Model initial weights
2. Move selection 
3. Sample selection
4. Sample split in sqlite

Current version supports that for single thread. We need separate seeds and rng per thread to support being reproducible.

We also need to have a 'reproducibility test' to make sure 2 runs result in identical outcomes.


### results for 2 blocks

```
2023-05-04 13:51:41,962 1924.73
2023-05-04 13:51:41,962 4 completions 1
2023-05-04 13:51:41,962 5 completions 1
2023-05-04 13:51:41,962 6 completions 3
2023-05-04 13:51:41,962 7 completions 7
2023-05-04 13:51:41,962 8 completions 16
2023-05-04 13:51:41,962 9 completions 42
2023-05-04 13:51:41,962 10 completions 100
2023-05-04 13:51:41,962 11 completions 237
2023-05-04 13:51:41,962 12 completions 641
2023-05-04 13:51:41,962 13 completions 1426
2023-05-04 13:51:41,962 14 completions 4148
2023-05-04 13:51:41,963 15 completions 9050
2023-05-04 13:51:41,963 16 completions 26623
2023-05-04 13:51:41,963 17 completions 56342
2023-05-04 13:51:41,963 18 completions 162576
2023-05-04 13:51:41,963 19 completions 333140
2023-05-04 13:51:41,963 20 completions 988559
2023-05-04 13:51:41,963 21 completions 2062078
2023-05-04 13:51:41,963 22 completions 5729220
2023-05-04 13:51:41,963 23 completions 11316601
2023-05-04 13:51:41,963 24 completions 28698014
2023-05-04 13:51:41,963 25 completions 54139375
2023-05-04 13:51:41,963 26 completions 124882199
2023-05-04 13:51:41,963 27 completions 224144835
2023-05-04 13:51:41,963 28 completions 469504672
2023-05-04 13:51:41,963 29 completions 777120438
2023-05-04 13:51:41,963 30 completions 1439417926
2023-05-04 13:51:41,963 31 completions 2180418124
2023-05-04 13:51:41,963 32 completions 3459818453
2023-05-04 13:51:41,963 33 completions 5647879987
2023-05-04 13:51:41,963 34 completions 9257971762
2023-05-04 13:51:41,963 35 completions 8186001888
```


### Notes after first tests

1. We are optimizing for winning a game, not 'minimize number of searches', which might be a little different objective, especially when taking into account transposition table. Let's plot a chart of number of nodes visited per model iteration.

2. AB search needs to work in more constrained environment. 

3. Self-play takes time and we need to allocate resources accordingly. What's the right way to allocate between training, self-play and evaluation?

### Smaller model testing

Testing much smaller model with 1 residual block using run_single_host. Is it still pretty good, getting solved 6x6 in 2817.94s after ~6-8h of self-play training. 
Let's also save the number of visits per level:

```
2817.94
4 completions 1
5 completions 1
6 completions 3
7 completions 7
8 completions 16
9 completions 39
10 completions 99
11 completions 238
12 completions 666
13 completions 1508
14 completions 4527
15 completions 10124
16 completions 30634
17 completions 66903
18 completions 197497
19 completions 417868
20 completions 1261545
21 completions 2679524
22 completions 7571763
23 completions 15084384
24 completions 39000247
25 completions 73820125
26 completions 173794540
27 completions 312279627
28 completions 666576712
29 completions 1102753640
30 completions 2079911894
31 completions 3148177652
32 completions 5087534461
33 completions 8269978135
34 completions 13661910047
35 completions 12136661237
```

As we'll be testing it on different HW, time will become meaningless. Number of visits per level would be a better metric to track.


model_id=72

```
2584.72
4 completions 1
5 completions 1
6 completions 3
7 completions 7
8 completions 16
9 completions 39
10 completions 95
11 completions 241
12 completions 618
13 completions 1508
14 completions 4165
15 completions 10086
16 completions 28204
17 completions 66235
18 completions 182531
19 completions 414940
20 completions 1179347
21 completions 2648001
22 completions 7084447
23 completions 14745886
24 completions 36335195
25 completions 71221906
26 completions 160803117
27 completions 296838184
28 completions 612246694
29 completions 1034191085
30 completions 1896300021
31 completions 2920686807
32 completions 4612049756
33 completions 7592654325
34 completions 12390480344
35 completions 10950869422
```

### Integration test

Can we do 'minimal' test? Something like 'play a few games', 'train small but better than random model', 'play a few games with new model again'?

Current version takes 10 min to run on Mac mini

### testing on fresh raspberry Pi
1. requirements are:
    - zmq
    - torch
    - numpy
    - matplotlib for some helper visualization scripts


### First version of model-guided alpha-beta search works good.

We end up with almost ~3x improvement even with very restricted evaluation (up to level 12): 3763.47s. 
Let's try different settings + MCTS on top.
Also try older (worse) model.
We get another improvement for using model up to 18 level (2190.68s) . At this state, however, we spend considerable resource on model evaluation. Can we batch this? Here we probably have to lock the thread and aggregate somewhere. 

Now that it works ok in principle, let's clean everything up.


### Testing/using model from alpha-beta

```
[x] Measure non-ordered a/b search for -5;-3: 
    8932.76s on Mac Mini M1
[ ] Create move ordering compile-time policy
[ ] Test with just model eval
[ ] Test with model + mcts
```

### What do we do next?

High level: confirm that everything works e2e on 6x6 board. 

```
[ ] Keep running current self-play procedure on Mac Mini;
[ ] Try larger model;
[x] Combine native code to single library
[ ] KV store for transposition table; What's the best option? Seemsl like we can start with whatever, and pick the right implementation after some experiments.
[ ] Distributed logging - use Kafka both for 'analytics-like' logging and sample logging; What do use to visualize analytics data?
[ ] Use our model in alpha-beta search. Check if we get any benefit of using model + search at lower levels.
[ ] Make unit tests/intergation tests
[ ] Create requirements.txt, check that it works on fresh instance
[ ] Make it work on multi-GPU instances
[ ] rename mctslib
[ ] handle shutdown signals
[ ] make each app a 'service'
```

### Current non-intrusive setup:
To run everything on the same host and keep number of models in sync with evaluation, we have a controller which pauses training if there are 2 or more not evaluated model snapshots;

### Logging from C

As we use C library from multiple python apps, it would be nice to control logging with the same python logging module.
To do that, we can build C library with some flag to indicate that logging should go to python lib.

### Current setup

1. M1 Mac Mini running self-play, data server and duel continously. 
2. Training model on laptop once in a while

### we are getting more services and need to organize them better.

1. Selfplay (can be 0-N)
2. Duel (can be 0-1)
3. Training (can be 0-1)
4. Duel vs a/b (can be ??)

### let's keep using random rollouts 

And more of them. This way we'll definitely get a better model and can test is as part of alpha-beta algo.

### Value prediction are pretty bad

What should we do:
1. Check the data - how often do we, for example, have duplicate input with different output? Our data is based on imperfect player doing some extra exploration, so that is definitely possible
2. Try optimize for value only and ignore action loss. 
3. See value error by ply
4. 


### Where exactly do we sample?

It is not entirely clear where do we sample and where do we add noise.
Re-reading the paper again, it seems like:
1. For self-play to generate the data we:
 - for first 30 moves sample from MCTS visit count to select a move to play
 - for next moves select move greedily (max MCTS visit count)
 - in addition to that, on each move selection we add Diriclet noise to root node on each move selection
 - in addition to that, we sample one of the 8 symmetries to evaluate the model

2. For model evaluation though, it is much less clear. What is the source of non-determinism? Only rotation?


### now we need the value part

1. Training on 1/-1/0 at first
2. What should be the loss?
3. Applying it?


### storing TT
HDD are pretty cheap, you can buy 22Tb for $400. something like $20/Tb.

1Tb == ~2-4B positions.


### logging value

Where do we join?
Let's say we have log lines like

game_id, board_a, board_b, probs, <>
then we separately log game_id -> result and join with all the samples.


### Training model v2 

There are two changes I'd like to do here:
1. actually add value head
2. Use full a/b at the latest stages to get better estimates from the start.

After we have a decent model, we can try using it to speed up a/b search.
We also need to have something to measure 'how decent model actually is'.


### optimizing to_flip
This is one of the most expensive operations, ~45% total CPU based on profiling.
We can avoid expensive some fill operations, especially at the later stages, by masking
the location. 

### TT zero window

Local TT will be overwritten so many times that it will be ok to optimize it for zero-window use-case.

### MCTS logic by ply

For self-play which we use to get the model, we can do a similar approach. The policies we need to handle are:
1. What do we do with move selection?
2. What do we do with value estimate at leaf.

For (1) we might have following options:
1. 


### Logic by ply

Assuming we have a model, we can do this:


For each level we can define two important policies:
1. Move selection policy
2. transposition table policy.

Move selection can be one of:
1. Action model with sampling + exploration noise 
2. Action model with sampling
3. Action model greedy
4. No model evaluation

Transposition table policy can be:
1. Shared, persistent TT
2. Local transposition table (what about canonical symmetry?)
3. No TT

Example policy selection could be:
1. Action model with sampling + exploration noise [4; 20)
2. Action model with sampling [20; 40)
3. Action model greedy [40; 55)
4. No model evaluation [55; inf]

1. Shared, persistent TT [4; 30)
2. Local transposition table [30; 60)
3. No TT [60; inf)

Actual values depend on various factors, including cost of model evaluation, model quality, cost of persistent storage and the compression used there.

This way we can start new a/b routine independently, even with different settings if we want to. One more level of granularity is local thread - for example, 'local TT' can still be shared among threads within same instance. We can also start 'greediest' a/b instance, which will always pick the top selection.

If we have that logic, can we keep getting new training data and refreshing the model? How different is it anyway compared to MCTS? What if we do that with null-window?

Basically: can we fill in transposition table AND train model at the same time? Do we need that at all? Just keep 2 separate processes? 

### applying model from A/B

We need to do that anyway, let's check how this could work.

We need to apply model:
1. for self-play to generate training data
2. for a/b search to get ordering
3. for duel to evaluate the model

Also, we'd like to do that:
1. on machines with nVidia cards, using TensorRT
2. on Apple M* using CoreML

The source of model is in pytorch.

### how to store TT?

```
    // for 0 window search we can further reduce size
    // for interval [a; a + 1] entry can be in 3 states:
    // [a; a] [a; a + 1], [a + 1, a + 1], so we need 2 bits for that
    // board itself is much larger space consumer though
    // naive way to save 4 bits would be to have bitboards be like this:
    // 1 - empty or not, with 4 center squares never being empty, so 32 bits
    // 2 - white 
    // other way to think about that - each square can be in 3 states, but we store 
    // 2 bits (thus, allowing for 4 states).
    // that should bring it down to 55 bits total
    // decoding might be slow, so we'll store that only for high enough levels

    // if we store with window = 2, how many states do we have to represent low/high?
    // [a; a+2] [a; a + 1] [a; a] [a + 1; a + 1] [a + 1; a + 2] [a+2; a+2]
    // 3 bits
    // definintely within 64 bit total.
```

What about larger 8*8 board? rather than storing 128 bit we can do the same thing, and get away with ~100 bits.
This will probably mean 'within 128 bit for entire state'. Also, if we allow to use part of value as a key, 
we can further reduce the size of each value. We can definitely do that for lower levels? 

Overall, we'll have 2 separate TT:
1. Shared, distributed, persistent. There we can optimize storage aggressively as it will be done on higher, more expensive to compute levels only
2. Local, shared between threads on the worker machines? Here we do care about processing speed, we just silently overwrite and probably can store part of the value as a key.

In case #2 for example, for 8x8 case the board will be 128 bits; everything else: player, skipped, scores - another 8 bit or so. depending on the size of TT (amount of RAM available) we can be more aggressive with storing part of key as value.

Would be great if we can, for example, get 63 bit for value + whatever bits for key and use 1 bit spin lock to synchronize. Seems unlikely though.


### null-window A/B search

Let's plan on making 0-window search. That will simplify things a little + we'll need to store less data.
If we can run it once, we can run it twice as well.


### what to do next and in what order?
For 6x6
1. Distributed/Serializable TT
2. A/B player to play against model/mcts/etc.
3. Introduce reading from TT to MCTS rollouts
4. Train a good model and use it for ordering in A/B
5. web UI to visualize games and player thoughts

### How to make distributed TT?

1. For each entry we at a minimum need to store:
- State itself (128 bit if silly)
- current player (1 bit)
- if there was a skip before (1 bit?) maybe have to be 2 bits to indicate end of game?
- alpha/beta. (4 bit each? depends on bounds we set)

2. We need a part of the table to be fully stored and distributed and other part can be just simple basic replacement local thing

3. How to compress/hash the board the most compact way? Othello has some properties, e.g. it is all connected in 2d.

4. Actual compression doesn't matter for now. 



### For better experimenting, we can do the following:
1. Have separate 'player' just for 1:1 games and visualizations. It won't be used for self-play generation, etc.
2. Implementations should be: full a/b, pure mcts, pure model, model mcts, etc.
3. Easy to see what each player thinks the next move should be

To do that we need to actually pick the best move? Current search procedure just assigns score.
Let's save first N layers of TT? 

We need to refactor A/B search so that it's more extensible and can be actual 'player'.

### Comparing M1 vs M2

Training:

M2
```
% python train_loop.py
loading last snapshot from DB: id=0
training on device mps
training on 1480640 recent samples
training loss: 3.598
validation loss: 3.598
.......... | 11.4 seconds | epoch 0: training loss: 3.426, validation loss: 3.425
.......... | 11.2 seconds | epoch 1: training loss: 3.273, validation loss: 3.275
.......... | 11.3 seconds | epoch 2: training loss: 3.115, validation loss: 3.115
.......... | 11.3 seconds | epoch 3: training loss: 2.942, validation loss: 2.941
```

M1
```
% python train_loop.py
loading last snapshot from DB: id=0
training on device mps
training on 1480640 recent samples
training loss: 3.609
validation loss: 3.615
.......... | 16.6 seconds | epoch 0: training loss: 3.402, validation loss: 3.398
.......... | 15.8 seconds | epoch 1: training loss: 3.267, validation loss: 3.261
.......... | 15.8 seconds | epoch 2: training loss: 3.147, validation loss: 3.146
.......... | 15.8 seconds | epoch 3: training loss: 3.031, validation loss: 3.026
```


### Individual components for entire thing:

1. Game Server/Database
2. Self-play loop
3. Evaluation/Duel loop
4. Model training loop
5. MCTS-AB Self-play loop
6. TT Library+Service+Backup

### distributed version: utilize MCTS

Parallelizing/distributing a/b search is not trivial, because the entire idea of the search 
procedure is to cut off branches (which we might have started in parallel).

Still, there are different approaches here. One thing we could do is:

First, we train a decent model through regular self-play. 
Next we start another slightly different self-play procedure: instead of collecting training data after level L we start full search and save result to global transposition table. 
Idea is to find nodes which are 'good enough' according to our model so that we might encounter them in full search.

We can start different instances and add some sampling for first N moves as usual.

Then we start actual full search (in a single thread) which uses same transposition table.

Part of the transposition table is global and shared across all nodes. Other part is local only (as communication/storage cost will be too high).

We can keep improving the model.

For example, in 6x6 case we complete 13-14 stones position in 1-2 seconds. This is ~22 levels remaining. Let's say for 8x8 case we'll do the same:
start full search at, say, 40 stones.

Maybe we should enqueue earlier to make each task larger? In 6x6 case at level 10 we spend ~30 seconds for each node. 

So we start N MCTS self-play processes, which stops game at 40 stones and passes the encountered position to full search queue. We do some exploration/sampling for first, say, 10-15 stones.
Gradually we can start full search at lower positions? We can do that right away? As long as we keep pushing data to shared TT. 
No, seems like we need to increase it gradually. Otherwise it is too hard to distribute and there'll be too much duplication? Still can be high enough, say, each task is 30 minutes? (level 6-7 for 6x6, should be fine for 35 for 8x8)

Enqueue checks that such state is not in the queue yet.

TT service.

TT consists of two parts:
* library. 
* remote service for sync

Library stores part globally (syncs to service) and other part locally (shared among threads but not among computation nodes).

We can configure:
* which layers are fully cached (no evictions)
* which layers are distributed
* which layers are cached locally
* some layers (e.g. last few) are not cached at all.

Example config:
1. Everything from 0..40 is fully cached and distributed. How much data is it? Likely terabytes.
2. At 35 we start full AB searches with small window.
3. 40-60 are cached locally.

If we start getting only a few new entries at level 35, start going to 34. Then, go to 33, etc?
Why do we need that if we still share TT? Do we get more/less duplication this way?


Generate even better data?
When we have this whole thing running we can also keep collecting training data.
For example, self-play which uses the transposition table we have in addition to the model? Or even full a/b search at some level?

Seems like we need to just reimplement everything in C++, except model training and maybe logging?

AB search - batching?

Let's say this takes a few years - are we ok with that?

How can we easily get help/add nodes?

Get new full search tasks, have access to shared transposition table, read/write? 

### ordering and TT

Now we don't do any smart ordering at all.
However, what we can easily do without using any extimates is to first check children in TT?

Not that many matches.

### Can we get bounds on how much the score could change in the last 1-2 moves?

```
8760.11
4 tt_hits 0 completions 1 cutoffs 0 evictions 0
5 tt_hits 3 completions 1 cutoffs 0 evictions 0
6 tt_hits 0 completions 3 cutoffs 2 evictions 0
7 tt_hits 0 completions 10 cutoffs 7 evictions 2
8 tt_hits 1 completions 23 cutoffs 14 evictions 12
9 tt_hits 0 completions 66 cutoffs 49 evictions 49
10 tt_hits 2 completions 194 cutoffs 139 evictions 177
11 tt_hits 16 completions 539 cutoffs 392 evictions 0
12 tt_hits 53 completions 1543 cutoffs 1135 evictions 0
13 tt_hits 136 completions 4154 cutoffs 2986 evictions 1
14 tt_hits 467 completions 11608 cutoffs 8587 evictions 2
15 tt_hits 1240 completions 30237 cutoffs 21622 evictions 27
16 tt_hits 3616 completions 83234 cutoffs 61781 evictions 220
17 tt_hits 10225 completions 209723 cutoffs 148579 evictions 1274
18 tt_hits 31421 completions 564169 cutoffs 418764 evictions 9351
19 tt_hits 83817 completions 1362264 cutoffs 955037 evictions 53491
20 tt_hits 238814 completions 3541440 cutoffs 2622360 evictions 344403
21 tt_hits 631009 completions 8100873 cutoffs 5604111 evictions 1653539
22 tt_hits 1697174 completions 20082961 cutoffs 14799625 evictions 8226528
23 tt_hits 4107851 completions 43276912 cutoffs 29475751 evictions 27315567
24 tt_hits 10014344 completions 102008512 cutoffs 74661344 evictions 83941809
25 tt_hits 22027528 completions 207367459 cutoffs 138754009 evictions 188182688
26 tt_hits 50358672 completions 463352577 cutoffs 336390752 evictions 441565793
27 tt_hits 104467393 completions 883555888 cutoffs 579243454 evictions 857574131
28 tt_hits 226774608 completions 1851753211 cutoffs 1330363650 evictions 1816206262
29 tt_hits 439859050 completions 3266311120 cutoffs 2085288119 evictions 3214931776
30 tt_hits 983365162 completions 6179288001 cutoffs 4386629128 evictions 6057265147
31 tt_hits 1600719362 completions 9785681504 cutoffs 6016289026 evictions 9599516612
32 tt_hits 3092104624 completions 16341646240 cutoffs 11143813367 evictions 16005662179
33 tt_hits 0 completions 27073774256 cutoffs 15580316417 evictions 0
34 tt_hits 0 completions 45652571509 cutoffs 27046025443 evictions 0
35 tt_hits 0 completions 11273103535 cutoffs 0 evictions 0
ll skip: 0 0 0 0
-4
```

For example, if we have 1 place left, the score can only change by 13 max (on 6x6 board)?
Therefore, if current score is too large/too small, we can skip applying last move.

13 is only for the corner. 
Any on the border is 11
Any in the interior is 10

Looks like it's working. What about next level:

For example, we have 2 free slots and know that there's at least one move. If player A makes the move, it will get at least 2 more points. After that, player B can make a move and can get up to 13 more points. So, player B can get at most 'now + 11'. We can check if that's guaranteed to be outside of alpha/beta.
What if we have 2 slots and both are valid moves? Can we get stricter limits? 

Can we get quick and stricter bound on the score increase?


What if we look 3 stones ahead? This time seems harder as each player might skip?

Some stats for 1 lookahead:
~12M samples at the level with 1 empty cell

```
format: 
count is_maximizing alpha beta score has_move 

486840 0 -5 -4 1 1
480261 0 -5 -4 3 1
437344 0 -5 -4 -1 1
426246 0 -5 -4 5 1
381894 0 -4 -3 3 1
361974 0 -4 -3 1 1
357977 0 -4 -3 5 1
354618 0 -5 -4 -3 1 # this is one example where we can skip applying move - any legal move will reduce the score by 2 or more
344732 0 -5 -4 7 1 # here depending on which cell is it, we might never get enough score to reach -5.
300374 0 -4 -3 7 1
300215 0 -4 -3 -1 1
259435 0 -5 -4 -5 1
256133 0 -5 -4 9 1
235978 0 -4 -3 9 1
222993 0 -4 -3 -3 1
176292 0 -5 -4 11 1
173923 0 -5 -4 -7 1
167930 0 -4 -3 11 1
150814 0 -4 -3 -5 1
...
```





### what boundary do we search at?

option 1: [-1; 1]. We'll know if it is a draw or not.
option 2: [-1; 0] or [0; 1]. Or both sequentially reusing transposition table.

Is there any implementation improvement (being faster, clearer, etc) in having 0 window?


### need to do hashing right

How to do that with rotation/canonical? Do we store all rotations and their hashes?
Do we compute it on the fly? (seems expensive?)
Or do we do that by level as well:
1. up to level X we use rotation and compute on the fly
2. after that we don't use rotation anyway and can store hash in the state (and update as we apply moves).


### even larger symmetry window + variable tt size

another 10% faster.
```
8962.58
4 tt_hits 0 completions 1 cutoffs 0 evictions 0
5 tt_hits 3 completions 1 cutoffs 0 evictions 0
6 tt_hits 0 completions 3 cutoffs 2 evictions 0
7 tt_hits 0 completions 10 cutoffs 7 evictions 2
8 tt_hits 1 completions 23 cutoffs 14 evictions 12
9 tt_hits 0 completions 66 cutoffs 49 evictions 49
10 tt_hits 2 completions 194 cutoffs 139 evictions 177
11 tt_hits 16 completions 539 cutoffs 392 evictions 0
12 tt_hits 55 completions 1520 cutoffs 1111 evictions 0
13 tt_hits 124 completions 4142 cutoffs 2988 evictions 1
14 tt_hits 470 completions 11568 cutoffs 8539 evictions 2
15 tt_hits 1240 completions 30314 cutoffs 21718 evictions 27
16 tt_hits 3641 completions 83397 cutoffs 61787 evictions 221
17 tt_hits 10279 completions 211156 cutoffs 149828 evictions 1319
18 tt_hits 31861 completions 568229 cutoffs 421242 evictions 9417
19 tt_hits 84861 completions 1378207 cutoffs 967321 evictions 54713
20 tt_hits 245162 completions 3581169 cutoffs 2647861 evictions 351778
21 tt_hits 644882 completions 8231377 cutoffs 5703665 evictions 1704513
22 tt_hits 1756508 completions 20367151 cutoffs 14978020 evictions 8437724
23 tt_hits 4236203 completions 44110268 cutoffs 30122178 evictions 28124880
24 tt_hits 10385110 completions 103646561 cutoffs 75650516 evictions 85779048
25 tt_hits 22801895 completions 211877839 cutoffs 142295466 evictions 193077454
26 tt_hits 52126345 completions 471711323 cutoffs 341309405 evictions 451071953
27 tt_hits 108211790 completions 904699793 cutoffs 595612418 evictions 881040070
28 tt_hits 235036158 completions 1888115516 cutoffs 1351367280 evictions 1859060936
29 tt_hits 456596090 completions 3347414167 cutoffs 2147739350 evictions 3309753502
30 tt_hits 1028316700 completions 6288962748 cutoffs 4444923170 evictions 6212154026
31 tt_hits 1670963701 completions 9990691747 cutoffs 6184370117 evictions 9891207090
32 tt_hits 3237618552 completions 16526272765 cutoffs 11204629761 evictions 16393132001
33 tt_hits 0 completions 27556864136 cutoffs 15977259843 evictions 0
34 tt_hits 0 completions 46325823889 cutoffs 27252091508 evictions 0
35 tt_hits 0 completions 11501629050 cutoffs 0 evictions 0
-4
```

### how to do multithreading?

Seems non-trivial. Maybe a specifically zero-window would be easier?

Well-known options are described in https://www.chessprogramming.org/Parallel_Search

One option which is similar to SMP is to run multiple MCTS games to get to 'promising' positions at somewhat lower levels, run full search from there and save it into transposition table. Full search from those 'good' positions can be parallelized. 




### implementing symmetries with delta swaps

Idea comes from https://www.chessprogramming.org/Flipping_Mirroring_and_Rotating#Diagonal, but different 
masks/shifts for 6x6 board.

We also do that in 3 shifts:
1. swap 3x3 squares with mask
```
000000
000000
000000
111000
111000
111000
``` 

2. swap 8 pairs with mask:
```
000000
100100
010010
000000
100100
010010
```

3. swap 4 one more pairs with 
```
000000
000000
100100
000000
000000
100100

```

After this we can find canonical representation at a higher levels efficiently, here's a test for 25 (same -3; -5 window):

```
9833.64
4 tt_hits 0 tt_rate 0 completions 1 cutoffs 0 evictions 0
5 tt_hits 3 tt_rate 1.78814e-05 completions 1 cutoffs 0 evictions 0
6 tt_hits 0 tt_rate 0 completions 3 cutoffs 2 evictions 0
7 tt_hits 0 tt_rate 0 completions 10 cutoffs 7 evictions 0
8 tt_hits 1 tt_rate 5.96046e-06 completions 23 cutoffs 14 evictions 0
9 tt_hits 0 tt_rate 0 completions 66 cutoffs 49 evictions 0
10 tt_hits 3 tt_rate 1.78814e-05 completions 193 cutoffs 138 evictions 0
11 tt_hits 13 tt_rate 7.7486e-05 completions 539 cutoffs 392 evictions 0
12 tt_hits 55 tt_rate 0.000327826 completions 1520 cutoffs 1111 evictions 0
13 tt_hits 124 tt_rate 0.000739098 completions 4142 cutoffs 2988 evictions 1
14 tt_hits 470 tt_rate 0.00280142 completions 11568 cutoffs 8539 evictions 2
15 tt_hits 1240 tt_rate 0.00739098 completions 30314 cutoffs 21718 evictions 27
16 tt_hits 3641 tt_rate 0.0217021 completions 83400 cutoffs 61790 evictions 221
17 tt_hits 10274 tt_rate 0.0612378 completions 211162 cutoffs 149831 evictions 1319
18 tt_hits 31856 tt_rate 0.189877 completions 568273 cutoffs 421285 evictions 9420
19 tt_hits 84814 tt_rate 0.505531 completions 1378329 cutoffs 967415 evictions 54715
20 tt_hits 245053 tt_rate 1.46063 completions 3581596 cutoffs 2648185 evictions 351832
21 tt_hits 644457 tt_rate 3.84126 completions 8232712 cutoffs 5704678 evictions 1704724
22 tt_hits 1755284 tt_rate 10.4623 completions 20371537 cutoffs 14981124 evictions 8438926
23 tt_hits 4234079 tt_rate 25.2371 completions 44121533 cutoffs 30129966 evictions 28128572
24 tt_hits 10381883 tt_rate 61.8808 completions 103674239 cutoffs 75668760 evictions 85790256
25 tt_hits 20406283 tt_rate 121.631 completions 214330701 cutoffs 144623407 evictions 195520648
26 tt_hits 44969949 tt_rate 268.042 completions 483541784 cutoffs 349543310 evictions 462924020
27 tt_hits 106610517 tt_rate 635.448 completions 934321669 cutoffs 613812724 evictions 910685953
28 tt_hits 236883686 tt_rate 1411.94 completions 1970929745 cutoffs 1412987991 evictions 1941741045
29 tt_hits 459561574 tt_rate 2739.2 completions 3547857290 cutoffs 2278670097 evictions 3509826700
30 tt_hits 955832329 tt_rate 5697.2 completions 6929345764 cutoffs 4889941367 evictions 6876734793
31 tt_hits 1708935701 tt_rate 10186.1 completions 11372841742 cutoffs 7078491053 evictions 11298241282
32 tt_hits 3255945757 tt_rate 19406.9 completions 19547282179 cutoffs 13268925850 evictions 19440230445
33 tt_hits 0 tt_rate 0 completions 32902336691 cutoffs 19171425961 evictions 0
34 tt_hits 0 tt_rate 0 completions 55536191497 cutoffs 32643799862 evictions 0
35 tt_hits 0 tt_rate 0 completions 14050239757 cutoffs 0 evictions 0
-4
```

### search with [-5; -3] window

```
13642.4
4 tt_hits 0 tt_rate 0 completions 1 cutoffs 0 evictions 0
5 tt_hits 3 tt_rate 1.78814e-05 completions 1 cutoffs 0 evictions 0
6 tt_hits 0 tt_rate 0 completions 3 cutoffs 2 evictions 0
7 tt_hits 0 tt_rate 0 completions 10 cutoffs 7 evictions 0
8 tt_hits 1 tt_rate 5.96046e-06 completions 23 cutoffs 14 evictions 0
9 tt_hits 0 tt_rate 0 completions 66 cutoffs 49 evictions 0
10 tt_hits 3 tt_rate 1.78814e-05 completions 193 cutoffs 138 evictions 0
11 tt_hits 13 tt_rate 7.7486e-05 completions 539 cutoffs 392 evictions 0
12 tt_hits 55 tt_rate 0.000327826 completions 1520 cutoffs 1111 evictions 0
13 tt_hits 124 tt_rate 0.000739098 completions 4142 cutoffs 2988 evictions 1
14 tt_hits 470 tt_rate 0.00280142 completions 11568 cutoffs 8539 evictions 2
15 tt_hits 1088 tt_rate 0.00648499 completions 30466 cutoffs 21860 evictions 29
16 tt_hits 3011 tt_rate 0.017947 completions 84380 cutoffs 62523 evictions 229
17 tt_hits 10318 tt_rate 0.0615001 completions 213469 cutoffs 151141 evictions 1375
18 tt_hits 33628 tt_rate 0.200438 completions 574588 cutoffs 426005 evictions 9581
19 tt_hits 89202 tt_rate 0.531685 completions 1393156 cutoffs 976931 evictions 55988
20 tt_hits 260529 tt_rate 1.55287 completions 3633662 cutoffs 2689438 evictions 361995
21 tt_hits 678954 tt_rate 4.04688 completions 8397248 cutoffs 5820322 evictions 1767276
22 tt_hits 1871909 tt_rate 11.1574 completions 21017543 cutoffs 15497151 evictions 8894480
23 tt_hits 4499267 tt_rate 26.8177 completions 46193515 cutoffs 31641859 evictions 30071943
24 tt_hits 11334792 tt_rate 67.5606 completions 110945952 cutoffs 81363209 evictions 93093143
25 tt_hits 25377637 tt_rate 151.263 completions 233104585 cutoffs 157805403 evictions 214342031
26 tt_hits 60900192 tt_rate 362.993 completions 537378262 cutoffs 391821227 evictions 516610408
27 tt_hits 130866422 tt_rate 780.025 completions 1074398821 cutoffs 717272615 evictions 1050240589
28 tt_hits 304276264 tt_rate 1813.63 completions 2351785038 cutoffs 1701372771 evictions 2320896084
29 tt_hits 618698674 tt_rate 3687.73 completions 4387973864 cutoffs 2868387837 evictions 4346080898
30 tt_hits 1363792613 tt_rate 8128.84 completions 8826417341 cutoffs 6281828507 evictions 8765577950
31 tt_hits 2545331413 tt_rate 15171.4 completions 14705204542 cutoffs 9268854810 evictions 14615223277
32 tt_hits 5062741655 tt_rate 30176.3 completions 25261539773 cutoffs 17267549504 evictions 25127107867
33 tt_hits 0 tt_rate 0 completions 43022788550 cutoffs 25216523916 evictions 0
34 tt_hits 0 tt_rate 0 completions 73346540868 cutoffs 43505036727 evictions 0
35 tt_hits 0 tt_rate 0 completions 19386296042 cutoffs 0 evictions 0
-4
```


### more improvements + narrowing a/b window

Narrowing a/b search window has quite a few advantages. If we want to prove that score is 0, just search in [-1, 1] range. This way more things will be cut off. 

Currently we can already get the result in reduced range in a few hours for 6x6 board.

Before we get to 8x8 board, we need the following:
1. Multithreading. How would batching for inference work here? 
2. Actually training/applying the model
3. Other smaller optimizations.

Is multithreading easier with zero-window? Seems like we only need to pass a tiny piece of info.


### some improvements

After making some improvements, we get 6x6 solved (answer is -4, which looks correct) in 70519.7s on a single core.
Need to optimize further so that we can iterate on it easily.

Final stats:
```
70519.7
4 tt_hits 0 tt_rate 0 completions 1 cutoffs 0 evictions 0
5 tt_hits 3 tt_rate 3.57628e-05 completions 1 cutoffs 0 evictions 0
6 tt_hits 0 tt_rate 0 completions 3 cutoffs 0 evictions 0
7 tt_hits 0 tt_rate 0 completions 14 cutoffs 8 evictions 0
8 tt_hits 1 tt_rate 1.19209e-05 completions 39 cutoffs 17 evictions 0
9 tt_hits 0 tt_rate 0 completions 146 cutoffs 104 evictions 0
10 tt_hits 5 tt_rate 5.96046e-05 completions 419 cutoffs 254 evictions 0
11 tt_hits 43 tt_rate 0.0005126 completions 1383 cutoffs 971 evictions 0
12 tt_hits 163 tt_rate 0.00194311 completions 4129 cutoffs 2768 evictions 1
13 tt_hits 564 tt_rate 0.0067234 completions 12778 cutoffs 9053 evictions 7
14 tt_hits 1785 tt_rate 0.0212789 completions 37210 cutoffs 26011 evictions 62
15 tt_hits 4811 tt_rate 0.0573516 completions 109107 cutoffs 78145 evictions 711
16 tt_hits 14153 tt_rate 0.168717 completions 307510 cutoffs 218211 evictions 5324
17 tt_hits 49957 tt_rate 0.595534 completions 847320 cutoffs 606579 evictions 38871
18 tt_hits 147879 tt_rate 1.76286 completions 2265280 cutoffs 1613543 evictions 262592
19 tt_hits 439906 tt_rate 5.24409 completions 5921716 cutoffs 4237403 evictions 1567997
20 tt_hits 1167546 tt_rate 13.9182 completions 15057674 cutoffs 10696030 evictions 7620272
21 tt_hits 3239774 tt_rate 38.6211 completions 37622918 cutoffs 26837174 evictions 27977699
22 tt_hits 7945389 tt_rate 94.7164 completions 91280323 cutoffs 64471994 evictions 79943487
23 tt_hits 21033442 tt_rate 250.738 completions 217852347 cutoffs 154468801 evictions 202983733
24 tt_hits 49622757 tt_rate 591.549 completions 503810783 cutoffs 353087195 evictions 481906086
25 tt_hits 123992980 tt_rate 1478.11 completions 1146130500 cutoffs 806680704 evictions 1109379499
26 tt_hits 281475146 tt_rate 3355.45 completions 2521680790 cutoffs 1750496533 evictions 2456207930
27 tt_hits 664293999 tt_rate 7919 completions 5443457117 cutoffs 3794768589 evictions 5322209178
28 tt_hits 1447751564 tt_rate 17258.5 completions 11288243009 cutoffs 7726724761 evictions 11066453700
29 tt_hits 3209015231 tt_rate 38254.4 completions 22746533889 cutoffs 15606699484 evictions 22349627869
30 tt_hits 6574243171 tt_rate 78371.1 completions 43264118111 cutoffs 28851259953 evictions 42581001318
31 tt_hits 13380814750 tt_rate 159512 completions 78141312098 cutoffs 51950247817 evictions 77010315654
32 tt_hits 0 tt_rate 0 completions 153628743984 cutoffs 96903708734 evictions 0
33 tt_hits 0 tt_rate 0 completions 282700470773 cutoffs 173232316425 evictions 0
34 tt_hits 0 tt_rate 0 completions 461245124973 cutoffs 244834041118 evictions 0
35 tt_hits 0 tt_rate 0 completions 141986948226 cutoffs 0 evictions 0
-4
```


### what do we do next?

Optimize 6x6 othello ab search further, so that we can quickly run and experiment there
1. multithreading
2. faster symmetry/canonical board computation
3. full transposition table for first N layers, no eviction
4. better hashing to reduce collisions
5. implement it with template and specialize the implementation for last few layers to avoid conditions etc.

if we do everything right on 32 core machine we should get the entire thing done within 1-2 hours.


### how to use this with A/B search?

First option is to just do both things 'independently' - first we get a strong model and then 
we use it in A/B search to identify the ordering.

What if we try combining both? For example:
1. instead of value model evaluation we can use transposition table data (if available)?
2. let's think backwards - assume we have complete transposition table and already 'solved' the game.
How can we use it to train the model?

### need 4th process - baseline duel

A baseline to play against raw MCTS with many rollouts.


### how to allocate resources between train/selfplay/duel

Also other tradeoffs.

We can easily make self-play/duel cheaper by doing less rollouts. Is it a good idea though?


### othello 6x6 first training loop

1. We overfit with the old settings
2. Sample diversity seems ok from the first glance, but that's not taking symmetries into account
3. We do get a model which is better than raw MCTS, so it 'works'.

Useful queries:
```
sqlite> select repeats, sum(1) from (select boards_tensor, probs_tensor, sum(1) as repeats from samples group by boards_tensor, probs_tensor) group by repeats;

sqlite> select produced_by_model, sum(1) from samples group by produced_by_model;

sqlite> select id, evaluation from models;
```


### TIL - torch.from_numpy() 

will not copy the underlying storage, need to be careful with multithreading here


### othello dumb7fill 

Looks like there's a well-known approach to this:

https://www.chessprogramming.org/Dumb7Fill




### othello check valid moves
We indeed can find it with some bit operations.
See [mctslib/games/experimental/othello6x6.cpp]. We can shift in the loop (or unroll if needed).

Need to make it better and work for both 6x6 and 8x8

How do we apply move as quickly as possible?

This we can probably do with some generated code?





### othello

Given that most likely and interesting candidate is 8x8 othello, let's implement that game.
Something I need to look at is the best way to find all legal moves (seems possible to do with bit operations?)
and applying the moves themselves.

If we represent a board as 2 64-bit ints.
All valid moves has to have opponent stone as a neighbor and be empty.

So, for example, to get bits which has neighbor to the right we probably can:

1. mask out right boundary 
```v0 = board[opp] & 0xfefefefefefefefe;```

2. shift to the right by 1

```v0 = (v0 >> 1)```

3. check that bits are in empty slots:

```v0 = v0 & (~(board[self] | board[opp]))```

After that we can do the same with other 7 directions, and OR all of them

That's not sufficient - we need to have own stone at the end of the line though.

We can pre-generate the masks to check? Something like we did for winner check in mnk game.

This looks a little slower than it could be though. 

Need to look into this more, there might be already implementations doing this using AVX/NEON.

### training data diversity and exploration rate
TBD. Put some chart here

Query to get it from db:
```
sqlite> select repeats, sum(1) from (select boards_tensor, probs_tensor, sum(1) as repeats from samples group by boards_tensor, probs_tensor) group by repeats;
```

What we can visualize is the rate of repetition as a function of 'explore till move N' ?

### NaN / Inf in the training data

Likely happens when I abruptly shut down self-play.
Need to fix still.


### wasted puct cycles

Despite high GPU utilization, we might be wasting some of it. We'll evaluate whole batch no matter what, even if game is not in the active state.

Measurements show that ~10% of all evaluations are wasted:

```puct cycles: 126976000, 12967108```

We can overcome that by introducing separate game queue (still in single thread).



### testing fp16 on a100

here we try the following setup:
1. 2048 batch size
2. 3 threads
3. fp16 allowed for trt
4. 1000 rollouts per move
5. 2 res blocks

285k samples received in 10 minutes, which is 285m evaluations in 10 min, which is ~500k samples/second. GPU util is at ~75%.

synthetic benchmark w fp16:
```
2,1,30.137,236700,0.127
2,2,30.137,473400,0.064
2,4,30.113,946800,0.032
2,8,30.182,1893600,0.016
2,16,30.417,3787200,0.008
2,32,30.344,7369600,0.004
2,64,30.420,10233600,0.003
2,128,30.415,13504000,0.002
2,256,30.809,17177600,0.002
2,512,30.183,19609600,0.002
```

Benchmark throughput is 650k samples/second. We can probably improve a little further.


### offload everything

Looking at the GPU util, seems like we can just offload IO to separate thread and be happy.
Commenting it out is >15games/second.

And GPU util is ~94-95%

### revisiting M2 ANE again

Based on synthetic test, with 256 batch size for 2 residual blocks 
it takes [0.022ms per sample](scripts/m2_ane_benchmark.csv#L19), so the throughput is ~45k samples per second.
With 1000 rollouts per move that would be equivalent to 45 moves per second. 

What do we observe in practice for self-play is [~23k moves per 10 minute](scripts/m2_ane_moves_per_minute.log), thus, 38 moves per second which is pretty good.

With 2 threads we get 24.8k moves per 10 minute, thus, 41 moves per second which is even better.

### micro-batching for self-play

if we will run quantized to fp8 on H100, we might expect 4x from quantization + ~5-6x from new HW. That would mean ~2M per second per GPU. If we have MPMC queue for 8 GPU workers, that's 16M/s operations which might have quite some overhead.

So, we can probably proceed with hybrid approach: do the same MPMC, but for entire batches, not individual samples.

### quantization (post-training)

just used in benchmark on A100. fp16 is ~2x more throughput and int8 2x more.

### Batched MCTS or just use good queue? 

Based on the benchmark I did individual A100 will be able to evaluate the model with 10 residual blocks 60-70k times per second.
This doesn't sound like too much, so we might be able to just use some fast enough queue implementation to aggreagate samples to batches. For example, we can use fb's folly, use futures which we execute in pool with 1 thread, which will aggregate data to a buffer and call inference.

Batched MCTS, as of this writing, while providing really good throughput during active phase has pauses where CPUs are busy but GPU is idle.

At the same time:
1. We might get (how much?) higher numbers if we quantize? Let's try with fp16 first.
2. Maybe other GPUs will get faster (e.g. H100?)
3. With that, it's possible we can get to millions of evaluations/second and queue might become a bottleneck
3. Thus, it is unclear if we'll be able to easily aggregate samples to a batch without hitting some issue where queue/syncronization becomes a bottleneck


Hybrid approach is possible:
1. we use batch MCTS on micro-batches of size, say, 128, rather than actual larger batch size (2048?)
2. we dynamically aggregate mini-batches to large batch
3. This way we can both get large batch, avoid any potential issues with work queue overhead and avoid long idle periods.


### Benchmark 

Let's do the following test for Apple's M2 ANE and whatever
CUDA GPUs we can find:
1. We'll follow rough model structure of AGZ anyway, but will likely change number of residual blocks.
2. We can change batch size

Let's run a test to measure time for 1 inference depending on
these settings.

### Higher-level plan

1. Implement Reversi state, support 6x6 and 8x8 sizes.
2. Finish missing pieces for e2e learning - value model part, check multi-GPU support
3. Try training 6x6 model on single multi-GPU machine
4. Once model is good enough, try using it for complete Alpha/Beta search ordering
5. Check: how good our ordering actually was, how often did we end up cutting significant portion of a tree
6. Check: how can we leverage our model, and do we need to have another head in the model for prediction 'what to store in transposition table'
7. Iterate if needed
8. Once this is done, apply the findings to 8x8 case.
9. Train larger model, for longer time
10. Once we have a model, start Alpha-Beta search. Consider distributed search this time.  


### Running on multi-GPU

tbd

### Running on lambda instances (until I make an image):
1. clone this repo
2. install torch2trt

```
pip install tensorrt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt/
sudo chown ubuntu /usr/local/lib/python3.8/dist-packages/
python setup.py install
```

3. clone https://github.com/okuvshynov/cubestat
4. ```pip install pynvml```
5. clone https://github.com/okuvshynov/vimrc, follow the instructions there
6. depending on where we'd want to start, scp db file to remote machine


### self-play throughput tests

on single A100 with model update, 2048 batch size and 3 CPU threads we reach 12+ games/s.

on Apple M2 with 256 batch and no model update
rate = 0.833 games/s


## Brief notes/history of building it

This is a set of notes for myself, so that I can later write it up in a more readable way.

### idea
1. Train a really strong model for some game (currently freestyle gomoku 8x8, maybe othello 8x8 in future) <-- currently here
2. Use that model to guide full search and find a solution to a game.

### First implementation of mcts 

At first, MCTS without any model was implemented: https://colab.research.google.com/drive/13Sir3YSGAwZLFJCYaIG4zk83hP9nj9Kk?usp=sharing

It's fairly readable and there are some interesting examples re: what search thinks it should do.

### How to train ML model?

In order to train ML model we'd need to collect training data from our player. Training data will roughly come in the form of list of pairs (board, selected_move) based on what search was selecting. To collect this data we need to let our player play against itself and log the board/move probabilities for each turn.

This would be either really slow or really weak if we move forward with our python implementation, so native state/mcts is implemented 

### Native search/state

We'd like to use it to run initial data collection. The following properties are important:
1. it should be still exploring enough and not having human knowledge baked in
2. it should run in reasonable time
3. it should be of somewhat decent quality

If we run the entire e2e procedure on a more powerful hardware, we can avoid this step. However, for experimentation,  building the whole pipeline and improving cold-start time on conventional hardware it's very useful.

Rough ideas:
1. Store board as a pair of int64 numbers, each representing stones of one player. Current implementation supports boards of up to 8x8 size
2. For search, preallocate buffer of nodes in the search tree in the linear array and reuse it for each search iteration
3. For checking if the game ended, generate a code to compute it with several binary operations. Check if the last move is 'winning'
4. Rather than rewriting entire thing in C++ interface it with Python so that we can use it for convenient visualization.

It can still be improved, of course, but it was good enough


### Training first model

After generating first training data, we split it into training and validation sets and train our first model with Pytorch. Some notes:
1. We use residual tower similar to AlphaGoZero, but with only a few layers (for now) and predicting only probabilities for the next move, no value estimates.
2. I trained it on macbook air with M2 chip using "mps" device, which is short for 'metal performance shaders' and makes it run on GPU.

### Using the model
After we get the model, we need to use it. The place to use it is within search procedure (which is written at C++ at this point). For experimentation flexibility and visualization we apply model from Python, thus, we needed to make callback from C++ back to Python. 

At this stage it was possible to utilize Apple's Neural Engine - part of their M1-M2 SoC which seem to be good at inference. 

In order to do that torch model is compiled to CoreML model and loaded from Python. 
The data exchange between native library (search) and Python: we use shared numpy arrays as a data buffer, where search will fill in input (boards) and model will fill in our predictions.

At this stage we can run a search with model to guide the search. Model will be used to suggest which paths/notes are worth exploring. (see PUCT)

At the end of each rollout, however, we still do random play as we do not have value model trained yet.

### Getting new training data.
After we confirmed that our new player is stronger than old, model-free search, we can use it to generate new, better training data.

After we collect the data, we can train the new model, and do it again and again. Automating this process requires a little more work though

### 3 processes
There are 3 processes which are running in parallel. They can be distributed or can run on the same host. These processes are:
1. Playing games to collect training data
2. Training a model based on the data we collect, saving snapshots periodically
3. Evaluating the model snapshots we produce, and if they are 'better', let self-play know to use the new one.

To keep track of the housekeeping here we implement game server. Rough idea:
1. Store current state in SQLite
2. Communicate with 3 processes above using 0MQ.

State roughly is:
1. training examples
2. model snapshots
3. the results of model evaluation

At this step we could run 3 processes independently

### Some improvements
While working, it's fairly inefficient at first.
1. We need to clean up old models/samples from db 
2. We need to pass only incremental training data update to trainer
3. Do batching in self-play

### Batching
Batching is really important for model inference on any device, but especially on GPU/ANE. As we care about throughput much more than latency, we can just play many games at once and 'sync' on model evaluation step. 

Batching is implemented by introducing batching variation of Monte-Carlo Tree Seacrh.
Specifically
1. Assume single-thread evaluation for now
2. Assume we'd like to have batches of size batch_size
3. Start batch_size games at once. Allocate MCTS instance for each game.
4. All of the above is done in single CPU thread + evaluates the model on neural engine. On more traditional for ML hardware we'll be able to evaluate on GPU.
5. It is good enough to continue, we can further optimize it when we get to GPU 
6. Compared to other methods in literature (e.g. see https://ludii.games/citations/ARXIV2021-1.pdf), as we don't care too much about latency, we are not trying to parallelize/batch individual game state evaluation. Instead, we are just running many games at a time.


### How exploration/sampling at initial stages of the game affect results?

8x8 is a draw, so as our player gets better, it's likely we'll get many draws. Sampling allows to get more win/lose situation, thus allowing to train value model.

### Training on CUDA

Rented A100 on lambda, loaded data there. 
Seems like saving snapshot to db was pretty slow, we should snapshot less often/check why it is slow.
Getting all the data from db + building symmetries also takes time.

### quick performance notes:
1. with 1000 rollouts and model with 2 residual blocks on M2 self-play does ~0.7 games/second with 1 thread, 128 batch size
2. TBD

![Deep Rl horizon chart here](static/DeepRL_example.png)

### installing torch2trt on lambda machines
```
pip install tensorrt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt/
sudo chown ubuntu /usr/local/lib/python3.8/dist-packages/
python setup.py install
```

### how to dynamically adjust complexity
for self-play, we can increase/decrease number of rollouts
for training - ?
all hyperparams - ?
model complexity

### running self-play on A100

Pretty good, close to 10 games/s. Need to optimize:
1. create 'evaluator' which we run on a thread pool (or several of them)

### run distributed
1. server + train on remote GPU
2. self-play on M2
3. eval also on GPU?
4. how to sync DB? 

### Next to/write about:

2. show how Apple's Instruments help in understanding performance/utilization of GPU/ANE
3. try on CUDA (rent some GPU machine?)
4. implement Value head for the model. Mention how the fact that the game is a draw affects that.
5. Larger model 
6. check that it works well against some strong players
7. See how well it runs on modern CUDA devices (A100/H100). How to utilize those?
8. Run e2e process for a while to get a strong model

...

9. Start implementing full search for the game of choice 


## some examples

Current way to run the process:
1. start game server: 

```% python game_server.py   # <-- modify the path to sqlite db file if needed```

2. start self-play: 

```python selfplay_loop.py   # <-- it will start playing 'no model' mcts with 500k rollouts OR get the best latest model from server```

3. start model training:

```python train_loop.py # <-- it will wait till it gets enough initial samples```

4. start model eval: 

```python duel_loop.py```

To monitor what's going on we can query sqlite db.