How could we deploy 6x6 e2e?

Overall, we have the following components:
1. self-play 
2. model training
3. model eval
4. model eval vs ab [?]
5. randomized distributed ab search [N]
6. model storage
7. sample storage
8. transposition table


What HW do we have now?

1. Mac Mini M1. 
2. Macbook Air M2
3. Raspberry PI 4 Model B
4. Raspberry PI 3 Model B+
5. Whatever we can rent on lambda. Needs callbacks?

We don't nesessarily have to run everything at the same time, but would be nice to have it running in ~realistic setting.

It would be also helpful to test how 'distributed' version would work.