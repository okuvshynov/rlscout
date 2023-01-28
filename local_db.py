# using sqlite. 

# I've got a little confused about thread-safety and cross-process access in python/sqlite. 
# There're several relevant links:
# - https://ricardoanderegg.com/posts/python-sqlite-thread-safety/
# - https://discuss.python.org/t/is-sqlite3-threadsafety-the-same-thing-as-sqlite3-threadsafe-from-the-c-library/11463
# - https://stackoverflow.com/questions/18207193/concurrent-writing-with-sqlite3

# here we abuse SQLite for both multithrading and cross-process concurrency. 
# need to change this to a dedicated process running whatever under the hood
# most likely I'll use 0mq for communication 

import sqlite3
from contextlib import closing

import sys
import time

create_samples_table = """
CREATE TABLE IF NOT EXISTS
samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    boards_tensor BLOB,
    probs_tensor BLOB,
    produced_by_model INTEGER
);
"""

# evaluation
# '+' -- winner, was best at the time it was produced
# '-' -- loser, wasn't best at the time it was produced
# '' -- empty -- not evaluated yet
create_models_table = """
CREATE TABLE IF NOT EXISTS
models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    torch_model BLOB,
    evaluation TEXT
);
"""

# outcome - 'a', 'b', '.'
create_duels_table = """
CREATE TABLE IF NOT EXISTS
duels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    a_model_id INTEGER,
    a_rollouts INTEGER,
    a_temp REAL,
    b_model_id INTEGER,
    b_rollouts INTEGER,
    b_temp REAL,
    outcome TEXT,
    a_time_per_move_us INT,
    b_time_per_move_us INT
);
"""

insert_outcome_sql = """
INSERT INTO
duels (a_model_id, a_rollouts, a_temp, b_model_id, b_rollouts, b_temp, outcome, a_time_per_move_us, b_time_per_move_us)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

insert_samples_sql = """
INSERT INTO
samples(boards_tensor, probs_tensor, produced_by_model) 
VALUES(?, ?, ?)
"""

select_best_model_sql = """
SELECT
    id, torch_model
FROM 
    models 
WHERE 
    evaluation='+'
ORDER BY id
DESC
LIMIT 1
"""

select_model_to_eval_sql = """
SELECT
    id, torch_model
FROM 
    models 
WHERE
    evaluation=''
ORDER BY id
LIMIT 1
"""

insert_model_sql = """
INSERT INTO
models(torch_model, evaluation) 
VALUES(?, ?)
"""

select_training_batch_sql = """
SELECT
    boards_tensor, probs_tensor
FROM 
    samples 
ORDER BY id
DESC
LIMIT ?
"""

select_model_by_id_sql = """
SELECT
    torch_model
FROM 
    models 
WHERE
    id = ?
"""

record_evaluation_sql = """
UPDATE
    models
SET
    evaluation = ?
WHERE
    id = ?
"""

class LocalDB:
    def __init__(self, filename=":memory:"):
        self.filename = filename
        self.conn = sqlite3.connect(filename, check_same_thread=False)
        self._setup_tables()



    #######
    # interface needed for self-play and duel

    def get_best_model(self):
        with closing(self.conn.cursor()) as cursor:
            return cursor.execute(select_best_model_sql).fetchone()

    def get_last_not_evaluated_model(self):
        model_to_eval = None
        try:
            with closing(self.conn.cursor()) as cursor:
                model_to_eval = cursor.execute(select_model_to_eval_sql).fetchone()
        finally:
            return model_to_eval

    def get_model(self, model_id):
        with closing(self.conn.cursor()) as cursor:
            model = cursor.execute(select_model_by_id_sql, (model_id, )).fetchone() 
            return model[0] if model is not None else None       

    #######
    # read interface needed for training
    def get_batch(self, size):
        try:
            with closing(self.conn.cursor()) as cursor:
                return cursor.execute(select_training_batch_sql, (size, )).fetchall()
        except:
            print('Error querying DB.')
        finally:
            pass


    ###
    ## all writes

    def append_sample(self, boards, probs, model_id=None):
        while True:
            try:
                with closing(self.conn.cursor()) as cursor:
                    cursor.execute(insert_samples_sql, (boards, probs, model_id))
                    return
            except sqlite3.OperationalError as err:
                print(f'SQLite err: {err}. retrying in 1s')
                time.sleep(1)

    def save_snapshot(self, model):
        while True:
            try:
                with closing(self.conn.cursor()) as cursor:
                    cursor.execute(insert_model_sql, (model, ''))
                    self.conn.commit()
                    (snapshot_id, ) = cursor.execute("select seq from sqlite_sequence where name='models'").fetchone()
                    return snapshot_id
            except sqlite3.OperationalError as err:
                print(f'SQLite err: {err}. retrying in 1s')
                time.sleep(1)

    def log_outcome(self, model_a, rollouts_a, temp_a, model_b, rollouts_b, temp_b, outcome, time_per_move_a, time_per_move_b):
        while True:
            try:
                with closing(self.conn.cursor()) as cursor:
                    cursor.execute(insert_outcome_sql, (model_a, rollouts_a, temp_a, model_b, rollouts_b, temp_b, outcome, time_per_move_a, time_per_move_b))
                    self.conn.commit()
                    return
            except sqlite3.OperationalError as err:
                print(f'SQLite err: {err}. retrying in 1s')
                time.sleep(1)


    def record_evaluation(self, model_id, evaluation):
        while True:
            try:
                with closing(self.conn.cursor()) as cursor:
                    cursor.execute(record_evaluation_sql, (evaluation, model_id))
                    self.conn.commit()
                    return
            except sqlite3.OperationalError as err:
                print(f'SQLite err: {err}. retrying in 1s')
                time.sleep(1)

    # all 
    def _setup_tables(self):
        try:
            with closing(self.conn.cursor()) as cursor:
                cursor.execute(create_samples_table)
                cursor.execute(create_models_table)
                cursor.execute(create_duels_table)
        finally:
            pass