# using sqlite. 

# I've got a little confused about thread-safety and cross-process access in python/sqlite. 
# There're several relevant links:
# - https://ricardoanderegg.com/posts/python-sqlite-thread-safety/
# - https://discuss.python.org/t/is-sqlite3-threadsafety-the-same-thing-as-sqlite3-threadsafe-from-the-c-library/11463
# - https://stackoverflow.com/questions/18207193/concurrent-writing-with-sqlite3

# but it seems like it should be safe. Let's try it out to see if that works at all.

import sqlite3
from contextlib import closing

create_samples_table = """
CREATE TABLE IF NOT EXISTS
samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    boards_tensor BLOB,
    probs_tensor BLOB,
    produced_by_model INTEGER
);

"""

create_models_table = """
CREATE TABLE IF NOT EXISTS
models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    torch_model BLOB,
    was_best INTEGER
);
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
    was_best=1
ORDER BY id
DESC
LIMIT 1
"""

insert_model_sql = """
INSERT INTO
models(torch_model, was_best) 
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


class LocalDB:
    def __init__(self, filename=":memory:"):
        self.filename = filename
        self.conn = sqlite3.connect(filename, check_same_thread=False)
        self._setup_tables()


    def _setup_tables(self):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(create_samples_table)
            cursor.execute(create_models_table)

    #######
    # interface needed for self-play:

    def append_sample(self, boards, probs, model_id=None):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(insert_samples_sql, (boards, probs, model_id))

    def try_commit(self):
        try:
            self.conn.commit()
        finally:
            pass

    def get_best_model(self):
        with closing(self.conn.cursor()) as cursor:
            return cursor.execute(select_best_model_sql).fetchone()

    #######
    # interface needed for training

    def get_batch(self, size):
        try:
            with closing(self.conn.cursor()) as cursor:
                return cursor.execute(select_training_batch_sql, (size, )).fetchall()
        except:
            print('Error querying DB.')
        finally:
            pass

    def save_snapshot(self, model):
        with closing(self.conn.cursor()) as cursor:
            try:
                cursor.execute(insert_model_sql, (model, True))
                self.conn.commit()
            finally:
                pass

    ###
    # debug 
    def query_samples(self):
        with closing(self.conn.cursor()) as cursor:
            samples = cursor.execute("select * from samples order by id desc").fetchall()
            print(samples)



if __name__ == "__main__":
    db = LocalDB()
    db.append_sample("ololo", "kokoko", 123)
    print(db.get_best_model())
    db.save_snapshot("oioi")
    print(db.get_best_model())
    db.query_samples()
    print(db.get_batch(100))