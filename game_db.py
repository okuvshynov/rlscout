import sqlite3
from contextlib import closing

init_samples = """
CREATE TABLE IF NOT EXISTS
samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    boards_tensor BLOB,
    probs_tensor BLOB,
    produced_by_model INTEGER
);
"""
init_models = """
CREATE TABLE IF NOT EXISTS
models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    torch_model BLOB,
    evaluation TEXT
);
"""

insert_samples_sql = """
INSERT INTO
samples
    (boards_tensor, probs_tensor, produced_by_model) 
VALUES
    (?, ?, ?)
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

select_last_model_sql = """
SELECT
    id, torch_model
FROM 
    models 
ORDER BY id
DESC
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

class GameDB:
    def __init__(self, filename):
        self.filename = filename
        self.conn = sqlite3.connect(filename, check_same_thread=False)
        self._setup_tables()

    def get_best_model(self):
        with closing(self.conn.cursor()) as cursor:
            return cursor.execute(select_best_model_sql).fetchone()

    def get_last_not_evaluated_model(self):
        with closing(self.conn.cursor()) as cursor:
            return cursor.execute(select_model_to_eval_sql).fetchone()

    def get_last_model(self):
        with closing(self.conn.cursor()) as cursor:
            return cursor.execute(select_model_to_eval_sql).fetchone()        

    def get_model(self, model_id):
        with closing(self.conn.cursor()) as cursor:
            model = cursor.execute(select_model_by_id_sql, (model_id, )).fetchone() 
            return model[0] if model is not None else None       

    def get_batch(self, size):
        with closing(self.conn.cursor()) as cursor:
            return cursor.execute(select_training_batch_sql, (size, )).fetchall()

    ###
    ## all writes

    def append_sample(self, boards, probs, model_id=None):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(insert_samples_sql, (boards, probs, model_id))
            self.conn.commit()

    def save_snapshot(self, model):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(insert_model_sql, (model, ''))
            self.conn.commit()

    def record_evaluation(self, model_id, evaluation):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute("""
                UPDATE
                    models
                SET
                    evaluation = ?
                WHERE
                    id = ?
            """, (evaluation, model_id))
            self.conn.commit()

    # all 
    def _setup_tables(self):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(init_models)
            cursor.execute(init_samples)