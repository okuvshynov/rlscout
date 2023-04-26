import sqlite3
from contextlib import closing

init_samples = """
CREATE TABLE IF NOT EXISTS
samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    boards_tensor BLOB,
    probs_tensor BLOB,
    game_id INTEGER,
    player INTEGER,
    skipped INTEGER,
    score INTEGER
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
    (boards_tensor, probs_tensor, game_id, score, player, skipped) 
VALUES
    (?, ?, ?, NULL, ?, ?)
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

count_models_to_eval_sql = """
SELECT
    count(*)
FROM 
    models 
WHERE
    evaluation=''
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
    id, score, boards_tensor, probs_tensor, player, skipped
FROM 
    samples 
WHERE
    id > ?
ORDER BY id
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

remove_old_samples = """
DELETE FROM
    samples
WHERE id IN (
    SELECT 
        id 
    FROM samples 
    ORDER BY 
        id 
    DESC
    LIMIT -1
    OFFSET ?)
"""

update_samples_with_scores_sql = """
UPDATE
    samples
SET
    score = ?
WHERE
    game_id = ?
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
        
    def count_models_to_eval(self):
        with closing(self.conn.cursor()) as cursor:
            cnt = cursor.execute(count_models_to_eval_sql).fetchone()
            return cnt[0]

    def get_last_model(self):
        with closing(self.conn.cursor()) as cursor:
            return cursor.execute(select_last_model_sql).fetchone()        

    def get_model(self, model_id):
        with closing(self.conn.cursor()) as cursor:
            model = cursor.execute(select_model_by_id_sql, (model_id, )).fetchone() 
            return model[0] if model is not None else None       

    def get_batch(self, size, from_id):
        with closing(self.conn.cursor()) as cursor:
            return cursor.execute(select_training_batch_sql, (from_id, size)).fetchall()

    def append_sample(self, boards, probs, game_id=None, player=0, skipped=0):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(insert_samples_sql, (boards, probs, game_id, player, skipped))
            self.conn.commit()

    def game_done(self, game_id, score):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(update_samples_with_scores_sql, (score, game_id))
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

    def cleanup_samples(self, samples_to_keep):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(remove_old_samples, (samples_to_keep, ))
            self.conn.commit()

    def get_stats(self):
        with closing(self.conn.cursor()) as cursor:
            return cursor.execute("select sum(1) from samples;").fetchall()
    
    def _setup_tables(self):
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(init_models)
            cursor.execute(init_samples)
