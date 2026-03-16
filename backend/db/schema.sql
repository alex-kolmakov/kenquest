-- KenQuest DuckDB schema
-- DuckDB is used both as DLT destination (raw_materials) and app database.

CREATE TABLE IF NOT EXISTS topics (
    id          VARCHAR PRIMARY KEY,
    name        VARCHAR NOT NULL,
    description VARCHAR DEFAULT '',
    status      VARCHAR DEFAULT 'pending',
    concept_count INTEGER DEFAULT 0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_materials (
    id          VARCHAR PRIMARY KEY,
    topic_id    VARCHAR NOT NULL,
    source      VARCHAR NOT NULL,   -- 'wikipedia', 'openstax', 'arxiv', 'doab'
    url         VARCHAR,
    title       VARCHAR,
    content     TEXT,
    -- Attribution fields (required for book sources)
    authors     VARCHAR,            -- comma-separated author names
    publisher   VARCHAR,
    license     VARCHAR,            -- e.g. 'CC BY 4.0'
    license_url VARCHAR,
    fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS concepts (
    id          VARCHAR PRIMARY KEY,   -- stable slug
    topic_id    VARCHAR NOT NULL REFERENCES topics(id),
    name        VARCHAR NOT NULL,
    description TEXT,
    difficulty  INTEGER CHECK (difficulty BETWEEN 1 AND 5),
    source_refs VARCHAR[],             -- list of raw_material ids
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prerequisite_edges (
    source_id   VARCHAR NOT NULL REFERENCES concepts(id),
    target_id   VARCHAR NOT NULL REFERENCES concepts(id),
    strength    FLOAT DEFAULT 1.0,
    rationale   VARCHAR DEFAULT '',
    PRIMARY KEY (source_id, target_id)
);

CREATE TABLE IF NOT EXISTS quiz_sessions (
    id              VARCHAR PRIMARY KEY,
    concept_id      VARCHAR NOT NULL REFERENCES concepts(id),
    questions_json  JSON NOT NULL,
    attempts_json   JSON DEFAULT '[]',
    passed          BOOLEAN DEFAULT FALSE,
    avg_score       FLOAT,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at    TIMESTAMP
);

CREATE TABLE IF NOT EXISTS concept_progress (
    topic_id    VARCHAR NOT NULL,
    concept_id  VARCHAR NOT NULL REFERENCES concepts(id),
    status      VARCHAR DEFAULT 'locked',
    best_score  FLOAT,
    attempts    INTEGER DEFAULT 0,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (topic_id, concept_id)
);

CREATE TABLE IF NOT EXISTS llm_cache (
    content_hash    VARCHAR PRIMARY KEY,
    prompt_hash     VARCHAR NOT NULL,
    response_json   JSON NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
