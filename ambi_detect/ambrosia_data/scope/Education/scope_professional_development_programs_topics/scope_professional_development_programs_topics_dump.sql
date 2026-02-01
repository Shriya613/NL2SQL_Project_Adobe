BEGIN TRANSACTION;
CREATE TABLE Professional_Development_Programs (
    program_id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_name TEXT,
    duration INTEGER,
    cost REAL
);
INSERT INTO "Professional_Development_Programs" VALUES(1,'PDP A',30,999.99);
INSERT INTO "Professional_Development_Programs" VALUES(2,'PDP B',60,1899.99);
INSERT INTO "Professional_Development_Programs" VALUES(3,'PDP C',45,799.99);
INSERT INTO "Professional_Development_Programs" VALUES(4,'PDP D',90,2999.99);
INSERT INTO "Professional_Development_Programs" VALUES(5,'PDP E',120,3899.99);
CREATE TABLE Professional_Development_Programs_Topics (
    pdp_topics_id INTEGER,
    topics_id INTEGER,
    FOREIGN KEY(pdp_topics_id) REFERENCES Professional_Development_Programs(program_id),
    FOREIGN KEY(topics_id) REFERENCES Topics(id)
);
INSERT INTO "Professional_Development_Programs_Topics" VALUES(1,1);
INSERT INTO "Professional_Development_Programs_Topics" VALUES(2,1);
INSERT INTO "Professional_Development_Programs_Topics" VALUES(3,1);
INSERT INTO "Professional_Development_Programs_Topics" VALUES(4,1);
INSERT INTO "Professional_Development_Programs_Topics" VALUES(5,1);
INSERT INTO "Professional_Development_Programs_Topics" VALUES(5,2);
CREATE TABLE Topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_name TEXT
);
INSERT INTO "Topics" VALUES(1,'Differentiated Instruction');
INSERT INTO "Topics" VALUES(2,'Inclusive Classroom Management');
INSERT INTO "Topics" VALUES(3,'Curriculum Design');
INSERT INTO "Topics" VALUES(4,'Assessment Techniques');
INSERT INTO "Topics" VALUES(5,'Technology Integration');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Topics',5);
INSERT INTO "sqlite_sequence" VALUES('Professional_Development_Programs',5);
COMMIT;
