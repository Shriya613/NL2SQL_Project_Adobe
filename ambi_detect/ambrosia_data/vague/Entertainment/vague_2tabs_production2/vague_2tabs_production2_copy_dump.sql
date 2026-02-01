BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT);
INSERT INTO "Actor" VALUES(1,'Tom','Cruise');
INSERT INTO "Actor" VALUES(2,'Matthew','McConaughey');
CREATE TABLE Play_Actor (
    play_id INTEGER,
    actor_id INTEGER,
    character_played TEXT,
    FOREIGN KEY(play_id) REFERENCES Stage_Play(id),
    FOREIGN KEY(actor_id) REFERENCES Actor(id));
INSERT INTO "Play_Actor" VALUES(1,1,'Hamlet');
INSERT INTO "Play_Actor" VALUES(2,1,'Othello');
INSERT INTO "Play_Actor" VALUES(1,2,'Laertes');
INSERT INTO "Play_Actor" VALUES(2,2,'Iago');
CREATE TABLE "Production" (
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT);
INSERT INTO "Production" VALUES(1,'The Royal Theatre','123 Main St');
INSERT INTO "Production" VALUES(2,'Drama House','456 Broadway');
CREATE TABLE Review (
    id INTEGER PRIMARY KEY,
    rating INTEGER ,
    reviewer_email TEXT,
    stage_play_id INTEGER,
    FOREIGN KEY(stage_play_id) REFERENCES Stage_Play(id));
INSERT INTO "Review" VALUES(1,8,'reviews@hotmail.com',1);
INSERT INTO "Review" VALUES(2,9,'reviews@outlook.com',2);
CREATE TABLE "Stage_Play" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    theater_company_id INTEGER,
    FOREIGN KEY(theater_company_id) REFERENCES "Production"(id));
INSERT INTO "Stage_Play" VALUES(1,'Hamlet','A classic tragedy by William Shakespeare.',1);
INSERT INTO "Stage_Play" VALUES(2,'Othello','Another great work from Shakespeare.',1);
COMMIT;
