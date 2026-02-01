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
CREATE TABLE Producer (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE);
INSERT INTO "Producer" VALUES(1,'John Doe','johndoe@gmail.com');
INSERT INTO "Producer" VALUES(2,'Jane Smith','janesmith@yahoo.com');
CREATE TABLE Review (
    id INTEGER PRIMARY KEY,
    rating INTEGER ,
    reviewer_email TEXT,
    stage_play_id INTEGER,
    FOREIGN KEY(stage_play_id) REFERENCES Stage_Play(id));
INSERT INTO "Review" VALUES(1,8,'reviews@hotmail.com',1);
INSERT INTO "Review" VALUES(2,9,'reviews@outlook.com',2);
CREATE TABLE Stage_Play (
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    theater_company_id INTEGER,
    producer_id INTEGER,
    FOREIGN KEY(theater_company_id) REFERENCES Theater_Company(id),
    FOREIGN KEY(producer_id) REFERENCES Producer(id));
INSERT INTO "Stage_Play" VALUES(1,'Hamlet','A classic tragedy by William Shakespeare.',1,1);
INSERT INTO "Stage_Play" VALUES(2,'Othello','Another great work from Shakespeare.',1,1);
CREATE TABLE Theater_Company (
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT);
INSERT INTO "Theater_Company" VALUES(1,'The Royal Theatre','123 Main St');
INSERT INTO "Theater_Company" VALUES(2,'Drama House','456 Broadway');
COMMIT;
