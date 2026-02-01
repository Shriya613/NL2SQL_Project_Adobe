BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT,
    birthYear INTEGER,
    playId INTEGER,
    FOREIGN KEY (playId) REFERENCES Play(id)
);
INSERT INTO "Actor" VALUES(1,'Tom','Hanks',1960,2);
INSERT INTO "Actor" VALUES(2,'John','Burton',1987,3);
INSERT INTO "Actor" VALUES(3,'Julia','Roberts',1990,4);
INSERT INTO "Actor" VALUES(4,'Samuel','L. Jackson',1989,1);
INSERT INTO "Actor" VALUES(5,'Denzel','Washington',1987,3);
CREATE TABLE Author(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "Author" VALUES(1,'Shakespeare');
INSERT INTO "Author" VALUES(2,'Arthur Miller');
INSERT INTO "Author" VALUES(3,'Neil Simon');
INSERT INTO "Author" VALUES(4,'Aphra Behn');
INSERT INTO "Author" VALUES(5,'Oscar Wilde');
CREATE TABLE Character(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    characterName TEXT,
    playId INTEGER,
    actorId INTEGER,
    FOREIGN KEY (playId) REFERENCES Play(id),
    FOREIGN KEY (actorId) REFERENCES Actor(id)
);
INSERT INTO "Character" VALUES(1,'Cassius',2,1);
INSERT INTO "Character" VALUES(2,'Titus Andronicus',1,4);
INSERT INTO "Character" VALUES(3,'Blanche DuBois',4,2);
INSERT INTO "Character" VALUES(4,'Felix Unger',3,3);
INSERT INTO "Character" VALUES(5,'Macbeth',1,5);
CREATE TABLE Play(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    authorId INTEGER,
    theatricalFormId INTEGER,
    FOREIGN KEY (authorId) REFERENCES Author(id),
    FOREIGN KEY (theatricalFormId) REFERENCES Theatrical_Form(id)
);
INSERT INTO "Play" VALUES(1,'Romeo and Juliet',1,1);
INSERT INTO "Play" VALUES(2,'Death of a Salesman',2,3);
INSERT INTO "Play" VALUES(3,'The Odd Couple',3,1);
INSERT INTO "Play" VALUES(4,'Othello',1,2);
INSERT INTO "Play" VALUES(5,'Aphra Behn''s Comedy',4,1);
CREATE TABLE Role(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roleDescription TEXT,
    characterId INTEGER,
    actorId INTEGER,
    FOREIGN KEY (characterId) REFERENCES Character(id),
    FOREIGN KEY (actorId) REFERENCES Actor(id)
);
INSERT INTO "Role" VALUES(1,'Friendly neighbourhood superhero',1,1);
INSERT INTO "Role" VALUES(2,'Rich business man',2,2);
INSERT INTO "Role" VALUES(3,'Kind old lady',3,3);
INSERT INTO "Role" VALUES(4,'Lead detective',4,4);
INSERT INTO "Role" VALUES(5,'Brave warrior',5,5);
CREATE TABLE Theatrical_Form(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    form TEXT UNIQUE
);
INSERT INTO "Theatrical_Form" VALUES(1,'Comedy');
INSERT INTO "Theatrical_Form" VALUES(2,'Tragedy');
INSERT INTO "Theatrical_Form" VALUES(3,'Drama');
INSERT INTO "Theatrical_Form" VALUES(4,'Melodrama');
INSERT INTO "Theatrical_Form" VALUES(5,'Farce');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Author',5);
INSERT INTO "sqlite_sequence" VALUES('Theatrical_Form',5);
INSERT INTO "sqlite_sequence" VALUES('Play',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
INSERT INTO "sqlite_sequence" VALUES('Character',5);
INSERT INTO "sqlite_sequence" VALUES('Role',5);
COMMIT;
