BEGIN TRANSACTION;
CREATE TABLE Actor(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Birthdate DATE
);
INSERT INTO "Actor" VALUES(1,'Laurence Olivier','1907-05-22');
INSERT INTO "Actor" VALUES(2,'Sir Ian McKellen','1948-05-25');
INSERT INTO "Actor" VALUES(3,'Denzel Washington','1954-12-27');
INSERT INTO "Actor" VALUES(4,'Samuel L Jackson','1948-12-21');
INSERT INTO "Actor" VALUES(5,'Tom Hiddleston','1981-02-09');
CREATE TABLE Character(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    PlayID INTEGER,
    FOREIGN KEY (PlayID) REFERENCES Play(ID)
);
INSERT INTO "Character" VALUES(1,'Juliet',1);
INSERT INTO "Character" VALUES(2,'Shakespeare',1);
INSERT INTO "Character" VALUES(3,'Puck',2);
INSERT INTO "Character" VALUES(4,'Katherine',4);
INSERT INTO "Character" VALUES(5,'Banquo',5);
CREATE TABLE Play(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Theatrical_Form TEXT,
    Author TEXT,
    Year_Published INT
);
INSERT INTO "Play" VALUES(1,'Romeo And Juliet','Tragedy','William Shakespeare',1597);
INSERT INTO "Play" VALUES(2,'A Midsummer Night''s Dream','Comedy','William Shakespeare',1596);
INSERT INTO "Play" VALUES(3,'Hamlet','Tragedy','William Shakespeare',1603);
INSERT INTO "Play" VALUES(4,'Taming Of Shrew','Comedy','Christopher Marlowe',1594);
INSERT INTO "Play" VALUES(5,'Macbeth','Tragedy','William Shakespeare',1606);
CREATE TABLE Production(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    PlayID INTEGER,
    TheaterID INTEGER,
    Start_Date DATE,
    End_Date DATE,
    FOREIGN KEY (PlayID) REFERENCES Play(ID),
    FOREIGN KEY (TheaterID) REFERENCES Theater(ID)
);
INSERT INTO "Production" VALUES(1,1,1,'1954-01-01','1955-06-01');
INSERT INTO "Production" VALUES(2,2,2,'1981-02-01','1982-06-01');
INSERT INTO "Production" VALUES(3,3,3,'1990-09-01','1991-09-01');
INSERT INTO "Production" VALUES(4,4,4,'2003-12-01','2004-01-01');
INSERT INTO "Production" VALUES(5,5,5,'2000-06-01','2000-08-01');
CREATE TABLE Role(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    CharacterID INTEGER,
    ActorID INTEGER,
    PerformanceDate DATE,
    FOREIGN KEY (CharacterID) REFERENCES Character(ID),
    FOREIGN KEY (ActorID) REFERENCES Actor(ID)
);
INSERT INTO "Role" VALUES(1,1,1,'1954-01-01');
INSERT INTO "Role" VALUES(2,1,2,'1981-02-01');
INSERT INTO "Role" VALUES(3,2,3,'2000-06-01');
INSERT INTO "Role" VALUES(4,3,4,'1990-09-01');
INSERT INTO "Role" VALUES(5,4,5,'2003-12-01');
CREATE TABLE Theater(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Theater" VALUES(1,'The Royal Shakespeare Company');
INSERT INTO "Theater" VALUES(2,'Broadway Theatre');
INSERT INTO "Theater" VALUES(3,'Drury Lane Theatre');
INSERT INTO "Theater" VALUES(4,'Harold Pinter Theatre');
INSERT INTO "Theater" VALUES(5,'Gielgud Theatre');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Play',5);
INSERT INTO "sqlite_sequence" VALUES('Character',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
INSERT INTO "sqlite_sequence" VALUES('Role',5);
INSERT INTO "sqlite_sequence" VALUES('Theater',5);
INSERT INTO "sqlite_sequence" VALUES('Production',5);
COMMIT;
