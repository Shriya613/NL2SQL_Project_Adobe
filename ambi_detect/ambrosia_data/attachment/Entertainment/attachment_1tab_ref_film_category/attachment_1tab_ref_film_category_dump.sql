BEGIN TRANSACTION;
CREATE TABLE Actors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT,
    lastName TEXT
);
INSERT INTO "Actors" VALUES(1,'Leonardo','DiCaprio');
INSERT INTO "Actors" VALUES(2,'Meryl','Streep');
CREATE TABLE Categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    categoryName TEXT UNIQUE
);
INSERT INTO "Categories" VALUES(1,'Fantasy');
INSERT INTO "Categories" VALUES(2,'Horror');
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    directorName TEXT
);
INSERT INTO "Directors" VALUES(1,'Peter Jackson');
INSERT INTO "Directors" VALUES(2,'Christopher Nolan');
CREATE TABLE Films (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    budget REAL ,
    categoryId INTEGER,
    FOREIGN KEY (categoryId) REFERENCES Categories(id)
);
INSERT INTO "Films" VALUES(1,'Lord of the Rings',20000000.0,1);
INSERT INTO "Films" VALUES(2,'Harry Potter',30000000.0,1);
INSERT INTO "Films" VALUES(3,'The Shining',20000000.0,2);
INSERT INTO "Films" VALUES(4,'It',18000000.0,2);
CREATE TABLE Roles (
    filmId INTEGER,
    actorId INTEGER,
    roleType TEXT,
    directorId INTEGER,
    FOREIGN KEY (filmId) REFERENCES Films(id),
    FOREIGN KEY (actorId) REFERENCES Actors(id),
    FOREIGN KEY (directorId) REFERENCES Directors(id),
    PRIMARY KEY (filmId, actorId, roleType, directorId)
);
INSERT INTO "Roles" VALUES(1,1,'Leading Role',1);
INSERT INTO "Roles" VALUES(2,2,'Supporting Role',2);
INSERT INTO "Roles" VALUES(1,2,'Villain',1);
INSERT INTO "Roles" VALUES(3,1,'Leading Role',1);
INSERT INTO "Roles" VALUES(4,2,'Supporting Role',2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Categories',2);
INSERT INTO "sqlite_sequence" VALUES('Films',4);
INSERT INTO "sqlite_sequence" VALUES('Actors',2);
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
COMMIT;
