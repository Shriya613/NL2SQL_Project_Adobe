BEGIN TRANSACTION;
CREATE TABLE Directors (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Country TEXT
);
INSERT INTO "Directors" VALUES(1,'John Doe','1980-01-01','USA');
INSERT INTO "Directors" VALUES(2,'Jane Smith','1975-02-02','UK');
CREATE TABLE Editors (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE,
    Country TEXT
);
INSERT INTO "Editors" VALUES(1,'Alice Johnson','1992-03-03','Canada');
INSERT INTO "Editors" VALUES(2,'Bob Brown','1988-04-04','Australia');
CREATE TABLE Episodes (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    DirectorID INTEGER,
    EditorID INTEGER,
    ReleaseDate DATE,
    FOREIGN KEY(DirectorID) REFERENCES Directors(ID),
    FOREIGN KEY(EditorID) REFERENCES Editors(ID)
);
INSERT INTO "Episodes" VALUES(1,'Pilot Episode',1,1,'2022-05-05');
INSERT INTO "Episodes" VALUES(2,'Episode 2',2,2,'2022-06-06');
CREATE TABLE Movies (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Year INT,
    Genre TEXT,
    DirectorID INTEGER,
    EditorID INTEGER,
    FOREIGN KEY(DirectorID) REFERENCES Directors(ID),
    FOREIGN KEY(EditorID) REFERENCES Editors(ID)
);
INSERT INTO "Movies" VALUES(1,'Movie A',2020,'Drama',1,1);
INSERT INTO "Movies" VALUES(2,'Movie B',2021,'Comedy',2,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
INSERT INTO "sqlite_sequence" VALUES('Editors',2);
INSERT INTO "sqlite_sequence" VALUES('Episodes',2);
INSERT INTO "sqlite_sequence" VALUES('Movies',2);
COMMIT;
