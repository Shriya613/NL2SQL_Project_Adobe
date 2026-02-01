BEGIN TRANSACTION;
CREATE TABLE "Additional_Roles"(
    "AdditionalRoleID" INTEGER PRIMARY KEY AUTOINCREMENT,
    Role TEXT,
    "Fee" DECIMAL(10,2));
INSERT INTO "Additional_Roles" VALUES(1,'Producer',100000);
INSERT INTO "Additional_Roles" VALUES(2,'Screenwriter',75000);
INSERT INTO "Additional_Roles" VALUES(3,'Composer',50000);
INSERT INTO "Additional_Roles" VALUES(4,'Editor',40000);
INSERT INTO "Additional_Roles" VALUES(5,'Art Director',35000);
CREATE TABLE DirectorFilms(
    DirectorFilmID INTEGER PRIMARY KEY AUTOINCREMENT,
    DirectorID INTEGER,
    FilmID INTEGER,
    FOREIGN KEY (DirectorID) REFERENCES Directors(DirectorID),
    FOREIGN KEY (FilmID) REFERENCES Films(FilmID));
INSERT INTO "DirectorFilms" VALUES(1,1,1);
INSERT INTO "DirectorFilms" VALUES(2,2,2);
INSERT INTO "DirectorFilms" VALUES(3,3,3);
INSERT INTO "DirectorFilms" VALUES(4,4,4);
INSERT INTO "DirectorFilms" VALUES(5,5,5);
CREATE TABLE Directors(
    DirectorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    BirthDate DATE);
INSERT INTO "Directors" VALUES(1,'Alfred Hitchcock','August 13, 1899');
INSERT INTO "Directors" VALUES(2,'Steven Spielberg','December 18, 1946');
INSERT INTO "Directors" VALUES(3,'Christopher Nolan','July 30, 1970');
INSERT INTO "Directors" VALUES(4,'Quentin Tarantino','March 27, 1963');
INSERT INTO "Directors" VALUES(5,'Martin Scorsese','November 22, 1942');
CREATE TABLE "Directors_Roles"(
    "DirectorRoleID" INTEGER PRIMARY KEY AUTOINCREMENT,
    DirectorID INTEGER,
    "AdditionalRoleID" INTEGER,
    FOREIGN KEY (DirectorID) REFERENCES Directors(DirectorID),
    FOREIGN KEY ("AdditionalRoleID") REFERENCES Additional_Roles(AdditionalRoleID));
INSERT INTO "Directors_Roles" VALUES(1,1,1);
INSERT INTO "Directors_Roles" VALUES(2,2,1);
INSERT INTO "Directors_Roles" VALUES(3,3,1);
INSERT INTO "Directors_Roles" VALUES(4,4,1);
INSERT INTO "Directors_Roles" VALUES(5,5,1);
INSERT INTO "Directors_Roles" VALUES(6,5,5);
CREATE TABLE Films(
    FilmID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INT);
INSERT INTO "Films" VALUES(1,'Psycho',1960);
INSERT INTO "Films" VALUES(2,'E.T.',1982);
INSERT INTO "Films" VALUES(3,'Inception',2010);
INSERT INTO "Films" VALUES(4,'Django Unchained',2012);
INSERT INTO "Films" VALUES(5,'Taxi Driver',1976);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Films',5);
INSERT INTO "sqlite_sequence" VALUES('DirectorFilms',5);
INSERT INTO "sqlite_sequence" VALUES('Additional_Roles',5);
INSERT INTO "sqlite_sequence" VALUES('Directors_Roles',6);
COMMIT;
