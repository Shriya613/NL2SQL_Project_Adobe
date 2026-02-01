BEGIN TRANSACTION;
CREATE TABLE Movies(
    MovieID INTEGER PRIMARY KEY,
    Title TEXT,
    ReleaseYear INT ,
    Genre TEXT);
INSERT INTO "Movies" VALUES(1,'The Shawshank Redemption',1994,'Drama');
INSERT INTO "Movies" VALUES(2,'Forrest Gump',1994,'Drama');
INSERT INTO "Movies" VALUES(3,'Pulp Fiction',1994,'Crime');
INSERT INTO "Movies" VALUES(4,'Good Will Hunting',1997,'Drama');
CREATE TABLE "Production"(
    "ID" INTEGER PRIMARY KEY,
    BirthDate DATE ,
    Film_Profession TEXT,
    Award TEXT, FullName TEXT);
INSERT INTO "Production" VALUES(1,'1960-07-23','Director','Oscar','John Doe');
INSERT INTO "Production" VALUES(2,'1985-08-10','Director','Golden Globe','Jane Smith');
INSERT INTO "Production" VALUES(3,'1975-09-11','Producer','Oscar','Alice Brown');
INSERT INTO "Production" VALUES(4,'1990-10-12','Producer','Emmy','Bob White');
COMMIT;
