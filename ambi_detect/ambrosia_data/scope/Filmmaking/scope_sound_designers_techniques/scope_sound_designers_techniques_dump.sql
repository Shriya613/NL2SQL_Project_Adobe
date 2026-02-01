BEGIN TRANSACTION;
CREATE TABLE Directors(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Firstname TEXT,
    Lastname TEXT);
INSERT INTO "Directors" VALUES(1,'Frank','Darabont');
INSERT INTO "Directors" VALUES(2,'Robert','Zemeckis');
INSERT INTO "Directors" VALUES(3,'Quentin','Tarantino');
INSERT INTO "Directors" VALUES(4,'Steven','Spielberg');
INSERT INTO "Directors" VALUES(5,'Tommy','Lee Jones');
CREATE TABLE Movie_Director(
    MovieID INTEGER REFERENCES Movies(ID),
    DirectorID INTEGER REFERENCES Directors(ID));
CREATE TABLE Movie_Producer(
    MovieID INTEGER REFERENCES Movies(ID),
    ProducerID INTEGER REFERENCES Producers(ID));
CREATE TABLE Movies(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    ReleaseYear INT);
INSERT INTO "Movies" VALUES(1,'The Shawshank Redemption',1994);
INSERT INTO "Movies" VALUES(2,'Forrest Gump',1994);
INSERT INTO "Movies" VALUES(3,'Pulp Fiction',1994);
INSERT INTO "Movies" VALUES(4,'Schindler''s List',1993);
INSERT INTO "Movies" VALUES(5,'Good Will Hunting',1997);
CREATE TABLE Producers(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Firstname TEXT,
    Lastname TEXT);
INSERT INTO "Producers" VALUES(1,'Gordon','Carroll');
INSERT INTO "Producers" VALUES(2,'Bill','Safran');
INSERT INTO "Producers" VALUES(3,'Cary','Brokaw');
INSERT INTO "Producers" VALUES(4,'Harvey','Weinstein');
INSERT INTO "Producers" VALUES(5,'David','Fincher');
CREATE TABLE Sound_Designer_Movie(
    SoundDesignerID INTEGER REFERENCES Sound_Designers(ID),
    MovieID INTEGER REFERENCES Movies(ID));
CREATE TABLE Sound_Designer_Technique(
    SoundDesignerID INTEGER REFERENCES Sound_Designers(ID),
    TechniqueID INTEGER REFERENCES Techniques(ID));
INSERT INTO "Sound_Designer_Technique" VALUES(1,1);
INSERT INTO "Sound_Designer_Technique" VALUES(2,1);
INSERT INTO "Sound_Designer_Technique" VALUES(3,1);
INSERT INTO "Sound_Designer_Technique" VALUES(4,1);
INSERT INTO "Sound_Designer_Technique" VALUES(5,1);
INSERT INTO "Sound_Designer_Technique" VALUES(5,3);
CREATE TABLE Sound_Designers(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Email TEXT);
INSERT INTO "Sound_Designers" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Sound_Designers" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Sound_Designers" VALUES(3,'Alice Johnson','alice@example.com');
INSERT INTO "Sound_Designers" VALUES(4,'Bob Brown','bob@example.com');
INSERT INTO "Sound_Designers" VALUES(5,'Charlie Green','charlie@example.com');
CREATE TABLE Techniques(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    MethodName TEXT);
INSERT INTO "Techniques" VALUES(1,'Foley Artistry');
INSERT INTO "Techniques" VALUES(2,'ADR Recording');
INSERT INTO "Techniques" VALUES(3,'Sound Editing');
INSERT INTO "Techniques" VALUES(4,'Mixing & Mastering');
INSERT INTO "Techniques" VALUES(5,'Field Recording');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Sound_Designers',5);
INSERT INTO "sqlite_sequence" VALUES('Techniques',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Producers',5);
COMMIT;
