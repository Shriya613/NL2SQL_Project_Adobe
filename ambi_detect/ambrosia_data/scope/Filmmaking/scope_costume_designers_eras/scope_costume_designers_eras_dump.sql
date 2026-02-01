BEGIN TRANSACTION;
CREATE TABLE Actors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT
);
INSERT INTO "Actors" VALUES(1,'Alex','Adams');
INSERT INTO "Actors" VALUES(2,'Ben','Blake');
INSERT INTO "Actors" VALUES(3,'Chris','Cooper');
INSERT INTO "Actors" VALUES(4,'Daniel','Dean');
INSERT INTO "Actors" VALUES(5,'Ethan','Edwards');
CREATE TABLE Costume_Designers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthdate DATE
);
INSERT INTO "Costume_Designers" VALUES(1,'John Smith','1960-07-23');
INSERT INTO "Costume_Designers" VALUES(2,'Jane Doe','1980-09-15');
INSERT INTO "Costume_Designers" VALUES(3,'Michael Brown','1990-11-30');
INSERT INTO "Costume_Designers" VALUES(4,'Emily Davis','1995-02-22');
INSERT INTO "Costume_Designers" VALUES(5,'David Johnson','1970-05-01');
CREATE TABLE Costume_Designers_Eras (
    costume_designer_id INTEGER REFERENCES Costume_Designers(id) ON DELETE CASCADE,
    eras_id INTEGER REFERENCES Eras(id) ON DELETE CASCADE,
    PRIMARY KEY (costume_designer_id, eras_id)
);
INSERT INTO "Costume_Designers_Eras" VALUES(1,1);
INSERT INTO "Costume_Designers_Eras" VALUES(2,2);
INSERT INTO "Costume_Designers_Eras" VALUES(3,3);
INSERT INTO "Costume_Designers_Eras" VALUES(4,4);
INSERT INTO "Costume_Designers_Eras" VALUES(5,5);
INSERT INTO "Costume_Designers_Eras" VALUES(2,1);
INSERT INTO "Costume_Designers_Eras" VALUES(3,1);
INSERT INTO "Costume_Designers_Eras" VALUES(4,1);
INSERT INTO "Costume_Designers_Eras" VALUES(5,1);
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT
);
INSERT INTO "Directors" VALUES(1,'Mark','Wood');
INSERT INTO "Directors" VALUES(2,'Nick','Hopkins');
INSERT INTO "Directors" VALUES(3,'Samantha','Green');
INSERT INTO "Directors" VALUES(4,'Tommy','Carter');
INSERT INTO "Directors" VALUES(5,'Paul','Evans');
CREATE TABLE Eras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    era_name TEXT
);
INSERT INTO "Eras" VALUES(1,'Victorian Era');
INSERT INTO "Eras" VALUES(2,'Modern Era');
INSERT INTO "Eras" VALUES(3,'Renaissance');
INSERT INTO "Eras" VALUES(4,'Baroque');
INSERT INTO "Eras" VALUES(5,'Gothic');
CREATE TABLE Film_Projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INT,
    director TEXT
);
INSERT INTO "Film_Projects" VALUES(1,'Film A',2000,'Alice Cooper');
INSERT INTO "Film_Projects" VALUES(2,'Film B',2005,'Bob Evans');
INSERT INTO "Film_Projects" VALUES(3,'Film C',2010,'Carol King');
INSERT INTO "Film_Projects" VALUES(4,'Film D',2015,'Dave Lee');
INSERT INTO "Film_Projects" VALUES(5,'Film E',2020,'Edward Wright');
CREATE TABLE Films_and_Actors (
    film_id INTEGER REFERENCES Film_Projects(id) ON DELETE CASCADE,
    actor_id INTEGER REFERENCES Actors(id) ON DELETE CASCADE,
    PRIMARY KEY (film_id, actor_id)
);
INSERT INTO "Films_and_Actors" VALUES(1,1);
INSERT INTO "Films_and_Actors" VALUES(1,2);
INSERT INTO "Films_and_Actors" VALUES(2,3);
INSERT INTO "Films_and_Actors" VALUES(3,4);
INSERT INTO "Films_and_Actors" VALUES(4,5);
CREATE TABLE Films_and_Directors (
    film_id INTEGER REFERENCES Film_Projects(id) ON DELETE CASCADE,
    director_id INTEGER REFERENCES Directors(id) ON DELETE CASCADE,
    PRIMARY KEY (film_id, director_id)
);
INSERT INTO "Films_and_Directors" VALUES(1,1);
INSERT INTO "Films_and_Directors" VALUES(2,2);
INSERT INTO "Films_and_Directors" VALUES(3,3);
INSERT INTO "Films_and_Directors" VALUES(4,4);
INSERT INTO "Films_and_Directors" VALUES(5,5);
CREATE TABLE Roles (
    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id INTEGER REFERENCES Actors(id) ON DELETE SET NULL,
    character_name TEXT
);
INSERT INTO "Roles" VALUES(1,1,'Role A');
INSERT INTO "Roles" VALUES(2,2,'Role B');
INSERT INTO "Roles" VALUES(3,3,'Role C');
INSERT INTO "Roles" VALUES(4,4,'Role D');
INSERT INTO "Roles" VALUES(5,5,'Role E');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Costume_Designers',5);
INSERT INTO "sqlite_sequence" VALUES('Eras',5);
INSERT INTO "sqlite_sequence" VALUES('Film_Projects',5);
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Roles',5);
COMMIT;
