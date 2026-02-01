BEGIN TRANSACTION;
CREATE TABLE Actors (
    actorID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INT
);
INSERT INTO "Actors" VALUES(1,'Actor1',30);
INSERT INTO "Actors" VALUES(2,'Actor2',25);
INSERT INTO "Actors" VALUES(3,'Actor3',35);
INSERT INTO "Actors" VALUES(4,'Actor4',28);
INSERT INTO "Actors" VALUES(5,'Actor5',40);
CREATE TABLE Cinematographers (
    cinematographerID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INT
);
INSERT INTO "Cinematographers" VALUES(1,'Cinematograher1',10);
INSERT INTO "Cinematographers" VALUES(2,'Cinematograher2',15);
CREATE TABLE Directors (
    directorID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birthdate DATE
);
INSERT INTO "Directors" VALUES(1,'Director1','1975-11-26');
INSERT INTO "Directors" VALUES(2,'Director2','1980-08-03');
CREATE TABLE FilmProjects (
    projectID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    releaseDate DATE
);
INSERT INTO "FilmProjects" VALUES(1,'Film1','2023-01-01');
INSERT INTO "FilmProjects" VALUES(2,'Film2','2023-07-08');
CREATE TABLE Fixtures (
    fixtureType IDENTIFIER PRIMARY KEY,
    description TEXT
);
INSERT INTO "Fixtures" VALUES('Softbox',NULL);
INSERT INTO "Fixtures" VALUES('LED Panel',NULL);
INSERT INTO "Fixtures" VALUES('Fresnel',NULL);
CREATE TABLE Lighting_Technicians (
    technicianID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    yearsExperience INT
);
INSERT INTO "Lighting_Technicians" VALUES(1,'Tech1',5);
INSERT INTO "Lighting_Technicians" VALUES(2,'Tech2',8);
INSERT INTO "Lighting_Technicians" VALUES(3,'Tech3',7);
INSERT INTO "Lighting_Technicians" VALUES(4,'Tech4',6);
INSERT INTO "Lighting_Technicians" VALUES(5,'Tech5',4);
CREATE TABLE Lighting_Technicians_Fixtures (
    technicianID INTEGER,
    fixtureType IDENTIFIER,
    FOREIGN KEY (technicianID) REFERENCES Lighting_Technicians(technicianID),
    FOREIGN KEY (fixtureType) REFERENCES Fixtures(fixtureType)
);
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(1,'Softbox');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(1,'LED Panel');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(2,'Softbox');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(2,'Fresnel');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(3,'Softbox');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(3,'LED Panel');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(4,'Softbox');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(4,'Fresnel');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(5,'Softbox');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(5,'LED Panel');
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(1,3);
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(2,3);
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(3,3);
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(4,3);
INSERT INTO "Lighting_Technicians_Fixtures" VALUES(5,3);
CREATE TABLE Roles (
    roleID INTEGER PRIMARY KEY AUTOINCREMENT,
    filmProjectID INTEGER,
    actorID INTEGER,
    FOREIGN KEY (filmProjectID) REFERENCES FilmProjects(projectID),
    FOREIGN KEY (actorID) REFERENCES Actors(actorID)
);
INSERT INTO "Roles" VALUES(1,1,1);
INSERT INTO "Roles" VALUES(2,2,2);
INSERT INTO "Roles" VALUES(3,1,3);
INSERT INTO "Roles" VALUES(4,2,4);
INSERT INTO "Roles" VALUES(5,1,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('FilmProjects',2);
INSERT INTO "sqlite_sequence" VALUES('Directors',2);
INSERT INTO "sqlite_sequence" VALUES('Cinematographers',2);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Roles',5);
INSERT INTO "sqlite_sequence" VALUES('Lighting_Technicians',5);
COMMIT;
