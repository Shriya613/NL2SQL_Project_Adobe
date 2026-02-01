BEGIN TRANSACTION;
CREATE TABLE Actor(
    actor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT,
    lastname TEXT);
INSERT INTO "Actor" VALUES(1,'Tim','Robin');
INSERT INTO "Actor" VALUES(2,'Marlon','Brando');
INSERT INTO "Actor" VALUES(3,'Brad','Pitt');
INSERT INTO "Actor" VALUES(4,'Tom','Hanks');
INSERT INTO "Actor" VALUES(5,'Jack','Nicholson');
CREATE TABLE Archived_Films(
    film_id INTEGER REFERENCES Films(id),
    archiver_id INTEGER REFERENCES Film_Archivists(id),
    preservation_method_id INTEGER REFERENCES Preservation_Methods(id));
INSERT INTO "Archived_Films" VALUES(NULL,1,1);
INSERT INTO "Archived_Films" VALUES(NULL,2,1);
INSERT INTO "Archived_Films" VALUES(NULL,3,1);
INSERT INTO "Archived_Films" VALUES(NULL,3,2);
CREATE TABLE Director(
    director_id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT,
    lastname TEXT);
INSERT INTO "Director" VALUES(1,'Frank','Darabont');
INSERT INTO "Director" VALUES(2,'Mario','Puzo');
INSERT INTO "Director" VALUES(3,'Quentin','Tarantino');
INSERT INTO "Director" VALUES(4,'Steven','Spielberg');
INSERT INTO "Director" VALUES(5,'Robert','Zemeckis');
CREATE TABLE Film_Archivists(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Film_Archivists" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Film_Archivists" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Film_Archivists" VALUES(3,'Michael Johnson','michael@example.com');
CREATE TABLE Films(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    year REAL  );
INSERT INTO "Films" VALUES(1,'The Shawshank Redemption',1994.0);
INSERT INTO "Films" VALUES(2,'The Godfather',1972.0);
INSERT INTO "Films" VALUES(3,'Pulp Fiction',1994.0);
INSERT INTO "Films" VALUES(4,'Schindler''s List',1993.0);
INSERT INTO "Films" VALUES(5,'Forrest Gump',1994.0);
CREATE TABLE Preservation_Methods(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    method TEXT,
    description TEXT);
INSERT INTO "Preservation_Methods" VALUES(1,'Film Negatives','Storing physical copies of films on celluloid.');
INSERT INTO "Preservation_Methods" VALUES(2,'Digital Files','Storing digital copies of films on hard drives or cloud storage.');
INSERT INTO "Preservation_Methods" VALUES(3,'Blu-ray Discs','Storing high definition video on Blu-ray discs.');
INSERT INTO "Preservation_Methods" VALUES(4,'DVDs','Storing standard definition video on DVDs.');
CREATE TABLE Producer(
    producer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstname TEXT,
    lastname TEXT);
INSERT INTO "Producer" VALUES(1,'Albert','Ruddy');
INSERT INTO "Producer" VALUES(2,'Gordon','Carroll');
INSERT INTO "Producer" VALUES(3,'Barry','Winkler');
INSERT INTO "Producer" VALUES(4,'Joel','Silver');
INSERT INTO "Producer" VALUES(5,'Mark','Harrison');
CREATE TABLE Roles(
    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
    film_id INTEGER REFERENCES Films(id),
    director_id INTEGER REFERENCES Director(director_id),
    producer_id INTEGER REFERENCES Producer(producer_id),
    actor_id INTEGER REFERENCES Actor(actor_id));
INSERT INTO "Roles" VALUES(1,1,2,3,4);
INSERT INTO "Roles" VALUES(2,2,1,4,5);
INSERT INTO "Roles" VALUES(3,3,6,5,7);
INSERT INTO "Roles" VALUES(4,4,8,7,8);
INSERT INTO "Roles" VALUES(5,5,9,8,9);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Film_Archivists',3);
INSERT INTO "sqlite_sequence" VALUES('Preservation_Methods',4);
INSERT INTO "sqlite_sequence" VALUES('Films',5);
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Producer',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
INSERT INTO "sqlite_sequence" VALUES('Roles',5);
COMMIT;
