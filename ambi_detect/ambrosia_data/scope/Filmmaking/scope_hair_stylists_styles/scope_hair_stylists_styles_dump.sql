BEGIN TRANSACTION;
CREATE TABLE Actors (
    actor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    birth_date DATE
);
INSERT INTO "Actors" VALUES(1,'Leonardo','DiCaprio','1974-11-11');
INSERT INTO "Actors" VALUES(2,'Ellen','Page','1987-01-21');
INSERT INTO "Actors" VALUES(3,'Christian','Bale','1974-01-30');
INSERT INTO "Actors" VALUES(4,'Anne','Hathaway','1982-11-12');
INSERT INTO "Actors" VALUES(5,'Natalie','Portman','1981-06-09');
CREATE TABLE Casting (
    casting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id INTEGER,
    role_id INTEGER,
    FOREIGN KEY(actor_id) REFERENCES Actors(actor_id),
    FOREIGN KEY(role_id) REFERENCES Roles(role_id)
);
INSERT INTO "Casting" VALUES(1,1,1);
INSERT INTO "Casting" VALUES(2,2,2);
INSERT INTO "Casting" VALUES(3,3,3);
INSERT INTO "Casting" VALUES(4,4,4);
INSERT INTO "Casting" VALUES(5,5,5);
CREATE TABLE Film_Projects (
    project_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE
);
INSERT INTO "Film_Projects" VALUES(1,'The Great Gatsby','2022-05-29');
INSERT INTO "Film_Projects" VALUES(2,'Inception','2010-07-16');
INSERT INTO "Film_Projects" VALUES(3,'Interstellar','2014-11-07');
INSERT INTO "Film_Projects" VALUES(4,'La La Land','2016-12-02');
INSERT INTO "Film_Projects" VALUES(5,'Black Swan','2010-12-01');
CREATE TABLE Hair_Stylists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INT
);
INSERT INTO "Hair_Stylists" VALUES(1,'John Doe',10);
INSERT INTO "Hair_Stylists" VALUES(2,'Jane Smith',8);
INSERT INTO "Hair_Stylists" VALUES(3,'Emily Johnson',6);
INSERT INTO "Hair_Stylists" VALUES(4,'Michael Brown',3);
INSERT INTO "Hair_Stylists" VALUES(5,'Linda Davis',7);
CREATE TABLE Hair_Stylists_Styles (
    hairdresser_id INTEGER,
    style_id INTEGER,
    FOREIGN KEY(hairdresser_id) REFERENCES Hair_Stylists(id),
    FOREIGN KEY(style_id) REFERENCES Styles(style_id)
);
INSERT INTO "Hair_Stylists_Styles" VALUES(1,1);
INSERT INTO "Hair_Stylists_Styles" VALUES(2,1);
INSERT INTO "Hair_Stylists_Styles" VALUES(3,1);
INSERT INTO "Hair_Stylists_Styles" VALUES(4,1);
INSERT INTO "Hair_Stylists_Styles" VALUES(5,1);
INSERT INTO "Hair_Stylists_Styles" VALUES(5,4);
CREATE TABLE Roles (
    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_name TEXT,
    film_project_id INTEGER,
    FOREIGN KEY(film_project_id) REFERENCES Film_Projects(project_id)
);
INSERT INTO "Roles" VALUES(1,'Gatsby',1);
INSERT INTO "Roles" VALUES(2,'Cobb',2);
INSERT INTO "Roles" VALUES(3,'Cooper',3);
INSERT INTO "Roles" VALUES(4,'Sebastian',4);
INSERT INTO "Roles" VALUES(5,'Arthur',5);
CREATE TABLE Styles (
    style_id INTEGER PRIMARY KEY AUTOINCREMENT,
    style_name TEXT,
    description TEXT
);
INSERT INTO "Styles" VALUES(1,'Period Hairstyle','A hairstyle from a specific historical period.');
INSERT INTO "Styles" VALUES(2,'Updo','A hairstyle where the hair is pulled up off the neck and secured at the top or back of the head.');
INSERT INTO "Styles" VALUES(3,'Bob','A short haircut that falls just below the chin.');
INSERT INTO "Styles" VALUES(4,'Pixie','A very short haircut with hair cut close to the scalp.');
INSERT INTO "Styles" VALUES(5,'Curl','A hairstyle characterized by curls or waves in the hair.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hair_Stylists',5);
INSERT INTO "sqlite_sequence" VALUES('Styles',5);
INSERT INTO "sqlite_sequence" VALUES('Film_Projects',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
INSERT INTO "sqlite_sequence" VALUES('Roles',5);
INSERT INTO "sqlite_sequence" VALUES('Casting',5);
COMMIT;
