BEGIN TRANSACTION;
CREATE TABLE Actor (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Date_Of_Birth DATE,
    Occupation TEXT);
INSERT INTO "Actor" VALUES(1,'Leonardo DiCaprio','11/11/1974','Actor');
INSERT INTO "Actor" VALUES(2,'Brad Pitt','18/12/1963','Actor');
INSERT INTO "Actor" VALUES(3,'Tom Hanks','09/07/1956','Actor');
INSERT INTO "Actor" VALUES(4,'Kate Winslet','05/10/1975','Actress');
INSERT INTO "Actor" VALUES(5,'Meryl Streep','22/06/1949','Actress');
CREATE TABLE Camera (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Brand_Name TEXT,
    Camera_Type TEXT,
    Cinematographer TEXT,
    Purchase_Year INTEGER,
    Currently_In_Use BOOLEAN DEFAULT FALSE);
INSERT INTO "Camera" VALUES(1,'Canon','DSLR','Janet Davis',2018,0);
INSERT INTO "Camera" VALUES(2,'Nikon','Mirrorless','Michael Smith',2020,0);
INSERT INTO "Camera" VALUES(3,'Panasonic','Hybrid','Jennifer Johnson',2019,0);
INSERT INTO "Camera" VALUES(4,'Sony','Mirrorless','David Brown',2018,0);
INSERT INTO "Camera" VALUES(5,'Fujifilm','Mirrorless','Emma Wilson',2018,0);
CREATE TABLE Director (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Birthdate DATE,
    Nationality TEXT);
INSERT INTO "Director" VALUES(1,'Christopher Nolan','20/07/1970','American');
INSERT INTO "Director" VALUES(2,'Quentin Tarantino','27/03/1963','American');
INSERT INTO "Director" VALUES(3,'Martin Scorsese','17/11/1942','American');
INSERT INTO "Director" VALUES(4,'Steven Spielberg','18/12/1946','American');
INSERT INTO "Director" VALUES(5,'Guillermo del Toro','09/10/1964','Mexican');
CREATE TABLE Film_Crew (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Role TEXT,
    Responsibility TEXT,
    Salary REAL);
INSERT INTO "Film_Crew" VALUES(1,'Producer','Overseeing all aspects of film production',100000.0);
INSERT INTO "Film_Crew" VALUES(2,'Editor','Assembling and fine-tuning footage',75000.0);
INSERT INTO "Film_Crew" VALUES(3,'Sound Designer','Creating sound effects and music score',60000.0);
INSERT INTO "Film_Crew" VALUES(4,'Screenwriter','Writing the screenplay',60000.0);
INSERT INTO "Film_Crew" VALUES(5,'Costume Designer','Designing and creating costumes for characters',50000.0);
CREATE TABLE Movie (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Release_Date DATE,
    Genre TEXT,
    Runtime INTEGER,
    Box_Office REAL);
INSERT INTO "Movie" VALUES(1,'The Dark Knight','18/07/2008','Action/Thriller',152,1004934398.0);
INSERT INTO "Movie" VALUES(2,'Pulp Fiction','14/10/1994','Crime/Drama',154,213842234.0);
INSERT INTO "Movie" VALUES(3,'Schindler''s List','20/12/1993','Biographical Drama',195,321171163.0);
INSERT INTO "Movie" VALUES(4,'Forrest Gump','06/07/1994','Romantic Comedy/Drama',142,677339259.0);
INSERT INTO "Movie" VALUES(5,'Parasite','08/05/2019','Thriller/Comedy',132,259801256.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Camera',5);
INSERT INTO "sqlite_sequence" VALUES('Director',5);
INSERT INTO "sqlite_sequence" VALUES('Movie',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
INSERT INTO "sqlite_sequence" VALUES('Film_Crew',5);
COMMIT;
