BEGIN TRANSACTION;
CREATE TABLE Actors (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT ,
    Roles INT,
    Nominations TEXT);
INSERT INTO "Actors" VALUES(1,'Samuel Wright',45,50,'10');
INSERT INTO "Actors" VALUES(2,'Amanda Miller',30,40,'8');
INSERT INTO "Actors" VALUES(3,'David Lee',50,60,'12');
CREATE TABLE Cinematographers (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT ,
    Movies_Cinematographer INT,
    Techniques TEXT);
INSERT INTO "Cinematographers" VALUES(1,'Peter James',45,20,'High Dynamic Range');
INSERT INTO "Cinematographers" VALUES(2,'Lucas Nelson',30,15,'Time Lapse Photography');
INSERT INTO "Cinematographers" VALUES(3,'Charles Stone',40,25,'Wide Angle Lens');
CREATE TABLE Costume_Designers (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT ,
    Experience INT,
    Famous_For TEXT);
INSERT INTO "Costume_Designers" VALUES(1,'Emily Johnson',45,18,'Oscar Winning Design');
INSERT INTO "Costume_Designers" VALUES(2,'Lisa White',35,17,'Versace Collection');
INSERT INTO "Costume_Designers" VALUES(3,'Robert Black',40,20,'Dior');
CREATE TABLE Directors (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT ,
    Films_Directed INT,
    Genre TEXT);
INSERT INTO "Directors" VALUES(1,'William Taylor',45,5,'Drama');
INSERT INTO "Directors" VALUES(2,'Anna Thompson',30,6,'Comedy');
INSERT INTO "Directors" VALUES(3,'Daniel Davis',50,10,'Thriller');
CREATE TABLE Producers (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT ,
    Budget INT,
    Projects_Produced INT);
INSERT INTO "Producers" VALUES(1,'Elizabeth Green',45,100000000,15);
INSERT INTO "Producers" VALUES(2,'Alexander King',30,50000000,10);
INSERT INTO "Producers" VALUES(3,'Richard Queen',40,150000000,20);
CREATE TABLE Script_Supervisors (
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT ,
    Experience INT,
    Awards TEXT);
INSERT INTO "Script_Supervisors" VALUES(1,'John Smith',45,20,'Academy Award');
INSERT INTO "Script_Supervisors" VALUES(2,'Jane Doe',30,15,'Golden Globe');
INSERT INTO "Script_Supervisors" VALUES(3,'Mike Brown',50,25,'BAFTA');
COMMIT;
