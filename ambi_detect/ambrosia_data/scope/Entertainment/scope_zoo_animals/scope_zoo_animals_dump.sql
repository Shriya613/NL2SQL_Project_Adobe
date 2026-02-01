BEGIN TRANSACTION;
CREATE TABLE Animals (
    animalID INTEGER PRIMARY KEY AUTOINCREMENT,
    animalName TEXT);
INSERT INTO "Animals" VALUES(1,'Elephant');
INSERT INTO "Animals" VALUES(2,'Lion');
INSERT INTO "Animals" VALUES(3,'Tiger');
INSERT INTO "Animals" VALUES(4,'Giraffe');
INSERT INTO "Animals" VALUES(5,'Rhino');
INSERT INTO "Animals" VALUES(6,'Penguin');
INSERT INTO "Animals" VALUES(7,'Kangaroo');
INSERT INTO "Animals" VALUES(8,'Flamingo');
INSERT INTO "Animals" VALUES(9,'Peacock');
INSERT INTO "Animals" VALUES(10,'Bear');
CREATE TABLE Zoo (
    zooID INTEGER PRIMARY KEY AUTOINCREMENT,
    zooName TEXT,
    location TEXT);
INSERT INTO "Zoo" VALUES(1,'Wildlife World','New York');
INSERT INTO "Zoo" VALUES(2,'Animal Kingdom','Los Angeles');
INSERT INTO "Zoo" VALUES(3,'Jungle Safari','Chicago');
INSERT INTO "Zoo" VALUES(4,'Exotic Creatures','Miami');
INSERT INTO "Zoo" VALUES(5,'African Savannah','Houston');
CREATE TABLE Zoo_Animals (
    zooID INTEGER,
    animalID INTEGER,
    FOREIGN KEY(zooID) REFERENCES Zoo(zooID),
    FOREIGN KEY(animalID) REFERENCES Animals(animalID));
INSERT INTO "Zoo_Animals" VALUES(1,1);
INSERT INTO "Zoo_Animals" VALUES(2,2);
INSERT INTO "Zoo_Animals" VALUES(3,3);
INSERT INTO "Zoo_Animals" VALUES(4,4);
INSERT INTO "Zoo_Animals" VALUES(5,5);
INSERT INTO "Zoo_Animals" VALUES(1,6);
INSERT INTO "Zoo_Animals" VALUES(2,7);
INSERT INTO "Zoo_Animals" VALUES(3,8);
INSERT INTO "Zoo_Animals" VALUES(4,9);
INSERT INTO "Zoo_Animals" VALUES(5,10);
INSERT INTO "Zoo_Animals" VALUES(2,1);
INSERT INTO "Zoo_Animals" VALUES(3,1);
INSERT INTO "Zoo_Animals" VALUES(4,1);
INSERT INTO "Zoo_Animals" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Animals',10);
INSERT INTO "sqlite_sequence" VALUES('Zoo',5);
COMMIT;
