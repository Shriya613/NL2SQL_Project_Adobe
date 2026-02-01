BEGIN TRANSACTION;
CREATE TABLE "Artifacts" (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Artifacts_Uncovered TEXT,
    Ancient_Remains TEXT,
    Archaeological_Site TEXT,
    EstimatedAge INT
);
INSERT INTO "Artifacts" VALUES(1,'Pottery shards','Fragment of a vase','Riverbend Park',200);
INSERT INTO "Artifacts" VALUES(2,'Animal bones','Fossilized teeth','Forest Grove',1500);
INSERT INTO "Artifacts" VALUES(3,'Stone tools','Chipped stone points','Mountain View',10000);
INSERT INTO "Artifacts" VALUES(4,'Clay figurines','Broken pieces','Valley Ridge',3000);
INSERT INTO "Artifacts" VALUES(5,'Beads','Glass beads','Hillsborough',500);
CREATE TABLE Equipment (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT,
    SerialNumber TEXT,
    PurchaseDate DATE
);
INSERT INTO "Equipment" VALUES(1,'Excavator','ABC123','2020-01-01');
INSERT INTO "Equipment" VALUES(2,'Bulldozer','DEF456','2019-06-15');
INSERT INTO "Equipment" VALUES(3,'Crane','GHI789','2021-03-20');
INSERT INTO "Equipment" VALUES(4,'Backhoe','JKL012','2018-09-10');
INSERT INTO "Equipment" VALUES(5,'Dump Truck','MNO345','2020-04-05');
CREATE TABLE Workers (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    ContactNumber TEXT,
    SkillLevel TEXT
);
INSERT INTO "Workers" VALUES(1,'John Doe','123-456-7890','Intermediate');
INSERT INTO "Workers" VALUES(2,'Jane Smith','987-654-3210','Expert');
INSERT INTO "Workers" VALUES(3,'Mike Johnson','555-123-4567','Beginner');
INSERT INTO "Workers" VALUES(4,'Emily Davis','555-987-6543','Advanced');
INSERT INTO "Workers" VALUES(5,'David Brown','555-765-4321','Intermediate');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Workers',5);
INSERT INTO "sqlite_sequence" VALUES('Equipment',5);
INSERT INTO "sqlite_sequence" VALUES('Artifacts',5);
COMMIT;
