BEGIN TRANSACTION;
CREATE TABLE Stadiums (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Seating_Capacity INT,
    Maximum_Occupancy INT,
    Name TEXT,
    Location TEXT,
    Year_Built INT,
    Type_of_Stadium TEXT
);
INSERT INTO "Stadiums" VALUES(1,70000,80000,'Wembley Stadium','London',2003,'Football');
INSERT INTO "Stadiums" VALUES(2,65000,75000,'Old Trafford','Manchester',1910,'Football');
INSERT INTO "Stadiums" VALUES(3,50000,60000,'Etihad Stadium','Manchester',2003,'Football');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Stadiums',3);
COMMIT;
