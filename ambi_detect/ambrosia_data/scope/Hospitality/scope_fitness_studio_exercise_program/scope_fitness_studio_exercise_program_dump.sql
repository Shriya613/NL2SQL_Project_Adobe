BEGIN TRANSACTION;
CREATE TABLE Exercise_Program(
    ProgramID INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT);
INSERT INTO "Exercise_Program" VALUES(1,'Yoga');
INSERT INTO "Exercise_Program" VALUES(2,'Aerobics');
INSERT INTO "Exercise_Program" VALUES(3,'Weightlifting');
INSERT INTO "Exercise_Program" VALUES(4,'Pilates');
INSERT INTO "Exercise_Program" VALUES(5,'Zumba');
CREATE TABLE Fitness_Studio(
    StudioID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Location TEXT);
INSERT INTO "Fitness_Studio" VALUES(1,'Studio A','Downtown');
INSERT INTO "Fitness_Studio" VALUES(2,'Studio B','Uptown');
INSERT INTO "Fitness_Studio" VALUES(3,'Studio C','Suburb');
INSERT INTO "Fitness_Studio" VALUES(4,'Studio D','Rural');
INSERT INTO "Fitness_Studio" VALUES(5,'Studio E','Beachside');
CREATE TABLE Fitness_Studio_Exercise_Program(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    FitnessStudioID INTEGER,
    ExerciseProgramID INTEGER,
    FOREIGN KEY(FitnessStudioID) REFERENCES Fitness_Studio(StudioID),
    FOREIGN KEY(ExerciseProgramID) REFERENCES Exercise_Program(ProgramID));
INSERT INTO "Fitness_Studio_Exercise_Program" VALUES(1,1,1);
INSERT INTO "Fitness_Studio_Exercise_Program" VALUES(2,2,1);
INSERT INTO "Fitness_Studio_Exercise_Program" VALUES(3,3,1);
INSERT INTO "Fitness_Studio_Exercise_Program" VALUES(4,4,1);
INSERT INTO "Fitness_Studio_Exercise_Program" VALUES(5,5,1);
INSERT INTO "Fitness_Studio_Exercise_Program" VALUES(6,5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Fitness_Studio',5);
INSERT INTO "sqlite_sequence" VALUES('Exercise_Program',5);
INSERT INTO "sqlite_sequence" VALUES('Fitness_Studio_Exercise_Program',6);
COMMIT;
