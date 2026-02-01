BEGIN TRANSACTION;
CREATE TABLE Data_Analysts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    years_of_experience INTEGER 
);
INSERT INTO "Data_Analysts" VALUES(1,'Alice',3);
INSERT INTO "Data_Analysts" VALUES(2,'Bob',2);
INSERT INTO "Data_Analysts" VALUES(3,'Carol',6);
INSERT INTO "Data_Analysts" VALUES(4,'David',8);
INSERT INTO "Data_Analysts" VALUES(5,'Eve',10);
CREATE TABLE Data_Analysts_Techniques(
    data_analyst_id INTEGER,
    technique_id INTEGER,
    FOREIGN KEY(data_analyst_id) REFERENCES Data_Analysts(id),
    FOREIGN KEY(technique_id) REFERENCES Techniques(id)
);
INSERT INTO "Data_Analysts_Techniques" VALUES(1,1);
INSERT INTO "Data_Analysts_Techniques" VALUES(2,1);
INSERT INTO "Data_Analysts_Techniques" VALUES(3,1);
INSERT INTO "Data_Analysts_Techniques" VALUES(4,1);
INSERT INTO "Data_Analysts_Techniques" VALUES(5,1);
INSERT INTO "Data_Analysts_Techniques" VALUES(5,4);
CREATE TABLE Techniques(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    technique_name TEXT
);
INSERT INTO "Techniques" VALUES(1,'Regression Analysis');
INSERT INTO "Techniques" VALUES(2,'Python Programming');
INSERT INTO "Techniques" VALUES(3,'SQL Query Writing');
INSERT INTO "Techniques" VALUES(4,'R Language');
INSERT INTO "Techniques" VALUES(5,'Tableau Dashboard Creation');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Data_Analysts',5);
INSERT INTO "sqlite_sequence" VALUES('Techniques',5);
COMMIT;
