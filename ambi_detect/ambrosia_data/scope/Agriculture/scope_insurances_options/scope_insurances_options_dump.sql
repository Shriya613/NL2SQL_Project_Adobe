BEGIN TRANSACTION;
CREATE TABLE Insurance_Options(
    insurance_id INTEGER,
    option_id INTEGER,
    FOREIGN KEY (insurance_id) REFERENCES Insurances(id),
    FOREIGN KEY (option_id) REFERENCES Options(id),
    PRIMARY KEY (insurance_id, option_id)
);
INSERT INTO "Insurance_Options" VALUES(1,1);
INSERT INTO "Insurance_Options" VALUES(2,1);
INSERT INTO "Insurance_Options" VALUES(3,1);
INSERT INTO "Insurance_Options" VALUES(4,1);
INSERT INTO "Insurance_Options" VALUES(5,1);
INSERT INTO "Insurance_Options" VALUES(5,4);
CREATE TABLE Insurances(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_holder TEXT,
    premium REAL 
);
INSERT INTO "Insurances" VALUES(1,'John Doe',1000.0);
INSERT INTO "Insurances" VALUES(2,'Jane Smith',2000.0);
INSERT INTO "Insurances" VALUES(3,'Mike Johnson',3000.0);
INSERT INTO "Insurances" VALUES(4,'Emily Brown',4000.0);
INSERT INTO "Insurances" VALUES(5,'Paul Davis',5000.0);
CREATE TABLE Options(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    option_name TEXT
);
INSERT INTO "Options" VALUES(1,'Revenue_Insurance');
INSERT INTO "Options" VALUES(2,'Liability_Insurance');
INSERT INTO "Options" VALUES(3,'Crop_Insurance');
INSERT INTO "Options" VALUES(4,'Equipment_Insurance');
INSERT INTO "Options" VALUES(5,'Worker_sCompensation');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Insurances',5);
INSERT INTO "sqlite_sequence" VALUES('Options',5);
COMMIT;
