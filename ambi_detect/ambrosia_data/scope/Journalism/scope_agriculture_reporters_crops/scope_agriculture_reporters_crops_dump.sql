BEGIN TRANSACTION;
CREATE TABLE Agriculture_Reporters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Agriculture_Reporters" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Agriculture_Reporters" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Agriculture_Reporters" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Agriculture_Reporters" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Agriculture_Reporters" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Agriculture_Reporters_Crops (
    agriculture_reporter_id INTEGER,
    crop_id INTEGER,
    FOREIGN KEY(agriculture_reporter_id) REFERENCES Agriculture_Reporters(id),
    FOREIGN KEY(crop_id) REFERENCES Crops(id),
    PRIMARY KEY(agriculture_reporter_id, crop_id)
);
INSERT INTO "Agriculture_Reporters_Crops" VALUES(1,1);
INSERT INTO "Agriculture_Reporters_Crops" VALUES(2,1);
INSERT INTO "Agriculture_Reporters_Crops" VALUES(3,2);
INSERT INTO "Agriculture_Reporters_Crops" VALUES(4,3);
INSERT INTO "Agriculture_Reporters_Crops" VALUES(5,4);
INSERT INTO "Agriculture_Reporters_Crops" VALUES(3,1);
INSERT INTO "Agriculture_Reporters_Crops" VALUES(4,1);
INSERT INTO "Agriculture_Reporters_Crops" VALUES(5,1);
CREATE TABLE Crops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_name TEXT,
    description TEXT
);
INSERT INTO "Crops" VALUES(1,'Corn','A cereal grain planted worldwide for its starchy endosperm.');
INSERT INTO "Crops" VALUES(2,'Wheat','A cereal grain widely grown for its seed.');
INSERT INTO "Crops" VALUES(3,'Rice','An edible seed of a plant, used as food.');
INSERT INTO "Crops" VALUES(4,'Barley','A cereal grain that is often processed into flour.');
INSERT INTO "Crops" VALUES(5,'Soybean','A legume native to East Asia, cultivated for its protein-rich seeds.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Agriculture_Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Crops',5);
COMMIT;
