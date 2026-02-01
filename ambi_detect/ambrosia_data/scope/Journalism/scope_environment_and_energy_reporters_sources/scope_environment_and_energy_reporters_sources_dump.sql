BEGIN TRANSACTION;
CREATE TABLE Environment_And_Energy_Reporters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Environment_And_Energy_Reporters" VALUES(1,'John Doe','john.doe@example.com');
INSERT INTO "Environment_And_Energy_Reporters" VALUES(2,'Jane Smith','jane.smith@example.com');
INSERT INTO "Environment_And_Energy_Reporters" VALUES(3,'Michael Johnson','michael.johnson@example.com');
INSERT INTO "Environment_And_Energy_Reporters" VALUES(4,'Emily Davis','emily.davis@example.com');
INSERT INTO "Environment_And_Energy_Reporters" VALUES(5,'David Brown','david.brown@example.com');
CREATE TABLE Environment_And_Energy_Reporters_Sources (
    reporter_id INTEGER,
    source_id INTEGER,
    FOREIGN KEY(reporter_id) REFERENCES Environment_And_Energy_Reporters(id),
    FOREIGN KEY(source_id) REFERENCES Sources(source_id)
);
INSERT INTO "Environment_And_Energy_Reporters_Sources" VALUES(1,1);
INSERT INTO "Environment_And_Energy_Reporters_Sources" VALUES(2,1);
INSERT INTO "Environment_And_Energy_Reporters_Sources" VALUES(3,1);
INSERT INTO "Environment_And_Energy_Reporters_Sources" VALUES(4,1);
INSERT INTO "Environment_And_Energy_Reporters_Sources" VALUES(5,1);
INSERT INTO "Environment_And_Energy_Reporters_Sources" VALUES(5,2);
CREATE TABLE Sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT,
    description TEXT
);
INSERT INTO "Sources" VALUES(1,'Solar_Power','A source on solar power technology and its applications.');
INSERT INTO "Sources" VALUES(2,'Wind_Energy','A source on wind energy technology and its applications.');
INSERT INTO "Sources" VALUES(3,'Hydroelectricity','A source on hydroelectricity technology and its applications.');
INSERT INTO "Sources" VALUES(4,'Nuclear_Power','A source on nuclear power technology and its applications.');
INSERT INTO "Sources" VALUES(5,'Bioenergy','A source on bioenergy technology and its applications.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Environment_And_Energy_Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Sources',5);
COMMIT;
